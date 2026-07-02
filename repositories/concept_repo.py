from __future__ import annotations
import logging
from uuid import UUID, uuid4
from neo4j import AsyncDriver
from core.constants import INTEREST_INIT, PROFESSIONAL_INIT, VECTOR_DIM

logger = logging.getLogger(__name__)


class ConceptRepository:
    def __init__(self, driver: AsyncDriver):
        self.driver = driver

    async def create_vector_index(self, dim: int = VECTOR_DIM) -> None:
        await self.driver.execute_query(
            """
            CREATE VECTOR INDEX concept_q_vector IF NOT EXISTS
            FOR (c:ConceptNode) ON c.q_vector
            OPTIONS { indexConfig: { `vector.dimensions`: $dim, `vector.similarity_function`: 'cosine' } }
            """,
            dim=dim,
        )

    # ── 兩階段向量粗精篩（Two-Stage Retrieval）───────────────────────────────────
    # Stage-1：對每個 query concept 向量呼叫 Neo4j Vector Index 做 KNN 粗篩，
    #          取代「拉全庫進 Python 記憶體再雙迴圈比對」的 O(N*M) 作法。
    # Stage-2：只對粗篩候選的少量 ConceptNode 抓取 EFFECTIVE 邊，交給
    #          concept_engine.compute_match_score() 做精細對齊評分。

    async def _vector_candidate_ids(
        self, query_vectors: list[list[float]], top_k: int
    ) -> set[str] | None:
        """回傳候選 ConceptNode id 聯集；索引不可用時回傳 None（呼叫端應 fallback 全表掃描）。"""
        if not query_vectors:
            return set()
        candidate_ids: set[str] = set()
        try:
            for vec in query_vectors:
                result = await self.driver.execute_query(
                    """
                    CALL db.index.vector.queryNodes('concept_q_vector', $top_k, $vector)
                    YIELD node, score
                    RETURN node.id AS id
                    """,
                    top_k=top_k, vector=vec,
                )
                candidate_ids.update(r["id"] for r in result.records)
        except Exception as e:
            logger.warning(f"Vector index 粗篩失敗，將 fallback 全表掃描：{e}")
            return None
        return candidate_ids

    async def get_kgs_concepts_for_query(
        self, query_vectors: list[list[float]], top_k: int = 100
    ) -> dict[UUID, list[dict]] | None:
        """兩階段版 get_all_kgs_concepts()：僅取粗篩候選概念的 KG EFFECTIVE 邊。回傳 None 代表需 fallback。"""
        candidate_ids = await self._vector_candidate_ids(query_vectors, top_k)
        if candidate_ids is None:
            return None
        if not candidate_ids:
            return {}
        result = await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph)-[e:EFFECTIVE]->(c:ConceptNode)
            WHERE c.id IN $ids
            RETURN kg.id AS kg_id, c.id AS concept_id, c.name AS name,
                   c.q_vector AS q_vector,
                   e.interest_score AS interest_score,
                   e.professional_score AS professional_score
            """,
            ids=list(candidate_ids),
        )
        result_map: dict[UUID, list[dict]] = {}
        for r in result.records:
            kg_id = UUID(r["kg_id"])
            result_map.setdefault(kg_id, []).append(dict(r))
        return result_map

    async def get_public_kgs_concepts_for_query(
        self, query_vectors: list[list[float]], top_k: int = 100
    ) -> dict[UUID, list[dict]] | None:
        """兩階段版 get_public_kgs_concepts()（World Agent 專用）。回傳 None 代表需 fallback。"""
        candidate_ids = await self._vector_candidate_ids(query_vectors, top_k)
        if candidate_ids is None:
            return None
        if not candidate_ids:
            return {}
        result = await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {is_public: true})-[e:EFFECTIVE]->(c:ConceptNode)
            WHERE c.id IN $ids
            RETURN kg.id AS kg_id, c.id AS concept_id, c.name AS name,
                   c.q_vector AS q_vector,
                   e.interest_score AS interest_score,
                   e.professional_score AS professional_score
            """,
            ids=list(candidate_ids),
        )
        result_map: dict[UUID, list[dict]] = {}
        for r in result.records:
            kg_id = UUID(r["kg_id"])
            result_map.setdefault(kg_id, []).append(dict(r))
        return result_map

    async def get_documents_concepts_for_query(
        self,
        query_vectors: list[list[float]],
        top_k: int = 100,
        exclude_doc_ids: list[UUID] | None = None,
    ) -> dict[UUID, list[dict]] | None:
        """兩階段版 get_all_documents_concepts()。回傳 None 代表需 fallback。"""
        candidate_ids = await self._vector_candidate_ids(query_vectors, top_k)
        if candidate_ids is None:
            return None
        if not candidate_ids:
            return {}
        exclude = [str(d) for d in exclude_doc_ids] if exclude_doc_ids else []
        result = await self.driver.execute_query(
            """
            MATCH (d:Document)-[e:EFFECTIVE]->(c:ConceptNode)
            WHERE c.id IN $ids AND NOT d.id IN $exclude
            RETURN d.id AS doc_id, c.id AS concept_id, c.name AS name,
                   c.q_vector AS q_vector,
                   e.interest_score AS interest_score,
                   e.professional_score AS professional_score
            """,
            ids=list(candidate_ids), exclude=exclude,
        )
        result_map: dict[UUID, list[dict]] = {}
        for r in result.records:
            doc_id = UUID(r["doc_id"])
            result_map.setdefault(doc_id, []).append(dict(r))
        return result_map

    async def get_or_create(self, name: str, domain: str, q_vector) -> UUID:
        vec = q_vector.tolist() if hasattr(q_vector, "tolist") else list(q_vector)
        result = await self.driver.execute_query(
            """
            MERGE (c:ConceptNode {name: $name, domain: $domain})
            ON CREATE SET c.id = $id, c.q_vector = $q_vector
            RETURN c.id AS id
            """,
            name=name, domain=domain, q_vector=vec, id=str(uuid4()),
        )
        return UUID(result.records[0]["id"])

    async def init_document_concept(
        self, doc_id: UUID, name: str,
        interest: float = INTEREST_INIT,
        professional: float = PROFESSIONAL_INIT,
    ) -> None:
        await self.driver.execute_query(
            """
            MATCH (d:Document {id: $doc_id})
            MATCH (c:ConceptNode {name: $name})
            MERGE (d)-[r:IMPLICIT]->(c)
            SET r.interest_score = $i, r.professional_score = $p, r.updated_at = datetime()
            """,
            doc_id=str(doc_id), name=name, i=interest, p=professional,
        )

    async def sync_document_effective(self, doc_id: UUID) -> None:
        await self.driver.execute_query(
            """
            MATCH (d:Document {id: $doc_id})-[i:IMPLICIT]->(c:ConceptNode)
            MERGE (d)-[e:EFFECTIVE]->(c)
            SET e.interest_score = i.interest_score, e.professional_score = i.professional_score
            """,
            doc_id=str(doc_id),
        )
        await self.driver.execute_query(
            """
            MATCH (d:Document {id: $doc_id})-[e:EFFECTIVE]->(c:ConceptNode)
            WHERE NOT (d)-[:IMPLICIT]->(c)
            DELETE e
            """,
            doc_id=str(doc_id),
        )

    async def get_document_concepts(self, doc_id: UUID) -> list[dict]:
        result = await self.driver.execute_query(
            """
            MATCH (d:Document {id: $doc_id})-[e:EFFECTIVE]->(c:ConceptNode)
            RETURN c.id AS id, c.name AS name, c.domain AS domain,
                   c.q_vector AS q_vector,
                   e.interest_score AS interest_score,
                   e.professional_score AS professional_score
            ORDER BY (e.interest_score + e.professional_score) DESC
            """,
            doc_id=str(doc_id),
        )
        return [dict(r) for r in result.records]

    async def get_all_documents_concepts(
        self, exclude_doc_ids: list[UUID] | None = None
    ) -> dict[UUID, list[dict]]:
        exclude = [str(d) for d in exclude_doc_ids] if exclude_doc_ids else []
        result = await self.driver.execute_query(
            """
            MATCH (d:Document)-[e:EFFECTIVE]->(c:ConceptNode)
            WHERE NOT d.id IN $exclude
            RETURN d.id AS doc_id, c.id AS concept_id, c.name AS name,
                   c.q_vector AS q_vector,
                   e.interest_score AS interest_score,
                   e.professional_score AS professional_score
            """,
            exclude=exclude,
        )
        result_map: dict[UUID, list[dict]] = {}
        for r in result.records:
            doc_id = UUID(r["doc_id"])
            result_map.setdefault(doc_id, []).append(dict(r))
        return result_map

    # ── KG 路由層概念 ──────────────────────────────────────────────────────────

    async def init_kg_concept(
        self, kg_id: UUID, name: str,
        interest: float = INTEREST_INIT,
        professional: float = PROFESSIONAL_INIT,
    ) -> None:
        """建立 KnowledgeGraph → ConceptNode 的 IMPLICIT 邊。"""
        await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $kg_id})
            MATCH (c:ConceptNode {name: $name})
            MERGE (kg)-[r:IMPLICIT]->(c)
            SET r.interest_score = $i, r.professional_score = $p, r.updated_at = datetime()
            """,
            kg_id=str(kg_id), name=name, i=interest, p=professional,
        )

    async def sync_kg_effective(self, kg_id: UUID) -> None:
        """將 KG 的 IMPLICIT 邊同步為 EFFECTIVE 邊（用於路由比對）。"""
        await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $kg_id})-[i:IMPLICIT]->(c:ConceptNode)
            MERGE (kg)-[e:EFFECTIVE]->(c)
            SET e.interest_score = i.interest_score, e.professional_score = i.professional_score
            """,
            kg_id=str(kg_id),
        )
        await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $kg_id})-[e:EFFECTIVE]->(c:ConceptNode)
            WHERE NOT (kg)-[:IMPLICIT]->(c)
            DELETE e
            """,
            kg_id=str(kg_id),
        )

    async def get_kg_concepts(self, kg_id: UUID) -> list[dict]:
        """取得單個 KG 的所有 EFFECTIVE 概念，格式與 get_document_concepts 一致。"""
        result = await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $kg_id})-[e:EFFECTIVE]->(c:ConceptNode)
            RETURN c.id AS id, c.name AS name, c.domain AS domain,
                   c.q_vector AS q_vector,
                   e.interest_score AS interest_score,
                   e.professional_score AS professional_score
            ORDER BY (e.interest_score + e.professional_score) DESC
            """,
            kg_id=str(kg_id),
        )
        return [dict(r) for r in result.records]

    async def get_all_kgs_concepts(self) -> dict[UUID, list[dict]]:
        """取得所有 KG 的 EFFECTIVE 概念，供分配器與路由器批次比對。"""
        result = await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph)-[e:EFFECTIVE]->(c:ConceptNode)
            RETURN kg.id AS kg_id, c.id AS concept_id, c.name AS name,
                   c.q_vector AS q_vector,
                   e.interest_score AS interest_score,
                   e.professional_score AS professional_score
            """
        )
        result_map: dict[UUID, list[dict]] = {}
        for r in result.records:
            kg_id = UUID(r["kg_id"])
            result_map.setdefault(kg_id, []).append(dict(r))
        return result_map

    async def get_public_kgs_concepts(self) -> dict[UUID, list[dict]]:
        """取得所有 is_public=true 的 KG 的 EFFECTIVE 概念（World Agent 專用）。"""
        result = await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {is_public: true})-[e:EFFECTIVE]->(c:ConceptNode)
            RETURN kg.id AS kg_id, c.id AS concept_id, c.name AS name,
                   c.q_vector AS q_vector,
                   e.interest_score AS interest_score,
                   e.professional_score AS professional_score
            """
        )
        result_map: dict[UUID, list[dict]] = {}
        for r in result.records:
            kg_id = UUID(r["kg_id"])
            result_map.setdefault(kg_id, []).append(dict(r))
        return result_map

    async def get_all_concepts(self) -> list[dict]:
        result = await self.driver.execute_query(
            """
            MATCH (c:ConceptNode)
            OPTIONAL MATCH (d:Document)-[:EFFECTIVE]->(c)
            WITH c, count(d) AS doc_count
            RETURN c.id AS id, c.name AS name, c.domain AS domain, doc_count
            ORDER BY doc_count DESC
            """
        )
        return [dict(r) for r in result.records]
