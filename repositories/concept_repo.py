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

    async def vector_search_concept_ids(self, query_vector, top_k: int) -> list[str]:
        """二階段檢索 Stage-1（粗篩）：用 concept_q_vector 向量索引取回最相近的 ConceptNode id。

        由 Neo4j 底層執行 KNN 運算，取代 Python 端的全表雙迴圈比對。
        """
        vec = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        result = await self.driver.execute_query(
            """
            CALL db.index.vector.queryNodes('concept_q_vector', $top_k, $vector)
            YIELD node
            RETURN node.id AS id
            """,
            top_k=top_k, vector=vec,
        )
        return [r["id"] for r in result.records]

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
        self,
        exclude_doc_ids: list[UUID] | None = None,
        concept_ids: list[str] | None = None,
    ) -> dict[UUID, list[dict]]:
        """取得所有文件的 EFFECTIVE 概念。

        `concept_ids` 為二階段檢索 Stage-2（精篩）用的候選過濾——非 None 時只回傳
        該候選集合內的概念，將原本的全表掃描縮限為 Stage-1 向量粗篩後的子集。
        """
        exclude = [str(d) for d in exclude_doc_ids] if exclude_doc_ids else []
        result = await self.driver.execute_query(
            """
            MATCH (d:Document)-[e:EFFECTIVE]->(c:ConceptNode)
            WHERE NOT d.id IN $exclude
              AND ($concept_ids IS NULL OR c.id IN $concept_ids)
            RETURN d.id AS doc_id, c.id AS concept_id, c.name AS name,
                   c.q_vector AS q_vector,
                   e.interest_score AS interest_score,
                   e.professional_score AS professional_score
            """,
            exclude=exclude, concept_ids=concept_ids,
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

    async def get_all_kgs_concepts(self, concept_ids: list[str] | None = None) -> dict[UUID, list[dict]]:
        """取得所有 KG 的 EFFECTIVE 概念，供分配器與路由器批次比對。

        `concept_ids` 非 None 時限定回傳該候選集合內的概念（二階段檢索 Stage-2 用）。
        """
        result = await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph)-[e:EFFECTIVE]->(c:ConceptNode)
            WHERE $concept_ids IS NULL OR c.id IN $concept_ids
            RETURN kg.id AS kg_id, c.id AS concept_id, c.name AS name,
                   c.q_vector AS q_vector,
                   e.interest_score AS interest_score,
                   e.professional_score AS professional_score
            """,
            concept_ids=concept_ids,
        )
        result_map: dict[UUID, list[dict]] = {}
        for r in result.records:
            kg_id = UUID(r["kg_id"])
            result_map.setdefault(kg_id, []).append(dict(r))
        return result_map

    async def get_public_kgs_concepts(self, concept_ids: list[str] | None = None) -> dict[UUID, list[dict]]:
        """取得所有 is_public=true 的 KG 的 EFFECTIVE 概念（World Agent 專用）。

        `concept_ids` 非 None 時限定回傳該候選集合內的概念（二階段檢索 Stage-2 用）。
        """
        result = await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {is_public: true})-[e:EFFECTIVE]->(c:ConceptNode)
            WHERE $concept_ids IS NULL OR c.id IN $concept_ids
            RETURN kg.id AS kg_id, c.id AS concept_id, c.name AS name,
                   c.q_vector AS q_vector,
                   e.interest_score AS interest_score,
                   e.professional_score AS professional_score
            """,
            concept_ids=concept_ids,
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
