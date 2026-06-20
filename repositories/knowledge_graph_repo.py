from __future__ import annotations
import logging
from pathlib import Path
from uuid import UUID, uuid4
from datetime import datetime

from neo4j import AsyncDriver

from models.knowledge_graph import KnowledgeGraph, KnowledgeGraphDetail

logger = logging.getLogger(__name__)


class KnowledgeGraphRepository:
    def __init__(self, driver: AsyncDriver):
        self.driver = driver

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def create(
        self,
        name: str,
        description: str,
        folder_path: str,
        owner_id: str = "default",
        is_public: bool = True,
        db_name: str = "",
    ) -> KnowledgeGraph:
        kg_id = str(uuid4())
        now = datetime.utcnow().isoformat()
        await self.driver.execute_query(
            """
            CREATE (kg:KnowledgeGraph {
                id: $id, name: $name, description: $description,
                folder_path: $folder_path, owner_id: $owner_id,
                is_public: $is_public, db_name: $db_name,
                doc_count: 0, entity_count: 0, relation_count: 0,
                created_at: datetime($now), updated_at: datetime($now)
            })
            """,
            id=kg_id, name=name, description=description,
            folder_path=folder_path, owner_id=owner_id,
            is_public=is_public, db_name=db_name, now=now,
        )
        kg = await self.get_by_id(UUID(kg_id))
        return kg  # type: ignore[return-value]

    async def get_by_id(self, kg_id: UUID) -> KnowledgeGraph | None:
        result = await self.driver.execute_query(
            "MATCH (kg:KnowledgeGraph {id: $id}) RETURN kg",
            id=str(kg_id),
        )
        if not result.records:
            return None
        return self._to_model(result.records[0]["kg"])

    async def get_by_name(self, name: str, owner_id: str = "default") -> KnowledgeGraph | None:
        result = await self.driver.execute_query(
            "MATCH (kg:KnowledgeGraph {name: $name, owner_id: $owner_id}) RETURN kg",
            name=name, owner_id=owner_id,
        )
        if not result.records:
            return None
        return self._to_model(result.records[0]["kg"])

    async def list_all(
        self, owner_id: str | None = None, include_private: bool = False
    ) -> list[KnowledgeGraph]:
        if owner_id:
            query = """
                MATCH (kg:KnowledgeGraph)
                WHERE kg.owner_id = $owner_id OR (kg.is_public = true AND $include_private = false)
                RETURN kg ORDER BY kg.created_at DESC
            """
        else:
            query = """
                MATCH (kg:KnowledgeGraph)
                WHERE kg.is_public = true OR $include_private = true
                RETURN kg ORDER BY kg.created_at DESC
            """
        result = await self.driver.execute_query(
            query, owner_id=owner_id or "", include_private=include_private
        )
        return [self._to_model(r["kg"]) for r in result.records]

    async def update(
        self,
        kg_id: UUID,
        name: str | None = None,
        description: str | None = None,
        is_public: bool | None = None,
    ) -> KnowledgeGraph | None:
        sets = ["kg.updated_at = datetime()"]
        params: dict = {"id": str(kg_id)}
        if name is not None:
            sets.append("kg.name = $name")
            params["name"] = name
        if description is not None:
            sets.append("kg.description = $description")
            params["description"] = description
        if is_public is not None:
            sets.append("kg.is_public = $is_public")
            params["is_public"] = is_public

        await self.driver.execute_query(
            f"MATCH (kg:KnowledgeGraph {{id: $id}}) SET {', '.join(sets)}",
            **params,
        )
        return await self.get_by_id(kg_id)

    async def delete(self, kg_id: UUID) -> bool:
        """刪除 KG 節點及其所有 EFFECTIVE 邊（不刪除 Document 節點）。"""
        result = await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $id})
            OPTIONAL MATCH (kg)-[r]-()
            DELETE r, kg
            RETURN count(kg) AS deleted
            """,
            id=str(kg_id),
        )
        return result.records[0]["deleted"] > 0

    # ── 文件關聯 ──────────────────────────────────────────────────────────────

    async def add_document(self, kg_id: UUID, doc_id: UUID) -> None:
        await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $kg_id})
            MATCH (d:Document {id: $doc_id})
            MERGE (kg)-[:CONTAINS]->(d)
            SET kg.doc_count = kg.doc_count + 1, kg.updated_at = datetime()
            """,
            kg_id=str(kg_id), doc_id=str(doc_id),
        )

    async def remove_document(self, kg_id: UUID, doc_id: UUID) -> None:
        await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $kg_id})-[r:CONTAINS]->(d:Document {id: $doc_id})
            DELETE r
            SET kg.doc_count = CASE WHEN kg.doc_count > 0 THEN kg.doc_count - 1 ELSE 0 END,
                kg.updated_at = datetime()
            """,
            kg_id=str(kg_id), doc_id=str(doc_id),
        )

    async def get_documents(self, kg_id: UUID) -> list[dict]:
        result = await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $kg_id})-[:CONTAINS]->(d:Document)
            RETURN d.id AS id, d.title AS title, d.file_type AS file_type,
                   d.file_path AS file_path, d.created_at AS created_at
            ORDER BY d.created_at DESC
            """,
            kg_id=str(kg_id),
        )
        return [dict(r) for r in result.records]

    # ── 統計更新 ──────────────────────────────────────────────────────────────

    async def get_db_name(self, kg_id: UUID) -> str:
        """取得 KG 的專用資料庫名稱（空白表示使用主資料庫）。"""
        result = await self.driver.execute_query(
            "MATCH (kg:KnowledgeGraph {id: $id}) RETURN coalesce(kg.db_name, '') AS db_name",
            id=str(kg_id),
        )
        if not result.records:
            return ""
        return result.records[0]["db_name"] or ""

    async def refresh_counts(self, kg_id: UUID) -> None:
        """重新計算並寫入 doc_count / entity_count / relation_count。"""
        db_name = await self.get_db_name(kg_id)

        # doc_count 永遠在主資料庫
        doc_result = await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $id})-[:CONTAINS]->(d:Document)
            RETURN count(d) AS doc_c
            """,
            id=str(kg_id),
        )
        doc_c = doc_result.records[0]["doc_c"] if doc_result.records else 0

        _REL = ("IS_A|PART_OF|CONTAINS|INSTANCE_OF|CAUSES|PREVENTS|ENABLES|IMPROVES|INHIBITS|"
                "USES|REQUIRES|PRODUCES|IMPLEMENTS|REPLACES|EXTENDS|CONTRASTS|SIMILAR_TO|OUTPERFORMS|"
                "DEFINED_AS|HAS_PROPERTY|MEASURED_BY|APPLIES_TO|PRECEDES|FOLLOWS|CO_OCCURS|"
                "INPUTS|TRANSFORMS|CREATED_BY|SOLVES|RELATED_TO")

        # entity / relation count：per-KG db 或 main db
        if db_name:
            ent_result = await self.driver.execute_query(
                "MATCH (e:Entity) RETURN count(e) AS ent_c",
                database_=db_name,
            )
            rel_result = await self.driver.execute_query(
                f"MATCH ()-[r:{_REL}]->() RETURN count(r) AS rel_c",
                database_=db_name,
            )
        else:
            ent_result = await self.driver.execute_query(
                "MATCH (e:Entity {kg_id: $id}) RETURN count(e) AS ent_c",
                id=str(kg_id),
            )
            rel_result = await self.driver.execute_query(
                f"MATCH (s:Entity {{kg_id: $id}})-[r:{_REL}]->() RETURN count(r) AS rel_c",
                id=str(kg_id),
            )

        ent_c = ent_result.records[0]["ent_c"] if ent_result.records else 0
        rel_c = rel_result.records[0]["rel_c"] if rel_result.records else 0

        await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $id})
            SET kg.doc_count = $doc_c, kg.entity_count = $ent_c,
                kg.relation_count = $rel_c, kg.updated_at = datetime()
            """,
            id=str(kg_id), doc_c=doc_c, ent_c=ent_c, rel_c=rel_c,
        )

    # ── Detail（含 top concepts / entities）──────────────────────────────────

    async def get_detail(self, kg_id: UUID) -> KnowledgeGraphDetail | None:
        kg = await self.get_by_id(kg_id)
        if kg is None:
            return None

        concepts_result = await self.driver.execute_query(
            """
            MATCH (kg:KnowledgeGraph {id: $id})-[e:EFFECTIVE]->(c:ConceptNode)
            RETURN c.name AS name ORDER BY (e.interest_score + e.professional_score) DESC LIMIT 10
            """,
            id=str(kg_id),
        )
        top_concepts = [r["name"] for r in concepts_result.records]

        _REL = ("IS_A|PART_OF|CONTAINS|INSTANCE_OF|CAUSES|PREVENTS|ENABLES|IMPROVES|INHIBITS|"
                "USES|REQUIRES|PRODUCES|IMPLEMENTS|REPLACES|EXTENDS|CONTRASTS|SIMILAR_TO|OUTPERFORMS|"
                "DEFINED_AS|HAS_PROPERTY|MEASURED_BY|APPLIES_TO|PRECEDES|FOLLOWS|CO_OCCURS|"
                "INPUTS|TRANSFORMS|CREATED_BY|SOLVES|RELATED_TO")
        db_name = kg.db_name
        if db_name:
            entities_result = await self.driver.execute_query(
                f"""
                MATCH (e:Entity)<-[r:{_REL}]-()
                WITH e, count(r) AS rel_count
                RETURN e.name AS name ORDER BY rel_count DESC LIMIT 10
                """,
                database_=db_name,
            )
        else:
            entities_result = await self.driver.execute_query(
                f"""
                MATCH (e:Entity {{kg_id: $id}})<-[r:{_REL}]-()
                WITH e, count(r) AS rel_count
                RETURN e.name AS name ORDER BY rel_count DESC LIMIT 10
                """,
                id=str(kg_id),
            )
        top_entities = [r["name"] for r in entities_result.records]

        return KnowledgeGraphDetail(
            **kg.model_dump(),
            top_concepts=top_concepts,
            top_entities=top_entities,
        )

    # ── 內部工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_model(node) -> KnowledgeGraph:
        d = dict(node)
        for k in ("created_at", "updated_at"):
            if hasattr(d.get(k), "iso_format"):
                d[k] = d[k].iso_format()
        return KnowledgeGraph(
            id=UUID(d["id"]),
            name=d["name"],
            description=d.get("description", ""),
            folder_path=d.get("folder_path", ""),
            owner_id=d.get("owner_id", "default"),
            is_public=d.get("is_public", True),
            db_name=d.get("db_name", "") or "",
            doc_count=d.get("doc_count", 0),
            entity_count=d.get("entity_count", 0),
            relation_count=d.get("relation_count", 0),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
        )
