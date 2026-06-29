from __future__ import annotations
import logging
from uuid import UUID, uuid4
from neo4j import AsyncDriver
from models.document import Document

logger = logging.getLogger(__name__)


class DocumentRepository:
    def __init__(self, driver: AsyncDriver):
        self.driver = driver

    async def create(
        self, title: str, content: str, file_path: str | None, file_type: str
    ) -> Document:
        doc_id = uuid4()
        # MERGE on file_path to prevent duplicates on repeated ingestion runs.
        # ON CREATE: assign a fresh UUID and timestamps.
        # ON MATCH:  update content/title only, preserve the original id and created_at.
        result = await self.driver.execute_query(
            """
            MERGE (d:Document {file_path: $file_path})
            ON CREATE SET
                d.id         = $id,
                d.title      = $title,
                d.content    = $content,
                d.file_type  = $file_type,
                d.created_at = datetime(),
                d.updated_at = datetime()
            ON MATCH SET
                d.title      = $title,
                d.content    = $content,
                d.file_type  = $file_type,
                d.updated_at = datetime()
            RETURN d
            """,
            id=str(doc_id), title=title, content=content,
            file_path=file_path, file_type=file_type,
        )
        return self._to_model(result.records[0]["d"])

    async def get_by_id(self, doc_id: UUID) -> Document | None:
        result = await self.driver.execute_query(
            "MATCH (d:Document {id: $id}) RETURN d",
            id=str(doc_id),
        )
        if not result.records:
            return None
        return self._to_model(result.records[0]["d"])

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Document]:
        result = await self.driver.execute_query(
            "MATCH (d:Document) RETURN d ORDER BY d.created_at DESC SKIP $offset LIMIT $limit",
            offset=offset, limit=limit,
        )
        return [self._to_model(r["d"]) for r in result.records]

    async def delete(self, doc_id: UUID) -> bool:
        result = await self.driver.execute_query(
            "MATCH (d:Document {id: $id}) DETACH DELETE d RETURN count(d) AS cnt",
            id=str(doc_id),
        )
        return result.records[0]["cnt"] > 0

    async def search_by_title(self, query: str) -> list[Document]:
        result = await self.driver.execute_query(
            "MATCH (d:Document) WHERE toLower(d.title) CONTAINS toLower($q) RETURN d LIMIT 20",
            q=query,
        )
        return [self._to_model(r["d"]) for r in result.records]

    async def get_count(self) -> int:
        result = await self.driver.execute_query(
            "MATCH (d:Document) RETURN count(d) AS cnt"
        )
        return result.records[0]["cnt"]

    async def get_orphan_documents(self, preview_chars: int = 300) -> list[dict]:
        """回傳未被任何 KG CONTAINS 的孤立文件（id, title, preview）。"""
        result = await self.driver.execute_query(
            """
            MATCH (d:Document)
            WHERE NOT ()-[:CONTAINS]->(d)
            RETURN d.id AS id, d.title AS title, d.content AS content
            ORDER BY d.created_at
            """
        )
        docs = []
        for r in result.records:
            preview = (r["content"] or "")[:preview_chars].replace("\n", " ")
            docs.append({"id": r["id"], "title": r["title"], "preview": preview})
        return docs

    def _to_model(self, node) -> Document:
        return Document(
            id=UUID(node["id"]),
            title=node["title"],
            content=node["content"],
            file_path=node.get("file_path"),
            file_type=node["file_type"],
            created_at=node["created_at"].to_native(),
            updated_at=node["updated_at"].to_native(),
        )
