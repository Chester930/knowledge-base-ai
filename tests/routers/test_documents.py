from __future__ import annotations
import io
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from httpx import AsyncClient, ASGITransport
from models.document import Document


def _make_doc(**kwargs) -> Document:
    defaults = dict(
        id=uuid4(), title="測試文件", content="測試內容",
        file_type="manual", file_path=None,
        created_at=datetime.now(), updated_at=datetime.now(),
    )
    defaults.update(kwargs)
    return Document(**defaults)


# ── GET /documents ────────────────────────────────────────────────────────────

class TestListDocuments:
    async def test_returns_list(self, test_app):
        docs = [_make_doc(title=f"doc{i}") for i in range(3)]
        mock_repo = AsyncMock()
        mock_repo.list_all.return_value = docs

        with patch("routers.documents.get_driver"), \
             patch("routers.documents.DocumentRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/documents")

        assert res.status_code == 200
        assert len(res.json()) == 3

    async def test_passes_limit_and_offset(self, test_app):
        mock_repo = AsyncMock()
        mock_repo.list_all.return_value = []

        with patch("routers.documents.get_driver"), \
             patch("routers.documents.DocumentRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.get("/documents?limit=5&offset=10")

        mock_repo.list_all.assert_called_once_with(5, 10)

    async def test_empty_db_returns_empty_list(self, test_app):
        mock_repo = AsyncMock()
        mock_repo.list_all.return_value = []

        with patch("routers.documents.get_driver"), \
             patch("routers.documents.DocumentRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/documents")

        assert res.status_code == 200
        assert res.json() == []


# ── GET /documents/{doc_id} ──────────────────────────────────────────────────

class TestGetDocument:
    async def test_found(self, test_app):
        doc = _make_doc()
        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = doc

        with patch("routers.documents.get_driver"), \
             patch("routers.documents.DocumentRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/documents/{doc.id}")

        assert res.status_code == 200
        assert res.json()["id"] == str(doc.id)
        assert res.json()["title"] == doc.title

    async def test_not_found_returns_404(self, test_app):
        mock_repo = AsyncMock()
        mock_repo.get_by_id.return_value = None

        with patch("routers.documents.get_driver"), \
             patch("routers.documents.DocumentRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/documents/{uuid4()}")

        assert res.status_code == 404

    async def test_invalid_uuid_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.get("/documents/not-a-uuid")

        assert res.status_code == 422


# ── POST /documents ──────────────────────────────────────────────────────────

class TestCreateDocument:
    async def test_creates_document_and_returns_201(self, test_app):
        doc = _make_doc(title="新文件", content="新內容")
        mock_repo = AsyncMock()
        mock_repo.create.return_value = doc

        with patch("routers.documents.get_driver"), \
             patch("routers.documents.DocumentRepository", return_value=mock_repo), \
             patch("routers.documents.extract_and_init_document_concepts", new=AsyncMock()):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/documents", json={
                    "title": "新文件", "content": "新內容", "file_type": "manual"
                })

        assert res.status_code == 201
        assert res.json()["title"] == "新文件"

    async def test_missing_title_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/documents", json={"content": "content only"})

        assert res.status_code == 422

    async def test_missing_content_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/documents", json={"title": "title only"})

        assert res.status_code == 422


# ── DELETE /documents/{doc_id} ───────────────────────────────────────────────

class TestDeleteDocument:
    async def test_delete_existing_returns_204(self, test_app):
        mock_repo = AsyncMock()
        mock_repo.delete.return_value = True

        with patch("routers.documents.get_driver"), \
             patch("routers.documents.DocumentRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.delete(f"/documents/{uuid4()}")

        assert res.status_code == 204

    async def test_delete_nonexistent_returns_404(self, test_app):
        mock_repo = AsyncMock()
        mock_repo.delete.return_value = False

        with patch("routers.documents.get_driver"), \
             patch("routers.documents.DocumentRepository", return_value=mock_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.delete(f"/documents/{uuid4()}")

        assert res.status_code == 404


# ── GET /documents/{doc_id}/concepts ────────────────────────────────────────

class TestGetDocumentConcepts:
    async def test_returns_concept_list(self, test_app):
        doc_id = uuid4()
        concepts_data = [
            {
                "id": str(uuid4()),
                "name": "機器學習",
                "domain": "general",
                "q_vector": [0.1] * 384,
                "interest_score": 0.8,
                "professional_score": 0.7,
            }
        ]
        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_document_concepts.return_value = concepts_data

        with patch("routers.documents.get_driver"), \
             patch("routers.documents.ConceptRepository", return_value=mock_concept_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/documents/{doc_id}/concepts")

        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["name"] == "機器學習"

    async def test_empty_concepts_returns_empty_list(self, test_app):
        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_document_concepts.return_value = []

        with patch("routers.documents.get_driver"), \
             patch("routers.documents.ConceptRepository", return_value=mock_concept_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/documents/{uuid4()}/concepts")

        assert res.status_code == 200
        assert res.json() == []


# ── POST /documents/upload ───────────────────────────────────────────────────

class TestUploadDocument:
    async def test_unsupported_format_returns_400(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post(
                "/documents/upload",
                files={"file": ("test.exe", b"binary content", "application/octet-stream")},
            )

        assert res.status_code == 400

    async def test_supported_txt_upload_returns_201(self, test_app):
        doc = _make_doc(title="uploaded", file_type="txt")

        with patch("routers.documents.ingest_file", new=AsyncMock(return_value=doc)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(
                    "/documents/upload",
                    files={"file": ("note.txt", b"\xe9\x80\x99\xe6\x98\xaf\xe5\x85\xa7\xe5\xae\xb9", "text/plain")},
                )

        assert res.status_code == 201
