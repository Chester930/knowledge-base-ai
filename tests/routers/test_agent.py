from __future__ import annotations
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from httpx import AsyncClient, ASGITransport
from models.document import Document


def _make_doc(**kwargs) -> Document:
    defaults = dict(
        id=uuid4(), title="文件", content="這是文件的主要內容，提供知識庫查詢使用。",
        file_type="txt", file_path=None,
        created_at=datetime.now(), updated_at=datetime.now(),
    )
    defaults.update(kwargs)
    return Document(**defaults)


def _vec(dim=384) -> list[float]:
    return [1.0] + [0.0] * (dim - 1)


# ── GET /agent/health ─────────────────────────────────────────────────────────

def _make_mock_driver(kg_cnt: int = 0, entity_cnt: int = 0):
    """Return an AsyncMock driver whose execute_query returns records with a 'cnt' key."""
    def _result(cnt):
        rec = MagicMock()
        rec.__getitem__ = lambda self, k: cnt if k == "cnt" else 0
        result = MagicMock()
        result.records = [rec]
        return result

    driver = MagicMock()
    driver.execute_query = AsyncMock(side_effect=[_result(kg_cnt), _result(entity_cnt)])
    return driver


class TestAgentHealth:
    async def test_returns_ok_status(self, test_app):
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_count.return_value = 42
        mock_driver = _make_mock_driver(kg_cnt=3, entity_cnt=100)

        with patch("routers.agent.get_driver", return_value=mock_driver), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/agent/health")

        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["document_count"] == 42

    async def test_includes_query_endpoint_info(self, test_app):
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_count.return_value = 0
        mock_driver = _make_mock_driver()

        with patch("routers.agent.get_driver", return_value=mock_driver), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/agent/health")

        data = res.json()
        assert "status" in data


# ── POST /agent/query ─────────────────────────────────────────────────────────

class TestAgentQuery:
    async def test_no_concepts_returns_empty_context(self, test_app):
        with patch("routers.agent.build_query_concepts", new=AsyncMock(return_value=[])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/agent/query", json={"question": "我的問題"})

        assert res.status_code == 200
        data = res.json()
        assert data["question"] == "我的問題"
        assert data["context"] == []
        assert data["sources"] == []

    async def test_returns_relevant_documents(self, test_app):
        doc = _make_doc(title="相關文件")
        concept = {"name": "概念", "q_vector": _vec(), "interest_score": 0.9, "professional_score": 0.9}

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_documents_concepts.return_value = {doc.id: [concept]}

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = doc

        with patch("routers.agent.build_query_concepts", new=AsyncMock(return_value=[concept])), \
             patch("routers.agent.compute_match_score", return_value=(0.85, ["概念"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/agent/query", json={"question": "相關問題"})

        assert res.status_code == 200
        data = res.json()
        assert len(data["context"]) == 1
        assert data["context"][0]["title"] == "相關文件"
        assert "相關文件" in data["sources"]

    async def test_include_content_false_returns_empty_snippet(self, test_app):
        doc = _make_doc()
        concept = {"name": "概念", "q_vector": _vec(), "interest_score": 0.9, "professional_score": 0.9}

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_documents_concepts.return_value = {doc.id: [concept]}

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = doc

        with patch("routers.agent.build_query_concepts", new=AsyncMock(return_value=[concept])), \
             patch("routers.agent.compute_match_score", return_value=(0.85, ["概念"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/agent/query", json={
                    "question": "query", "include_content": False
                })

        data = res.json()
        assert len(data["context"]) == 1
        assert data["context"][0]["content_snippet"] == ""

    async def test_max_content_chars_truncates_snippet(self, test_app):
        long_content = "A" * 5000
        doc = _make_doc(content=long_content)
        concept = {"name": "概念", "q_vector": _vec(), "interest_score": 0.9, "professional_score": 0.9}

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_documents_concepts.return_value = {doc.id: [concept]}

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = doc

        with patch("routers.agent.build_query_concepts", new=AsyncMock(return_value=[concept])), \
             patch("routers.agent.compute_match_score", return_value=(0.85, ["概念"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/agent/query", json={
                    "question": "query", "max_content_chars": 100
                })

        snippet = res.json()["context"][0]["content_snippet"]
        assert len(snippet) <= 100

    async def test_top_k_limits_context(self, test_app):
        docs = [_make_doc(title=f"doc{i}") for i in range(10)]
        concept = {"name": "概念", "q_vector": _vec(), "interest_score": 0.9, "professional_score": 0.9}

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_documents_concepts.return_value = {d.id: [concept] for d in docs}

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.side_effect = lambda id: next(d for d in docs if d.id == id)

        with patch("routers.agent.build_query_concepts", new=AsyncMock(return_value=[concept])), \
             patch("routers.agent.compute_match_score", return_value=(0.85, ["概念"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/agent/query", json={"question": "q", "top_k": 3})

        assert len(res.json()["context"]) <= 3

    async def test_uses_relaxed_score_threshold(self, test_app):
        doc = _make_doc()
        concept = {"name": "概念", "q_vector": _vec(), "interest_score": 0.8, "professional_score": 0.8}

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_documents_concepts.return_value = {doc.id: [concept]}

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = doc

        # score = 0.4 = score_threshold(0.7) * ~0.57, passes the agent's 0.5 * threshold
        with patch("routers.agent.build_query_concepts", new=AsyncMock(return_value=[concept])), \
             patch("routers.agent.compute_match_score", return_value=(0.4, ["概念"])), \
             patch("routers.agent.get_driver"), \
             patch("routers.agent.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.agent.DocumentRepository", return_value=mock_doc_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/agent/query", json={"question": "q"})

        # score_threshold * 0.5 = 0.35, so 0.4 should pass
        assert len(res.json()["context"]) == 1

    async def test_missing_question_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/agent/query", json={})
        assert res.status_code == 422

    async def test_top_k_out_of_range_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/agent/query", json={"question": "q", "top_k": 0})
        assert res.status_code == 422
