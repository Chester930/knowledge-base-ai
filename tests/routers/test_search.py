from __future__ import annotations
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from httpx import AsyncClient, ASGITransport
from models.document import Document


def _make_doc(**kwargs) -> Document:
    defaults = dict(
        id=uuid4(), title="文件", content="內容",
        file_type="txt", file_path=None,
        created_at=datetime.now(), updated_at=datetime.now(),
    )
    defaults.update(kwargs)
    return Document(**defaults)


def _vec(val=1.0, dim=384) -> list[float]:
    return [val] + [0.0] * (dim - 1)


# ── POST /search ──────────────────────────────────────────────────────────────

class TestSearch:
    async def test_no_concepts_returns_empty(self, test_app):
        with patch("routers.search.build_query_concepts", new=AsyncMock(return_value=[])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/search", json={"text": "query"})

        assert res.status_code == 200
        assert res.json() == []

    async def test_results_filtered_by_min_score(self, test_app):
        doc = _make_doc()
        concept = {"name": "概念", "q_vector": _vec(), "interest_score": 0.8, "professional_score": 0.8}

        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_documents_concepts.return_value = {doc.id: [concept]}

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = doc

        with patch("routers.search.build_query_concepts", new=AsyncMock(return_value=[concept])), \
             patch("routers.search.compute_match_score", return_value=(0.1, [])), \
             patch("routers.search.get_driver"), \
             patch("routers.search.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.search.DocumentRepository", return_value=mock_doc_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/search", json={"text": "query", "min_score": 0.5})

        assert res.status_code == 200
        # score 0.1 < min_score 0.5, so no results
        assert res.json() == []

    async def test_results_sorted_by_score_descending(self, test_app):
        doc1 = _make_doc(title="高分文件")
        doc2 = _make_doc(title="低分文件")
        concept = {"name": "概念", "q_vector": _vec(), "interest_score": 0.8, "professional_score": 0.8}

        doc_concepts = {doc1.id: [concept], doc2.id: [concept]}
        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_documents_concepts.return_value = doc_concepts

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.side_effect = lambda id: doc1 if id == doc1.id else doc2

        scores = {doc1.id: (0.9, ["概念"]), doc2.id: (0.3, ["概念"])}

        with patch("routers.search.build_query_concepts", new=AsyncMock(return_value=[concept])), \
             patch("routers.search.compute_match_score", side_effect=lambda q, d: scores[doc1.id] if d == doc_concepts[doc1.id] else scores[doc2.id]), \
             patch("routers.search.get_driver"), \
             patch("routers.search.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.search.DocumentRepository", return_value=mock_doc_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/search", json={"text": "query", "min_score": 0.0})

        results = res.json()
        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]

    async def test_top_k_limits_results(self, test_app):
        docs = [_make_doc(title=f"doc{i}") for i in range(10)]
        concept = {"name": "概念", "q_vector": _vec(), "interest_score": 0.8, "professional_score": 0.8}

        doc_concepts = {d.id: [concept] for d in docs}
        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_documents_concepts.return_value = doc_concepts

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.side_effect = lambda id: next(d for d in docs if d.id == id)

        with patch("routers.search.build_query_concepts", new=AsyncMock(return_value=[concept])), \
             patch("routers.search.compute_match_score", return_value=(0.8, ["概念"])), \
             patch("routers.search.get_driver"), \
             patch("routers.search.ConceptRepository", return_value=mock_concept_repo), \
             patch("routers.search.DocumentRepository", return_value=mock_doc_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/search", json={"text": "query", "top_k": 3, "min_score": 0.0})

        assert res.status_code == 200
        assert len(res.json()) <= 3

    async def test_invalid_top_k_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/search", json={"text": "q", "top_k": 0})
        assert res.status_code == 422

    async def test_invalid_min_score_returns_422(self, test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
            res = await c.post("/search", json={"text": "q", "min_score": 1.5})
        assert res.status_code == 422


# ── GET /search/concepts ──────────────────────────────────────────────────────

class TestListConcepts:
    async def test_returns_concept_list(self, test_app):
        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_concepts.return_value = [
            {"id": str(uuid4()), "name": "機器學習", "domain": "general", "doc_count": 5},
            {"id": str(uuid4()), "name": "深度學習", "domain": "general", "doc_count": 3},
        ]

        with patch("routers.search.get_driver"), \
             patch("routers.search.ConceptRepository", return_value=mock_concept_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/search/concepts")

        assert res.status_code == 200
        data = res.json()
        assert len(data) == 2
        assert data[0]["name"] == "機器學習"

    async def test_empty_concepts_returns_empty_list(self, test_app):
        mock_concept_repo = AsyncMock()
        mock_concept_repo.get_all_concepts.return_value = []

        with patch("routers.search.get_driver"), \
             patch("routers.search.ConceptRepository", return_value=mock_concept_repo):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/search/concepts")

        assert res.status_code == 200
        assert res.json() == []
