"""
KG Version Control 端點測試 — Phase 3c

GET /kg/{id}/changelog
GET /kg/{id}/diff?since=ISO8601
GET /kg/{id}/snapshot?at=ISO8601
"""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from httpx import AsyncClient, ASGITransport


def _make_kg(kg_id: str | None = None, db_name: str | None = None):
    kg = MagicMock()
    kg.id = uuid4() if kg_id is None else type("UUID", (), {"__str__": lambda s: kg_id})()
    kg.name = "測試 KG"
    kg.db_name = db_name
    return kg


def _make_driver(records=None):
    result = MagicMock()
    result.records = records or []
    driver = MagicMock()
    driver.execute_query = AsyncMock(return_value=result)
    return driver


def _fact_record(subject="深度學習", rel_type="USES", verb="使用", object_="GPU",
                 created_at="2026-06-21T10:00:00", updated_at=None):
    rec = MagicMock()
    rec.__getitem__ = lambda self, k: {
        "subject": subject, "subject_type": "算法", "rel_type": rel_type,
        "verb": verb, "object": object_, "object_type": "工具",
        "confidence": 1, "source_doc_id": "doc-001",
        "created_at": created_at, "updated_at": updated_at,
    }.get(k)
    rec.get = lambda k, default=None: {
        "subject": subject, "subject_type": "算法", "rel_type": rel_type,
        "verb": verb, "object": object_, "object_type": "工具",
        "confidence": 1, "source_doc_id": "doc-001",
        "created_at": created_at, "updated_at": updated_at,
    }.get(k, default)
    return rec


# ── GET /kg/{id}/changelog ────────────────────────────────────────────────────

class TestKgChangelog:
    async def test_returns_200_with_facts(self, test_app):
        kg_id = str(uuid4())
        kg = _make_kg(kg_id)
        driver = _make_driver(records=[_fact_record()])

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=kg)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/changelog")

        assert res.status_code == 200
        data = res.json()
        assert "facts" in data
        assert "kg_id" in data
        assert "kg_name" in data

    async def test_change_type_created_when_no_updated_at(self, test_app):
        kg_id = str(uuid4())
        kg = _make_kg(kg_id)
        driver = _make_driver(records=[_fact_record(updated_at=None)])

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=kg)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/changelog")

        facts = res.json()["facts"]
        if facts:
            assert facts[0]["change_type"] == "created"

    async def test_change_type_updated_when_has_updated_at(self, test_app):
        kg_id = str(uuid4())
        kg = _make_kg(kg_id)
        driver = _make_driver(records=[_fact_record(updated_at="2026-06-21T12:00:00")])

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=kg)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/changelog")

        facts = res.json()["facts"]
        if facts:
            assert facts[0]["change_type"] == "updated"

    async def test_kg_not_found_returns_404(self, test_app):
        kg_id = str(uuid4())
        driver = _make_driver()

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=None)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/changelog")

        assert res.status_code == 404

    async def test_invalid_uuid_returns_422(self, test_app):
        driver = _make_driver()
        with patch("routers.versioning.get_driver", return_value=driver):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/kg/not-a-uuid/changelog")
        assert res.status_code in (404, 422)

    async def test_default_limit_is_50(self, test_app):
        kg_id = str(uuid4())
        kg = _make_kg(kg_id)
        driver = _make_driver()

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=kg)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/changelog")

        assert res.status_code == 200
        data = res.json()
        assert data["limit"] == 50

    async def test_offset_param_passed_through(self, test_app):
        kg_id = str(uuid4())
        kg = _make_kg(kg_id)
        driver = _make_driver()

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=kg)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/changelog?offset=10")

        assert res.status_code == 200
        assert res.json()["offset"] == 10


# ── GET /kg/{id}/diff ─────────────────────────────────────────────────────────

class TestKgDiff:
    async def test_returns_200_with_required_fields(self, test_app):
        kg_id = str(uuid4())
        kg = _make_kg(kg_id)
        driver = _make_driver(records=[_fact_record(), _fact_record(updated_at="2026-06-21T12:00:00")])

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=kg)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/diff?since=2026-06-20T00:00:00")

        assert res.status_code == 200
        data = res.json()
        assert "facts" in data
        assert "since" in data
        assert "fact_count" in data
        assert "created_count" in data
        assert "updated_count" in data

    async def test_counts_match_change_types(self, test_app):
        kg_id = str(uuid4())
        kg = _make_kg(kg_id)
        # 1 created + 1 updated
        driver = _make_driver(records=[
            _fact_record(updated_at=None),
            _fact_record(subject="Transformer", updated_at="2026-06-21T12:00:00"),
        ])

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=kg)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/diff?since=2026-06-20T00:00:00")

        data = res.json()
        assert data["created_count"] + data["updated_count"] == data["fact_count"]

    async def test_since_is_required(self, test_app):
        kg_id = str(uuid4())
        driver = _make_driver()

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=_make_kg(kg_id))
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/diff")

        assert res.status_code == 422

    async def test_kg_not_found_returns_404(self, test_app):
        kg_id = str(uuid4())
        driver = _make_driver()

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=None)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/diff?since=2026-06-20T00:00:00")

        assert res.status_code == 404

    async def test_malformed_since_returns_422_not_500(self, test_app):
        """格式錯誤的 since 應回 422，而非讓 Neo4j datetime() 拋未捕捉例外變成 500。"""
        kg_id = str(uuid4())
        driver = _make_driver()

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=_make_kg(kg_id))
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/diff?since=not-a-real-date")

        assert res.status_code == 422
        # 驗證發生在查詢 KG 之前，Neo4j 完全不應被觸碰
        driver.execute_query.assert_not_called()


# ── GET /kg/{id}/snapshot ─────────────────────────────────────────────────────

class TestKgSnapshot:
    async def test_returns_200_with_required_fields(self, test_app):
        kg_id = str(uuid4())
        kg = _make_kg(kg_id)
        driver = _make_driver(records=[_fact_record()])

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=kg)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/snapshot?at=2026-06-21T23:59:59")

        assert res.status_code == 200
        data = res.json()
        assert "facts" in data
        assert "snapshot_at" in data
        assert "fact_count" in data

    async def test_at_is_required(self, test_app):
        kg_id = str(uuid4())
        driver = _make_driver()

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=_make_kg(kg_id))
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/snapshot")

        assert res.status_code == 422

    async def test_kg_not_found_returns_404(self, test_app):
        kg_id = str(uuid4())
        driver = _make_driver()

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=None)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/snapshot?at=2026-06-21T23:59:59")

        assert res.status_code == 404

    async def test_empty_snapshot_returns_zero_count(self, test_app):
        kg_id = str(uuid4())
        kg = _make_kg(kg_id)
        driver = _make_driver(records=[])

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=kg)
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/snapshot?at=2020-01-01T00:00:00")

        data = res.json()
        assert data["fact_count"] == 0
        assert data["facts"] == []

    async def test_malformed_at_returns_422_not_500(self, test_app):
        """格式錯誤的 at 應回 422，而非讓 Neo4j datetime() 拋未捕捉例外變成 500。"""
        kg_id = str(uuid4())
        driver = _make_driver()

        with patch("routers.versioning.get_driver", return_value=driver), \
             patch("routers.versioning.KnowledgeGraphRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=_make_kg(kg_id))
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get(f"/kg/{kg_id}/snapshot?at=yesterday")

        assert res.status_code == 422
        driver.execute_query.assert_not_called()
