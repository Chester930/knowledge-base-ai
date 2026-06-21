"""
Subscription 端點測試 — Phase 3d

GET  /world/subscriptions
POST /world/subscribe
DELETE /world/subscribe/{kb_id}
PATCH /world/subscribe/{kb_id}/pause
POST /world/sync-subscriptions
POST /world/sync-subscriptions/{kb_id}
"""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from httpx import AsyncClient, ASGITransport

from services.subscription_service import Subscription


def _sub(kb_id=None, kb_name="Test KG", status="active", last_sync_at="") -> Subscription:
    return Subscription(
        instance_id="chester",
        kb_id=kb_id or str(uuid4()),
        kb_name=kb_name,
        aura_uri="neo4j+s://test.aura.com",
        read_token="tok",
        status=status,
        last_sync_at=last_sync_at,
    )


def _mock_manager(subs=None):
    mgr = MagicMock()
    subs = subs or []
    mgr.list_all = AsyncMock(return_value=subs)
    mgr.get_by_kb_id = AsyncMock(side_effect=lambda kb_id: next((s for s in subs if s.kb_id == kb_id), None))
    mgr.add = AsyncMock()
    mgr.remove = AsyncMock(return_value=True)
    mgr.set_status = AsyncMock()
    mgr.set_last_sync = AsyncMock()
    return mgr


# ── GET /world/subscriptions ──────────────────────────────────────────────────

class TestListSubscriptions:
    async def test_returns_200_and_count(self, test_app):
        subs = [_sub(kb_name="KG1"), _sub(kb_name="KG2")]
        with patch("routers.subscription.get_subscription_manager", return_value=_mock_manager(subs)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/subscriptions")

        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 2
        assert len(data["subscriptions"]) == 2

    async def test_empty_subscriptions(self, test_app):
        with patch("routers.subscription.get_subscription_manager", return_value=_mock_manager([])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/subscriptions")

        assert res.status_code == 200
        assert res.json()["count"] == 0

    async def test_subscription_fields_present(self, test_app):
        subs = [_sub(kb_name="AI KG", status="active", last_sync_at="2026-06-21T10:00:00Z")]
        with patch("routers.subscription.get_subscription_manager", return_value=_mock_manager(subs)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.get("/world/subscriptions")

        sub_data = res.json()["subscriptions"][0]
        assert "kb_name" in sub_data
        assert "status" in sub_data
        assert "last_sync_at" in sub_data
        assert "instance_id" in sub_data


# ── POST /world/subscribe ─────────────────────────────────────────────────────

class TestSubscribe:
    def _body(self, **kwargs):
        base = {
            "instance_id": "chester",
            "kb_id": str(uuid4()),
            "kb_name": "Test KG",
            "aura_uri": "neo4j+s://xyz.databases.neo4j.io",
        }
        base.update(kwargs)
        return base

    async def test_subscribe_returns_201(self, test_app):
        mgr = _mock_manager()
        with patch("routers.subscription.get_subscription_manager", return_value=mgr):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/subscribe", json=self._body())

        assert res.status_code == 201
        data = res.json()
        assert "message" in data
        assert "kb_id" in data

    async def test_subscribe_missing_required_field_returns_422(self, test_app):
        mgr = _mock_manager()
        with patch("routers.subscription.get_subscription_manager", return_value=mgr):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/subscribe", json={"kb_id": str(uuid4())})

        assert res.status_code == 422

    async def test_duplicate_returns_409(self, test_app):
        mgr = _mock_manager()
        mgr.add = AsyncMock(side_effect=ValueError("已訂閱"))
        with patch("routers.subscription.get_subscription_manager", return_value=mgr):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/subscribe", json=self._body())

        assert res.status_code == 409

    async def test_manager_add_called_with_correct_data(self, test_app):
        mgr = _mock_manager()
        kb_id = str(uuid4())
        with patch("routers.subscription.get_subscription_manager", return_value=mgr):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                await c.post("/world/subscribe", json=self._body(kb_id=kb_id))

        mgr.add.assert_called_once()
        sub_arg = mgr.add.call_args[0][0]
        assert sub_arg.kb_id == kb_id
        assert sub_arg.instance_id == "chester"


# ── DELETE /world/subscribe/{kb_id} ──────────────────────────────────────────

class TestUnsubscribe:
    async def test_delete_existing_returns_200(self, test_app):
        kb_id = str(uuid4())
        mgr = _mock_manager()
        mgr.remove = AsyncMock(return_value=True)
        with patch("routers.subscription.get_subscription_manager", return_value=mgr):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.delete(f"/world/subscribe/{kb_id}")

        assert res.status_code == 200

    async def test_delete_nonexistent_returns_404(self, test_app):
        kb_id = str(uuid4())
        mgr = _mock_manager()
        mgr.remove = AsyncMock(return_value=False)
        with patch("routers.subscription.get_subscription_manager", return_value=mgr):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.delete(f"/world/subscribe/{kb_id}")

        assert res.status_code == 404


# ── PATCH /world/subscribe/{kb_id}/pause ─────────────────────────────────────

class TestPauseSubscription:
    async def test_pause_active_subscription(self, test_app):
        kb_id = str(uuid4())
        subs = [_sub(kb_id=kb_id, status="active")]
        mgr = _mock_manager(subs)
        with patch("routers.subscription.get_subscription_manager", return_value=mgr):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.patch(f"/world/subscribe/{kb_id}/pause", json={"paused": True})

        assert res.status_code == 200
        assert res.json()["status"] == "paused"
        mgr.set_status.assert_called_once_with(kb_id, "paused")

    async def test_resume_paused_subscription(self, test_app):
        kb_id = str(uuid4())
        subs = [_sub(kb_id=kb_id, status="paused")]
        mgr = _mock_manager(subs)
        with patch("routers.subscription.get_subscription_manager", return_value=mgr):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.patch(f"/world/subscribe/{kb_id}/pause", json={"paused": False})

        assert res.status_code == 200
        assert res.json()["status"] == "active"

    async def test_nonexistent_returns_404(self, test_app):
        kb_id = str(uuid4())
        mgr = _mock_manager([])
        with patch("routers.subscription.get_subscription_manager", return_value=mgr):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.patch(f"/world/subscribe/{kb_id}/pause", json={"paused": True})

        assert res.status_code == 404


# ── POST /world/sync-subscriptions ───────────────────────────────────────────

class TestSyncAllSubscriptions:
    async def test_returns_sync_summary(self, test_app):
        mock_results = [{"kb_id": str(uuid4()), "kb_name": "KG1", "merged": 50, "error": None}]
        with patch("routers.subscription.sync_all_subscriptions", new=AsyncMock(return_value=mock_results)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/sync-subscriptions")

        assert res.status_code == 200
        data = res.json()
        assert data["synced"] == 1
        assert data["total_merged"] == 50
        assert "errors" in data
        assert "results" in data

    async def test_empty_subscriptions_synced_zero(self, test_app):
        with patch("routers.subscription.sync_all_subscriptions", new=AsyncMock(return_value=[])):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post("/world/sync-subscriptions")

        assert res.status_code == 200
        assert res.json()["synced"] == 0


# ── POST /world/sync-subscriptions/{kb_id} ───────────────────────────────────

class TestSyncOneSubscription:
    async def test_sync_existing_subscription_success(self, test_app):
        kb_id = str(uuid4())
        subs = [_sub(kb_id=kb_id)]
        mgr = _mock_manager(subs)
        with patch("routers.subscription.get_subscription_manager", return_value=mgr), \
             patch("routers.subscription.sync_subscription",
                   new=AsyncMock(return_value={"merged": 100, "error": None})), \
             patch("services.subscription_service._now_iso", return_value="2026-06-21T12:00:00Z"):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(f"/world/sync-subscriptions/{kb_id}")

        assert res.status_code == 200
        data = res.json()
        assert data["merged"] == 100
        assert data["error"] is None

    async def test_sync_nonexistent_returns_404(self, test_app):
        kb_id = str(uuid4())
        mgr = _mock_manager([])
        with patch("routers.subscription.get_subscription_manager", return_value=mgr):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(f"/world/sync-subscriptions/{kb_id}")

        assert res.status_code == 404

    async def test_sync_error_updates_status(self, test_app):
        kb_id = str(uuid4())
        subs = [_sub(kb_id=kb_id)]
        mgr = _mock_manager(subs)
        with patch("routers.subscription.get_subscription_manager", return_value=mgr), \
             patch("routers.subscription.sync_subscription",
                   new=AsyncMock(return_value={"merged": 0, "error": "DB unreachable"})):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as c:
                res = await c.post(f"/world/sync-subscriptions/{kb_id}")

        assert res.status_code == 200
        assert res.json()["error"] == "DB unreachable"
        mgr.set_status.assert_called_once()
