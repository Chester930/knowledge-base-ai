"""
Subscription Service 測試 — Phase 3d

- SubscriptionManager: add/remove/list/set_status/set_last_sync
- sync_all_subscriptions: 逐筆同步，超時處理
- Subscription dataclass 預設值
"""
from __future__ import annotations
import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from uuid import uuid4

from services.subscription_service import (
    Subscription,
    SubscriptionManager,
    sync_all_subscriptions,
    sync_subscription,
    _now_iso,
)


# ── Subscription dataclass ────────────────────────────────────────────────────

class TestSubscriptionDataclass:
    def test_default_status_is_active(self):
        s = Subscription(instance_id="a", kb_id="b", kb_name="KG", aura_uri="neo4j+s://x")
        assert s.status == "active"

    def test_default_interval_is_6(self):
        s = Subscription(instance_id="a", kb_id="b", kb_name="KG", aura_uri="neo4j+s://x")
        assert s.sync_interval_hours == 6

    def test_default_error_msg_empty(self):
        s = Subscription(instance_id="a", kb_id="b", kb_name="KG", aura_uri="neo4j+s://x")
        assert s.error_msg == ""

    def test_last_sync_at_default_empty(self):
        s = Subscription(instance_id="a", kb_id="b", kb_name="KG", aura_uri="neo4j+s://x")
        assert s.last_sync_at == ""


# ── SubscriptionManager ───────────────────────────────────────────────────────

def _fresh_manager() -> SubscriptionManager:
    mgr = SubscriptionManager.__new__(SubscriptionManager)
    mgr._lock = asyncio.Lock()
    mgr._subs = []
    mgr._loaded = True
    return mgr


def _make_sub(kb_id=None, status="active") -> Subscription:
    return Subscription(
        instance_id="chester",
        kb_id=kb_id or str(uuid4()),
        kb_name="Test KG",
        aura_uri="neo4j+s://test.io",
        status=status,
    )


class TestSubscriptionManagerAdd:
    async def test_add_new_subscription(self):
        mgr = _fresh_manager()
        sub = _make_sub()
        with patch.object(mgr, "_save", new=AsyncMock()):
            await mgr.add(sub)
        assert len(mgr._subs) == 1
        assert mgr._subs[0].kb_id == sub.kb_id

    async def test_add_duplicate_raises_value_error(self):
        mgr = _fresh_manager()
        sub = _make_sub()
        mgr._subs = [sub]
        with patch.object(mgr, "_save", new=AsyncMock()):
            with pytest.raises(ValueError, match="已訂閱"):
                await mgr.add(sub)

    async def test_add_saves_to_disk(self):
        mgr = _fresh_manager()
        sub = _make_sub()
        save_mock = AsyncMock()
        with patch.object(mgr, "_save", new=save_mock):
            await mgr.add(sub)
        save_mock.assert_called_once()


class TestSubscriptionManagerRemove:
    async def test_remove_existing_returns_true(self):
        mgr = _fresh_manager()
        sub = _make_sub()
        mgr._subs = [sub]
        with patch.object(mgr, "_save", new=AsyncMock()):
            result = await mgr.remove(sub.kb_id)
        assert result is True
        assert len(mgr._subs) == 0

    async def test_remove_nonexistent_returns_false(self):
        mgr = _fresh_manager()
        with patch.object(mgr, "_save", new=AsyncMock()):
            result = await mgr.remove("nonexistent-id")
        assert result is False


class TestSubscriptionManagerList:
    async def test_list_all_returns_copy(self):
        mgr = _fresh_manager()
        mgr._subs = [_make_sub(), _make_sub()]
        result = await mgr.list_all()
        assert len(result) == 2

    async def test_get_by_kb_id_found(self):
        mgr = _fresh_manager()
        sub = _make_sub()
        mgr._subs = [sub]
        found = await mgr.get_by_kb_id(sub.kb_id)
        assert found is not None
        assert found.kb_id == sub.kb_id

    async def test_get_by_kb_id_not_found(self):
        mgr = _fresh_manager()
        mgr._subs = []
        found = await mgr.get_by_kb_id("no-such-id")
        assert found is None


class TestSubscriptionManagerSetStatus:
    async def test_set_status_updates_subscription(self):
        mgr = _fresh_manager()
        sub = _make_sub(status="active")
        mgr._subs = [sub]
        with patch.object(mgr, "_save", new=AsyncMock()):
            await mgr.set_status(sub.kb_id, "error", "connection failed")
        assert mgr._subs[0].status == "error"
        assert mgr._subs[0].error_msg == "connection failed"

    async def test_set_last_sync_updates_fields(self):
        mgr = _fresh_manager()
        sub = _make_sub()
        mgr._subs = [sub]
        with patch.object(mgr, "_save", new=AsyncMock()):
            await mgr.set_last_sync(sub.kb_id, "2026-06-21T12:00:00Z")
        assert mgr._subs[0].last_sync_at == "2026-06-21T12:00:00Z"
        assert mgr._subs[0].status == "active"
        assert mgr._subs[0].error_msg == ""


# ── Persistence (_load / _save) ───────────────────────────────────────────────

class TestSubscriptionManagerPersistence:
    async def test_load_from_json(self):
        mgr = SubscriptionManager.__new__(SubscriptionManager)
        mgr._lock = asyncio.Lock()
        mgr._subs = []
        mgr._loaded = False

        file_data = json.dumps({
            "subscriptions": [
                {"instance_id": "a", "kb_id": "b", "kb_name": "KG",
                 "aura_uri": "neo4j+s://x", "read_token": "", "last_sync_at": "",
                 "sync_interval_hours": 6, "status": "active", "error_msg": "", "local_kg_id": ""}
            ]
        })
        with patch("services.subscription_service._SUBSCRIPTIONS_FILE") as mock_path:
            mock_path.exists.return_value = True
            mock_path.read_text.return_value = file_data
            await mgr._load()

        assert len(mgr._subs) == 1
        assert mgr._subs[0].kb_id == "b"

    async def test_load_missing_file_gives_empty(self):
        mgr = _fresh_manager()
        mgr._loaded = False
        with patch("services.subscription_service._SUBSCRIPTIONS_FILE") as mock_path:
            mock_path.exists.return_value = False
            await mgr._load()
        assert mgr._subs == []

    async def test_save_writes_json(self, tmp_path):
        mgr = _fresh_manager()
        mgr._subs = [_make_sub()]
        target = tmp_path / "subscriptions.json"
        with patch("services.subscription_service._SUBSCRIPTIONS_FILE", target):
            with patch("services.subscription_service._now_iso", return_value="2026-06-21T00:00:00Z"):
                await mgr._save()

        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["updated_at"] == "2026-06-21T00:00:00Z"
        assert len(data["subscriptions"]) == 1

    async def test_save_leaves_no_leftover_tmp_files(self, tmp_path):
        mgr = _fresh_manager()
        mgr._subs = [_make_sub()]
        target = tmp_path / "subscriptions.json"
        with patch("services.subscription_service._SUBSCRIPTIONS_FILE", target):
            await mgr._save()

        siblings = list(tmp_path.iterdir())
        assert siblings == [target], f"應只留下 subscriptions.json，實際：{siblings}"

    async def test_save_is_atomic_original_untouched_on_failure(self, tmp_path):
        mgr = _fresh_manager()
        mgr._subs = [_make_sub()]
        target = tmp_path / "subscriptions.json"
        with patch("services.subscription_service._SUBSCRIPTIONS_FILE", target):
            await mgr._save()
        original_content = target.read_text(encoding="utf-8")

        mgr._subs = [_make_sub(kb_id="new-kb-id")]
        with patch("services.subscription_service._SUBSCRIPTIONS_FILE", target), \
             patch("os.replace", side_effect=OSError("模擬寫入中斷")):
            with pytest.raises(OSError):
                await mgr._save()

        assert target.read_text(encoding="utf-8") == original_content
        assert list(tmp_path.iterdir()) == [target]


# ── sync_all_subscriptions ────────────────────────────────────────────────────

class TestSyncAllSubscriptions:
    async def test_empty_active_subs_returns_empty_list(self):
        mgr = _fresh_manager()
        with patch("services.subscription_service.SubscriptionManager.get", return_value=mgr):
            result = await sync_all_subscriptions()
        assert result == []

    async def test_paused_subs_are_skipped(self):
        mgr = _fresh_manager()
        mgr._subs = [_make_sub(status="paused")]
        with patch("services.subscription_service.SubscriptionManager.get", return_value=mgr):
            result = await sync_all_subscriptions()
        assert result == []

    async def test_active_sub_is_synced(self):
        mgr = _fresh_manager()
        sub = _make_sub(status="active")
        mgr._subs = [sub]
        with patch("services.subscription_service.SubscriptionManager.get", return_value=mgr), \
             patch("services.subscription_service.sync_subscription",
                   new=AsyncMock(return_value={"merged": 10, "error": None})), \
             patch.object(mgr, "_save", new=AsyncMock()):
            result = await sync_all_subscriptions()

        assert len(result) == 1
        assert result[0]["merged"] == 10
        assert result[0]["error"] is None

    async def test_timeout_marks_error(self):
        mgr = _fresh_manager()
        sub = _make_sub(status="active")
        mgr._subs = [sub]

        async def _timeout_wait_for(coro, timeout=None):
            coro.close()  # 關閉 coroutine 避免 never-awaited warning
            raise asyncio.TimeoutError()

        with patch("services.subscription_service.SubscriptionManager.get", return_value=mgr), \
             patch("services.subscription_service.sync_subscription", new=AsyncMock()), \
             patch.object(mgr, "set_status", new=AsyncMock()), \
             patch("services.subscription_service.asyncio.wait_for", new=_timeout_wait_for):
            result = await sync_all_subscriptions()

        assert len(result) == 1
        assert result[0]["error"] == "timeout"

    async def test_exception_marks_error(self):
        mgr = _fresh_manager()
        sub = _make_sub(status="active")
        mgr._subs = [sub]
        with patch("services.subscription_service.SubscriptionManager.get", return_value=mgr), \
             patch("services.subscription_service.sync_subscription",
                   new=AsyncMock(side_effect=RuntimeError("boom"))), \
             patch.object(mgr, "set_status", new=AsyncMock()):
            result = await sync_all_subscriptions()

        assert len(result) == 1
        assert "boom" in result[0]["error"]


# ── _now_iso ──────────────────────────────────────────────────────────────────

class TestNowIso:
    def test_returns_iso_string(self):
        ts = _now_iso()
        assert "T" in ts
        assert ts.endswith("+00:00") or ts.endswith("Z") or "UTC" in ts or "+" in ts
