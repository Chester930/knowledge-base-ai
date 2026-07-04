from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from models.kb_skill import KBRegistry, KBSkill
from services.federation_service import (
    FederationCache,
    get_federation_cache,
    startup_prefetch,
    shutdown_cleanup,
)
import services.federation_service as federation_service


def _close_instead_of_schedule(coro):
    """取代 asyncio.create_task：只關閉背景協程、不真正排程執行，避免 'coroutine was never awaited' 警告。"""
    coro.close()
    return MagicMock()


def _make_skill(kb_id="kb-1", is_local=False, aura_uri="neo4j+s://x", read_token="tok"):
    return KBSkill(
        instance_id="remote",
        kb_id=kb_id,
        name="Remote KG",
        last_sync="2026-01-01T00:00:00+00:00",
        is_local=is_local,
        aura_uri=aura_uri,
        read_token=read_token,
    )


# ── get_remote_registry / _fetch ──────────────────────────────────────────────

class TestGetRemoteRegistry:
    async def test_returns_cached_registry_within_ttl(self):
        cache = FederationCache()
        cached = KBRegistry(updated_at="now")
        cache._github_registry = cached
        with patch("services.federation_service.time.time", return_value=cache._fetched_at + 10):
            result = await cache.get_remote_registry()
        assert result is cached

    async def test_no_url_configured_returns_empty_registry(self):
        cache = FederationCache()
        with patch("services.federation_service.settings") as mock_settings:
            mock_settings.github_registry_url = ""
            result = await cache.get_remote_registry()
        assert result.skills == []

    async def test_successful_fetch_populates_registry(self):
        cache = FederationCache()
        payload = {
            "version": "1.0",
            "updated_at": "now",
            "skills": [_make_skill().model_dump()],
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = payload

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__.return_value = mock_client

        with patch("services.federation_service.settings") as mock_settings, \
             patch("services.federation_service.httpx.AsyncClient", return_value=mock_client_cm):
            mock_settings.github_registry_url = "https://example.com/registry.json"
            result = await cache.get_remote_registry()

        assert len(result.skills) == 1
        assert result.skills[0].kb_id == "kb-1"

    async def test_fetch_failure_falls_back_to_empty_registry(self):
        cache = FederationCache()
        mock_client_cm = AsyncMock()
        mock_client_cm.__aenter__.side_effect = RuntimeError("network down")

        with patch("services.federation_service.settings") as mock_settings, \
             patch("services.federation_service.httpx.AsyncClient", return_value=mock_client_cm):
            mock_settings.github_registry_url = "https://example.com/registry.json"
            result = await cache.get_remote_registry()

        assert result.skills == []


# ── merged_registry ───────────────────────────────────────────────────────────

class TestMergedRegistry:
    async def test_local_skills_take_priority_over_remote_duplicates(self):
        cache = FederationCache()
        local_skill = _make_skill(kb_id="dup", is_local=True)
        remote_skill = _make_skill(kb_id="dup", is_local=False)
        other_remote = _make_skill(kb_id="other", is_local=False)

        local_registry = KBRegistry(updated_at="now", skills=[local_skill])
        remote_registry = KBRegistry(updated_at="now", skills=[remote_skill, other_remote])

        with patch("services.federation_service.load_registry", return_value=local_registry), \
             patch.object(cache, "get_remote_registry", new=AsyncMock(return_value=remote_registry)):
            merged = await cache.merged_registry()

        ids = [s.kb_id for s in merged.skills]
        assert ids.count("dup") == 1
        assert "other" in ids
        assert merged.skills[[s.kb_id for s in merged.skills].index("dup")].is_local is True


# ── get_shard_driver ───────────────────────────────────────────────────────────

class TestGetShardDriver:
    async def test_local_skill_returns_none(self):
        cache = FederationCache()
        skill = _make_skill(is_local=True)
        result = await cache.get_shard_driver(skill)
        assert result is None

    async def test_missing_aura_uri_returns_none(self):
        cache = FederationCache()
        skill = _make_skill(is_local=False, aura_uri="")
        result = await cache.get_shard_driver(skill)
        assert result is None

    async def test_circuit_breaker_skips_recent_failure(self):
        cache = FederationCache()
        skill = _make_skill(kb_id="flaky")
        cache._shard_status["flaky"] = "offline"
        with patch("services.federation_service.time.time", return_value=cache._offline_timestamps.get("flaky", 0.0) + 10):
            cache._offline_timestamps["flaky"] = 0.0
            result = await cache.get_shard_driver(skill)
        assert result is None

    async def test_successful_connection_cached_for_reuse(self):
        cache = FederationCache()
        skill = _make_skill(kb_id="good")
        mock_driver = AsyncMock()

        with patch("services.federation_service.AsyncGraphDatabase") as mock_graphdb:
            mock_graphdb.driver.return_value = mock_driver
            d1 = await cache.get_shard_driver(skill)
            d2 = await cache.get_shard_driver(skill)

        assert d1 is mock_driver
        assert d2 is mock_driver
        mock_graphdb.driver.assert_called_once()
        assert cache._shard_status["good"] == "online"

    async def test_connection_failure_marks_offline(self):
        cache = FederationCache()
        skill = _make_skill(kb_id="bad")

        with patch("services.federation_service.AsyncGraphDatabase") as mock_graphdb:
            mock_graphdb.driver.return_value.verify_connectivity = AsyncMock(
                side_effect=RuntimeError("連線失敗"))
            result = await cache.get_shard_driver(skill)

        assert result is None
        assert cache._shard_status["bad"] == "offline"


# ── mark_shard_offline ────────────────────────────────────────────────────────

class TestMarkShardOffline:
    def test_marks_status_and_removes_driver(self):
        cache = FederationCache()
        mock_driver = MagicMock()
        mock_driver.close = AsyncMock()
        cache._drivers["x"] = mock_driver

        with patch("services.federation_service.asyncio.create_task",
                   side_effect=_close_instead_of_schedule) as mock_create_task:
            cache.mark_shard_offline("x")

        assert cache._shard_status["x"] == "offline"
        assert "x" not in cache._drivers
        mock_create_task.assert_called_once()

    def test_noop_driver_close_when_not_connected(self):
        cache = FederationCache()
        cache.mark_shard_offline("never-connected")
        assert cache._shard_status["never-connected"] == "offline"


# ── status ────────────────────────────────────────────────────────────────────

class TestStatus:
    def test_reports_shard_summary(self):
        cache = FederationCache()
        cache._github_registry = KBRegistry(
            updated_at="now",
            skills=[_make_skill(kb_id="a"), _make_skill(kb_id="b")],
        )
        cache._shard_status["a"] = "online"

        with patch("services.federation_service.load_registry",
                   return_value=KBRegistry(updated_at="now", skills=[])), \
             patch("services.federation_service.settings") as mock_settings:
            mock_settings.github_registry_url = "https://x"
            result = cache.status()

        assert result["remote_skill_count"] == 2
        shard_a = next(s for s in result["shards"] if s["kb_id"] == "a")
        assert shard_a["status"] == "online"


# ── close ─────────────────────────────────────────────────────────────────────

class TestClose:
    async def test_closes_all_drivers_and_clears(self):
        cache = FederationCache()
        d1, d2 = AsyncMock(), AsyncMock()
        cache._drivers = {"a": d1, "b": d2}
        await cache.close()
        d1.close.assert_called_once()
        d2.close.assert_called_once()
        assert cache._drivers == {}

    async def test_close_swallows_individual_driver_errors(self):
        cache = FederationCache()
        d1 = AsyncMock()
        d1.close.side_effect = RuntimeError("boom")
        cache._drivers = {"a": d1}
        await cache.close()  # 不應拋出例外
        assert cache._drivers == {}


# ── singleton / lifespan hooks ────────────────────────────────────────────────

class TestSingleton:
    def test_get_federation_cache_returns_same_instance(self):
        federation_service._cache = None
        c1 = get_federation_cache()
        c2 = get_federation_cache()
        assert c1 is c2
        federation_service._cache = None


class TestLifespanHooks:
    async def test_startup_prefetch_schedules_background_task(self):
        federation_service._cache = None
        with patch("services.federation_service.asyncio.create_task",
                   side_effect=_close_instead_of_schedule) as mock_create_task:
            await startup_prefetch()
        mock_create_task.assert_called_once()
        federation_service._cache = None

    async def test_shutdown_cleanup_closes_cache_when_present(self):
        cache = FederationCache()
        cache.close = AsyncMock()
        federation_service._cache = cache
        await shutdown_cleanup()
        cache.close.assert_called_once()
        federation_service._cache = None

    async def test_shutdown_cleanup_noop_when_no_cache(self):
        federation_service._cache = None
        await shutdown_cleanup()  # 不應拋出例外
