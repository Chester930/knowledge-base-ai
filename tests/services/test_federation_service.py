from __future__ import annotations
import asyncio
import time

import pytest

from models.kb_skill import KBRegistry
from services.federation_service import FederationCache


@pytest.fixture
def cache():
    return FederationCache()


async def test_cold_start_fetches_synchronously(cache, monkeypatch):
    """尚無任何快取時，get_remote_registry() 必須同步等待第一次下載完成。"""
    calls = []

    async def fake_fetch(self):
        calls.append(1)
        self._github_registry = KBRegistry(updated_at="x")
        self._fetched_at = time.time()

    monkeypatch.setattr(FederationCache, "_fetch", fake_fetch)

    result = await cache.get_remote_registry()
    assert len(calls) == 1
    assert result.updated_at == "x"


async def test_warm_cache_returns_immediately_without_fetch(cache, monkeypatch):
    """快取未過期時，不應觸發任何下載。"""
    cache._github_registry = KBRegistry(updated_at="warm")
    cache._fetched_at = time.time()

    calls = []

    async def fake_fetch(self):
        calls.append(1)

    monkeypatch.setattr(FederationCache, "_fetch", fake_fetch)

    result = await cache.get_remote_registry()
    assert result.updated_at == "warm"
    assert calls == []


async def test_expired_cache_returns_stale_immediately_and_refreshes_in_background(cache, monkeypatch):
    """
    快取過期時，get_remote_registry() 不可阻塞等待 GitHub 下載完成 ——
    必須立即回傳舊資料，並在背景排入刷新任務。
    """
    cache._github_registry = KBRegistry(updated_at="stale")
    cache._fetched_at = time.time() - 999999  # 遠早於 TTL，視為過期

    refreshed = asyncio.Event()

    async def fake_fetch(self):
        self._github_registry = KBRegistry(updated_at="fresh")
        self._fetched_at = time.time()
        refreshed.set()

    monkeypatch.setattr(FederationCache, "_fetch", fake_fetch)

    result = await cache.get_remote_registry()
    assert result.updated_at == "stale"  # 立即回傳，不等待背景刷新

    await asyncio.wait_for(refreshed.wait(), timeout=1.0)
    assert cache._github_registry.updated_at == "fresh"


async def test_expired_cache_does_not_spawn_duplicate_background_refreshes(cache, monkeypatch):
    """快取過期時連續呼叫多次，只應排入一個背景刷新任務。"""
    cache._github_registry = KBRegistry(updated_at="stale")
    cache._fetched_at = time.time() - 999999

    call_count = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_fetch(self):
        nonlocal call_count
        call_count += 1
        started.set()
        await release.wait()
        self._github_registry = KBRegistry(updated_at="fresh")
        self._fetched_at = time.time()

    monkeypatch.setattr(FederationCache, "_fetch", fake_fetch)

    await cache.get_remote_registry()
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await cache.get_remote_registry()
    await cache.get_remote_registry()

    release.set()
    await asyncio.sleep(0.05)
    assert call_count == 1


async def test_force_refresh_blocks_until_fetch_completes(cache, monkeypatch):
    """force_refresh() 供管理端點使用，必須同步等待下載完成才回傳。"""

    async def fake_fetch(self):
        await asyncio.sleep(0.01)
        self._github_registry = KBRegistry(updated_at="forced")
        self._fetched_at = time.time()

    monkeypatch.setattr(FederationCache, "_fetch", fake_fetch)

    result = await cache.force_refresh()
    assert result.updated_at == "forced"
