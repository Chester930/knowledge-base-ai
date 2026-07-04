"""
Federation Service — Phase 2b: GitHub Registry

啟動時從 GitHub 下載 registry.json，快取遠端分片連線池。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx
from neo4j import AsyncGraphDatabase, AsyncDriver

from core.config import settings
from models.kb_skill import KBRegistry, KBSkill
from services.kb_skill_service import load_registry

logger = logging.getLogger(__name__)

_CACHE_TTL = 1800  # 30 分鐘


class FederationCache:
    """GitHub registry 快取 + 遠端 AuraDB 連線池"""

    def __init__(self) -> None:
        self._github_registry: Optional[KBRegistry] = None
        self._fetched_at: float = 0.0
        self._drivers: dict[str, AsyncDriver] = {}   # kb_id -> AsyncDriver
        self._shard_status: dict[str, str] = {}      # kb_id -> "online" | "offline" | "local"
        self._offline_timestamps: dict[str, float] = {}  # kb_id -> timestamp (熔斷冷卻時間戳)
        self._lock = asyncio.Lock()
        self._refreshing = False   # 避免快取過期時同時排入多個背景刷新任務

    # ── GitHub Registry ───────────────────────────────────────────────────────

    async def get_remote_registry(self) -> KBRegistry:
        """
        回傳 GitHub 遠端 registry（快取 30 分鐘）。
        快取過期時「不」同步等待下載完成 —— 直接回傳目前快取（可能稍舊），
        並在背景排入一次刷新，避免 /world/chat 等熱路徑卡在 GitHub HTTP 請求上。
        僅在完全冷啟動（尚無任何快取）時才同步等待一次。
        """
        if self._github_registry is None:
            async with self._lock:
                if self._github_registry is None:
                    await self._fetch()
            return self._github_registry or KBRegistry(updated_at="")

        if (time.time() - self._fetched_at) >= _CACHE_TTL and not self._refreshing:
            self._refreshing = True
            asyncio.create_task(self._background_refresh())

        return self._github_registry

    async def _background_refresh(self) -> None:
        try:
            async with self._lock:
                await self._fetch()
        finally:
            self._refreshing = False

    async def force_refresh(self) -> KBRegistry:
        """強制同步刷新（供管理端點 /world/federation/refresh 呼叫，會阻塞直到完成）。"""
        async with self._lock:
            await self._fetch()
        return self._github_registry or KBRegistry(updated_at="")

    async def _fetch(self) -> None:
        url = settings.github_registry_url
        if not url:
            logger.debug("GITHUB_REGISTRY_URL 未設定，跳過遠端 registry 下載")
            if self._github_registry is None:
                self._github_registry = KBRegistry(updated_at="")
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                self._github_registry = KBRegistry(**data)
                self._fetched_at = time.time()
                logger.info(
                    f"GitHub registry 下載成功：{len(self._github_registry.skills)} 個 KB Skill"
                )
        except Exception as e:
            logger.warning(f"GitHub registry 下載失敗：{e}")
            if self._github_registry is None:
                self._github_registry = KBRegistry(updated_at="")

    # ── 合併本機 + 遠端 registry ──────────────────────────────────────────────

    async def merged_registry(self) -> KBRegistry:
        """合併本機 registry（優先）+ GitHub 遠端 registry。"""
        local = load_registry()
        remote = await self.get_remote_registry()

        local_ids = {s.kb_id for s in local.skills}
        merged_skills = list(local.skills)
        for skill in remote.skills:
            if skill.kb_id not in local_ids:
                merged_skills.append(skill)

        from datetime import datetime, timezone
        return KBRegistry(
            version=local.version,
            updated_at=datetime.now(timezone.utc).isoformat(),
            skills=merged_skills,
        )

    # ── AuraDB 連線池 ─────────────────────────────────────────────────────────

    async def get_shard_driver(self, skill: KBSkill) -> Optional[AsyncDriver]:
        """取得（或建立）遠端 AuraDB 的 AsyncDriver。本機分片回傳 None。"""
        if skill.is_local or not skill.aura_uri:
            return None

        # 熔斷保護：若處於 offline 且距離上次失敗小於 60 秒，則直接略過連線嘗試
        if self._shard_status.get(skill.kb_id) == "offline":
            last_fail = self._offline_timestamps.get(skill.kb_id, 0.0)
            if time.time() - last_fail < 60.0:
                logger.debug(f"AuraDB 處於熔斷冷卻期，略過連線：{skill.name} [{skill.instance_id}]")
                return None

        if skill.kb_id not in self._drivers:
            try:
                auth = (skill.read_token or "neo4j", "")
                driver = AsyncGraphDatabase.driver(skill.aura_uri, auth=auth)
                await driver.verify_connectivity()
                self._drivers[skill.kb_id] = driver
                self._shard_status[skill.kb_id] = "online"
                self._offline_timestamps.pop(skill.kb_id, None)
                logger.info(f"AuraDB 連線建立：{skill.name} ({skill.instance_id})")
            except Exception as e:
                logger.warning(f"AuraDB 連線失敗 [{skill.name}]：{e}")
                self._shard_status[skill.kb_id] = "offline"
                self._offline_timestamps[skill.kb_id] = time.time()
                return None

        return self._drivers.get(skill.kb_id)

    def mark_shard_offline(self, kb_id: str) -> None:
        """標記某個分片為離線（查詢逾時或失敗時呼叫）。"""
        self._shard_status[kb_id] = "offline"
        self._offline_timestamps[kb_id] = time.time()
        driver = self._drivers.pop(kb_id, None)
        if driver:
            asyncio.create_task(driver.close())

    # ── 狀態資訊 ──────────────────────────────────────────────────────────────

    def status(self) -> dict:
        import datetime
        fetched_iso = (
            datetime.datetime.fromtimestamp(self._fetched_at).isoformat()
            if self._fetched_at else None
        )
        remote_skills = self._github_registry.skills if self._github_registry else []
        local = load_registry()
        local_ids = {s.kb_id for s in local.skills}

        shards = []
        for skill in remote_skills:
            is_local = skill.kb_id in local_ids
            shards.append({
                "kb_id": skill.kb_id,
                "name": skill.name,
                "instance_id": skill.instance_id,
                "is_local": is_local,
                "status": "local" if is_local else self._shard_status.get(skill.kb_id, "pending"),
                "entity_count": skill.entity_count,
                "relation_count": skill.relation_count,
            })

        return {
            "github_registry_url": settings.github_registry_url or None,
            "last_fetched": fetched_iso,
            "cache_ttl_seconds": _CACHE_TTL,
            "remote_skill_count": len(remote_skills),
            "active_connections": len(self._drivers),
            "shards": shards,
        }

    async def close(self) -> None:
        for driver in list(self._drivers.values()):
            try:
                await driver.close()
            except Exception:
                pass
        self._drivers.clear()
        logger.info("Federation 連線池已關閉")


# ── 單例 ──────────────────────────────────────────────────────────────────────

_cache: Optional[FederationCache] = None


def get_federation_cache() -> FederationCache:
    global _cache
    if _cache is None:
        _cache = FederationCache()
    return _cache


# ── Lifespan hooks ────────────────────────────────────────────────────────────

async def startup_prefetch() -> None:
    """應用啟動時，背景預先下載 GitHub registry（不阻塞啟動流程）。"""
    cache = get_federation_cache()
    asyncio.create_task(cache.get_remote_registry())
    logger.info("Phase 2b: GitHub registry 預取任務已排隊")


async def shutdown_cleanup() -> None:
    """應用關閉時，清理遠端 AuraDB 連線。"""
    global _cache
    if _cache:
        await _cache.close()
