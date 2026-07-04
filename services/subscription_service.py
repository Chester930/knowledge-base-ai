"""
KG Subscription Service — Phase 3d

管理本機訂閱清單（subscriptions.json），提供：
- SubscriptionManager：讀寫訂閱設定、觸發同步
- sync_subscription()：從遠端 AuraDB 拉取 SVO 事實並寫入本機
- sync_all_subscriptions()：同步所有 active 訂閱（APScheduler 定時呼叫）
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SUBSCRIPTIONS_FILE = Path("subscriptions.json")

_ALL_REL = (
    "IS_A|PART_OF|CONTAINS|INSTANCE_OF|CAUSES|PREVENTS|ENABLES|IMPROVES|INHIBITS|"
    "USES|REQUIRES|PRODUCES|IMPLEMENTS|REPLACES|EXTENDS|CONTRASTS|SIMILAR_TO|OUTPERFORMS|"
    "DEFINED_AS|HAS_PROPERTY|MEASURED_BY|APPLIES_TO|PRECEDES|FOLLOWS|CO_OCCURS|"
    "INPUTS|TRANSFORMS|CREATED_BY|SOLVES|RELATED_TO"
)


@dataclass
class Subscription:
    instance_id: str       # 遠端 instance 擁有者 ID
    kb_id: str             # 遠端 KG 的 UUID（在遠端 AuraDB 上的 kg_id）
    kb_name: str           # 顯示名稱
    aura_uri: str          # 遠端 AuraDB 連線 URI
    read_token: str = ""   # 可選 read-only token
    last_sync_at: str = "" # 上次成功同步時間（ISO 8601）
    sync_interval_hours: int = 6  # 自動同步間隔（小時）
    status: str = "active" # active | paused | error
    error_msg: str = ""
    local_kg_id: str = ""  # 對應本機 KG UUID（本機存放位置）


class SubscriptionManager:
    """單例：管理 subscriptions.json 讀寫與同步狀態。"""

    _instance: Optional["SubscriptionManager"] = None

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._subs: list[Subscription] = []
        self._loaded = False

    @classmethod
    def get(cls) -> "SubscriptionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            await self._load()
            self._loaded = True

    async def _load(self) -> None:
        if not _SUBSCRIPTIONS_FILE.exists():
            self._subs = []
            return
        try:
            data = json.loads(_SUBSCRIPTIONS_FILE.read_text(encoding="utf-8"))
            self._subs = [Subscription(**s) for s in data.get("subscriptions", [])]
            logger.info(f"訂閱清單載入：{len(self._subs)} 筆")
        except Exception as e:
            logger.error(f"載入 subscriptions.json 失敗：{e}")
            self._subs = []

    async def _save(self) -> None:
        """
        原子寫入 subscriptions.json：先寫暫存檔再 os.replace()，避免程式在
        寫入中途崩潰留下截斷的 JSON（與 kb_skill_service.py 的 save_registry()
        同一套修法）。
        """
        data = {
            "version": 1,
            "updated_at": _now_iso(),
            "subscriptions": [asdict(s) for s in self._subs],
        }
        content = json.dumps(data, ensure_ascii=False, indent=2)

        _SUBSCRIPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(_SUBSCRIPTIONS_FILE.parent) or ".",
            prefix=f".{_SUBSCRIPTIONS_FILE.name}.", suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_name, _SUBSCRIPTIONS_FILE)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    async def list_all(self) -> list[Subscription]:
        await self._ensure_loaded()
        return list(self._subs)

    async def get_by_kb_id(self, kb_id: str) -> Optional[Subscription]:
        await self._ensure_loaded()
        return next((s for s in self._subs if s.kb_id == kb_id), None)

    async def add(self, sub: Subscription) -> None:
        await self._ensure_loaded()
        async with self._lock:
            if any(s.kb_id == sub.kb_id for s in self._subs):
                raise ValueError(f"已訂閱 kb_id={sub.kb_id}")
            self._subs.append(sub)
            await self._save()

    async def remove(self, kb_id: str) -> bool:
        await self._ensure_loaded()
        async with self._lock:
            before = len(self._subs)
            self._subs = [s for s in self._subs if s.kb_id != kb_id]
            if len(self._subs) < before:
                await self._save()
                return True
            return False

    async def _update_sub(self, kb_id: str, **kwargs) -> None:
        async with self._lock:
            for s in self._subs:
                if s.kb_id == kb_id:
                    for k, v in kwargs.items():
                        setattr(s, k, v)
            await self._save()

    async def set_status(self, kb_id: str, status: str, error_msg: str = "") -> None:
        await self._update_sub(kb_id, status=status, error_msg=error_msg)

    async def set_last_sync(self, kb_id: str, ts: str) -> None:
        await self._update_sub(kb_id, last_sync_at=ts, status="active", error_msg="")


def get_subscription_manager() -> SubscriptionManager:
    return SubscriptionManager.get()


# ── 同步單一訂閱 ───────────────────────────────────────────────────────────────

async def sync_subscription(sub: Subscription) -> dict:
    """
    從遠端 AuraDB 拉取 SVO 事實，寫入本機 Neo4j。
    回傳 {"merged": N, "error": "..." | None}
    """
    from services.federation_service import get_federation_cache
    from models.kb_skill import KBSkill
    from core.database import get_driver

    logger.info(f"開始同步訂閱：{sub.kb_name} [{sub.instance_id}]")

    # 取得遠端 AuraDB driver
    skill = KBSkill(
        instance_id=sub.instance_id,
        kb_id=sub.kb_id,
        name=sub.kb_name,
        is_local=False,
        aura_uri=sub.aura_uri,
        read_token=sub.read_token,
    )
    cache = get_federation_cache()
    remote_driver = await cache.get_shard_driver(skill)
    if remote_driver is None:
        return {"merged": 0, "error": "無法連線遠端 AuraDB"}

    # 從遠端拉取所有 SVO 三元組（分批）
    BATCH = 500
    offset = 0
    all_rows: list[dict] = []
    try:
        while True:
            result = await remote_driver.execute_query(
                f"""
                MATCH (s:Entity)-[r:{_ALL_REL}]->(o:Entity)
                WHERE s.kg_id = $kb_id OR $kb_id IS NULL
                RETURN
                    s.name AS subject, s.type AS subject_type,
                    type(r) AS rel_type, r.verb AS verb,
                    o.name AS object, o.type AS object_type,
                    r.source_doc_id AS source_doc_id
                SKIP $skip LIMIT $lim
                """,
                kb_id=sub.kb_id, skip=offset, lim=BATCH,
            )
            rows = result.records
            if not rows:
                break
            all_rows.extend(rows)
            offset += BATCH
            if len(rows) < BATCH:
                break
    except Exception as e:
        return {"merged": 0, "error": f"遠端查詢失敗：{e}"}

    if not all_rows:
        return {"merged": 0, "error": None}

    # 寫入本機（本機 KG 以 instance_id+kb_id 命名空間隔離）
    local_driver = get_driver()
    local_kg_id = sub.local_kg_id or f"{sub.instance_id}::{sub.kb_id}"

    merged_total = 0
    # 按 rel_type 分批寫入
    from collections import defaultdict
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_rel[r["rel_type"]].append(r)

    try:
        for rel_type, rows in by_rel.items():
            batch = [
                {
                    "subject": r["subject"],
                    "s_type": r.get("subject_type") or "Entity",
                    "object": r["object"],
                    "o_type": r.get("object_type") or "Entity",
                    "verb": r.get("verb") or rel_type,
                    "doc_id": r.get("source_doc_id") or "",
                }
                for r in rows
            ]
            result = await local_driver.execute_query(
                f"""
                UNWIND $rows AS row
                MERGE (s:Entity {{name: row.subject, kg_id: $kg_id}})
                  ON CREATE SET s.type = row.s_type, s.created_at = datetime()
                  ON MATCH SET  s.type = CASE WHEN s.type IS NULL THEN row.s_type ELSE s.type END
                MERGE (o:Entity {{name: row.object, kg_id: $kg_id}})
                  ON CREATE SET o.type = row.o_type, o.created_at = datetime()
                  ON MATCH SET  o.type = CASE WHEN o.type IS NULL THEN row.o_type ELSE o.type END
                MERGE (s)-[rel:{rel_type} {{source_doc_id: row.doc_id}}]->(o)
                  ON CREATE SET rel.verb = row.verb, rel.confidence = 1, rel.created_at = datetime()
                  ON MATCH SET  rel.confidence = rel.confidence + 1,
                               rel.verb = row.verb, rel.updated_at = datetime()
                RETURN count(rel) AS merged
                """,
                rows=batch, kg_id=local_kg_id,
            )
            for rec in result.records:
                merged_total += rec.get("merged") or 0
    except Exception as e:
        return {"merged": merged_total, "error": f"本機寫入失敗：{e}"}

    logger.info(f"訂閱同步完成：{sub.kb_name} — 共 {merged_total} 條關係")
    return {"merged": merged_total, "error": None}


# ── 同步所有訂閱 ───────────────────────────────────────────────────────────────

async def sync_all_subscriptions() -> list[dict]:
    """並行同步所有 active 訂閱，回傳每筆結果清單（併發優化）。"""
    manager = get_subscription_manager()
    subs = await manager.list_all()
    active = [s for s in subs if s.status != "paused"]

    if not active:
        logger.info("無 active 訂閱需要同步")
        return []

    # 限制最大並發同步數為 5，防止佔用過多連線 Socket
    sem = asyncio.Semaphore(5)

    async def _sync_one_with_sem(sub: Subscription) -> dict:
        async with sem:
            try:
                res = await asyncio.wait_for(sync_subscription(sub), timeout=60.0)
                if res["error"]:
                    await manager.set_status(sub.kb_id, "error", res["error"])
                    return {"kb_id": sub.kb_id, "kb_name": sub.kb_name, **res}
                else:
                    await manager.set_last_sync(sub.kb_id, _now_iso())
                    return {"kb_id": sub.kb_id, "kb_name": sub.kb_name, **res}
            except asyncio.TimeoutError:
                await manager.set_status(sub.kb_id, "error", "同步超時（60s）")
                return {"kb_id": sub.kb_id, "kb_name": sub.kb_name, "merged": 0, "error": "timeout"}
            except Exception as e:
                await manager.set_status(sub.kb_id, "error", str(e))
                return {"kb_id": sub.kb_id, "kb_name": sub.kb_name, "merged": 0, "error": str(e)}

    tasks = [_sync_one_with_sem(sub) for sub in active]
    results = await asyncio.gather(*tasks)
    return list(results)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
