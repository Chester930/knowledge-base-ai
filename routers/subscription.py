"""
KG Subscription Router — Phase 3d

訂閱管理 API：
- GET  /world/subscriptions             列出所有訂閱
- POST /world/subscribe                 新增訂閱
- DELETE /world/subscribe/{kb_id}       取消訂閱
- PATCH /world/subscribe/{kb_id}/pause  暫停 / 恢復訂閱
- POST /world/sync-subscriptions        手動觸發全部同步
- POST /world/sync-subscriptions/{kb_id} 同步單一訂閱
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.subscription_service import (
    Subscription,
    get_subscription_manager,
    sync_all_subscriptions,
    sync_subscription,
)

router = APIRouter(prefix="/world", tags=["subscriptions"])
logger = logging.getLogger(__name__)


# ── 請求模型 ──────────────────────────────────────────────────────────────────

class SubscribeRequest(BaseModel):
    instance_id: str
    kb_id: str
    kb_name: str
    aura_uri: str
    read_token: str = ""
    local_kg_id: str = ""
    sync_interval_hours: int = 6


class PauseRequest(BaseModel):
    paused: bool


# ── 路由 ──────────────────────────────────────────────────────────────────────

@router.get("/subscriptions", summary="列出所有訂閱（Phase 3d）")
async def list_subscriptions():
    manager = get_subscription_manager()
    subs = await manager.list_all()
    return {
        "count": len(subs),
        "subscriptions": [
            {
                "instance_id": s.instance_id,
                "kb_id": s.kb_id,
                "kb_name": s.kb_name,
                "aura_uri": s.aura_uri[:30] + "…" if len(s.aura_uri) > 30 else s.aura_uri,
                "last_sync_at": s.last_sync_at,
                "sync_interval_hours": s.sync_interval_hours,
                "status": s.status,
                "error_msg": s.error_msg,
                "local_kg_id": s.local_kg_id,
            }
            for s in subs
        ],
    }


@router.post("/subscribe", status_code=201, summary="新增 KG 訂閱（Phase 3d）")
async def subscribe(body: SubscribeRequest):
    manager = get_subscription_manager()
    sub = Subscription(
        instance_id=body.instance_id,
        kb_id=body.kb_id,
        kb_name=body.kb_name,
        aura_uri=body.aura_uri,
        read_token=body.read_token,
        local_kg_id=body.local_kg_id,
        sync_interval_hours=body.sync_interval_hours,
    )
    try:
        await manager.add(sub)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"message": f"已訂閱 {body.kb_name}", "kb_id": body.kb_id}


@router.delete("/subscribe/{kb_id}", summary="取消 KG 訂閱（Phase 3d）")
async def unsubscribe(kb_id: str):
    manager = get_subscription_manager()
    removed = await manager.remove(kb_id)
    if not removed:
        raise HTTPException(status_code=404, detail="訂閱不存在")
    return {"message": "已取消訂閱", "kb_id": kb_id}


@router.patch("/subscribe/{kb_id}/pause", summary="暫停 / 恢復訂閱（Phase 3d）")
async def pause_subscription(kb_id: str, body: PauseRequest):
    manager = get_subscription_manager()
    sub = await manager.get_by_kb_id(kb_id)
    if not sub:
        raise HTTPException(status_code=404, detail="訂閱不存在")
    new_status = "paused" if body.paused else "active"
    await manager.set_status(kb_id, new_status)
    return {"kb_id": kb_id, "status": new_status}


@router.post("/sync-subscriptions", summary="手動觸發全部訂閱同步（Phase 3d）")
async def sync_all():
    results = await sync_all_subscriptions()
    total_merged = sum(r.get("merged", 0) for r in results)
    errors = [r for r in results if r.get("error")]
    return {
        "synced": len(results),
        "total_merged": total_merged,
        "errors": len(errors),
        "results": results,
    }


@router.post("/sync-subscriptions/{kb_id}", summary="同步單一訂閱（Phase 3d）")
async def sync_one(kb_id: str):
    manager = get_subscription_manager()
    sub = await manager.get_by_kb_id(kb_id)
    if not sub:
        raise HTTPException(status_code=404, detail="訂閱不存在")
    result = await sync_subscription(sub)
    if result["error"]:
        await manager.set_status(kb_id, "error", result["error"])
    else:
        from services.subscription_service import _now_iso
        await manager.set_last_sync(kb_id, _now_iso())
    return {"kb_id": kb_id, "kb_name": sub.kb_name, **result}
