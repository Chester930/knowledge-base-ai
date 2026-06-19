from __future__ import annotations
import json
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from core.database import get_driver
from models.knowledge_graph import (
    BuildGraphRequest,
    KnowledgeGraph,
    KnowledgeGraphCreate,
    KnowledgeGraphDetail,
    KnowledgeGraphUpdate,
)
from repositories.knowledge_graph_repo import KnowledgeGraphRepository
from pydantic import BaseModel
from services.knowledge_graph_service import (
    auto_cluster_kgs, confirm_auto_cluster, create_kg, delete_kg, refresh_kg_concepts,
)
from services.svo_service import apply_type_labels, build_graph_for_kg, get_kg_graph

router = APIRouter(prefix="/knowledge-graphs", tags=["knowledge-graphs"])
logger = logging.getLogger(__name__)


@router.post("", response_model=KnowledgeGraph, status_code=201, summary="建立 KG")
async def create(body: KnowledgeGraphCreate):
    try:
        return await create_kg(
            name=body.name,
            description=body.description,
            owner_id=body.owner_id,
            is_public=body.is_public,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("", response_model=list[KnowledgeGraph], summary="列出所有 KG")
async def list_kgs(
    owner_id: str | None = Query(default=None, description="過濾特定擁有者"),
    include_private: bool = Query(default=False),
):
    repo = KnowledgeGraphRepository(get_driver())
    return await repo.list_all(owner_id=owner_id, include_private=include_private)


@router.get("/{kg_id}", response_model=KnowledgeGraphDetail, summary="取得 KG 詳情")
async def get_detail(kg_id: UUID):
    repo = KnowledgeGraphRepository(get_driver())
    detail = await repo.get_detail(kg_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"KG 不存在：{kg_id}")
    return detail


@router.put("/{kg_id}", response_model=KnowledgeGraph, summary="更新 KG")
async def update(kg_id: UUID, body: KnowledgeGraphUpdate):
    repo = KnowledgeGraphRepository(get_driver())
    kg = await repo.update(
        kg_id,
        name=body.name,
        description=body.description,
        is_public=body.is_public,
    )
    if kg is None:
        raise HTTPException(status_code=404, detail=f"KG 不存在：{kg_id}")
    return kg


@router.delete("/{kg_id}", status_code=204, summary="刪除 KG")
async def delete(
    kg_id: UUID,
    delete_files: bool = Query(default=False, description="同時刪除 workspace 資料夾"),
):
    ok = await delete_kg(kg_id, delete_files=delete_files)
    if not ok:
        raise HTTPException(status_code=404, detail=f"KG 不存在：{kg_id}")


@router.get("/{kg_id}/documents", summary="列出 KG 下的文件")
async def list_documents(kg_id: UUID):
    repo = KnowledgeGraphRepository(get_driver())
    kg = await repo.get_by_id(kg_id)
    if kg is None:
        raise HTTPException(status_code=404, detail=f"KG 不存在：{kg_id}")
    docs = await repo.get_documents(kg_id)
    return {"kg_id": str(kg_id), "kg_name": kg.name, "count": len(docs), "documents": docs}


@router.post("/{kg_id}/refresh", status_code=200, summary="刷新 KG 路由層概念")
async def refresh(kg_id: UUID):
    """重新聚合 KG 下所有文件的概念，更新路由層 EFFECTIVE 邊。"""
    repo = KnowledgeGraphRepository(get_driver())
    if not await repo.get_by_id(kg_id):
        raise HTTPException(status_code=404, detail=f"KG 不存在：{kg_id}")
    await refresh_kg_concepts(kg_id)
    return {"status": "ok", "kg_id": str(kg_id)}


# ── SVO 知識層 ────────────────────────────────────────────────────────────────

@router.post("/{kg_id}/build-graph", summary="觸發 SVO 提取（SSE 串流進度）")
async def build_graph(kg_id: UUID, body: BuildGraphRequest = BuildGraphRequest()):
    """
    對 KG 下的文件執行 SVO 三元組提取並 MERGE 進 Neo4j。
    回傳 SSE 串流，依序推送 chunk_start / chunk_done / done / error 事件。

    - `doc_ids`：指定處理的文件，省略則處理全部
    - `force_rebuild`：先清除此 KG 的所有 Entity 再重建
    """
    repo = KnowledgeGraphRepository(get_driver())
    if not await repo.get_by_id(kg_id):
        raise HTTPException(status_code=404, detail=f"KG 不存在：{kg_id}")

    async def event_stream():
        kg = await KnowledgeGraphRepository(get_driver()).get_by_id(kg_id)
        db_name = kg.db_name if kg else ""

        async for progress in build_graph_for_kg(
            kg_id,
            doc_ids=body.doc_ids,
            force_rebuild=body.force_rebuild,
        ):
            payload = {
                "event": progress.event,
                "chunk_idx": progress.chunk_idx,
                "total_chunks": progress.total_chunks,
                "triples_extracted": progress.triples_extracted,
                "triples_merged": progress.triples_merged,
                "message": progress.message,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            if progress.event == "done":
                try:
                    label_stats = await apply_type_labels(kg_id, db_name=db_name)
                    total_labeled = sum(label_stats.values())
                    yield f"data: {json.dumps({'event': 'labels_done', 'labeled': total_labeled, 'stats': label_stats}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    logger.warning(f"標籤套用失敗：{e}")
                    yield f"data: {json.dumps({'event': 'labels_error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{kg_id}/graph", summary="取得 KG 的 Entity + RELATION 清單")
async def get_graph(
    kg_id: UUID,
    limit: int = Query(default=200, ge=1, le=1000),
    min_confidence: int = Query(default=1, ge=1),
):
    """
    回傳 KG 的知識層圖結構：Entity 節點清單 + RELATION 邊清單。
    可用於前端知識圖譜視覺化。
    """
    repo = KnowledgeGraphRepository(get_driver())
    if not await repo.get_by_id(kg_id):
        raise HTTPException(status_code=404, detail=f"KG 不存在：{kg_id}")
    return await get_kg_graph(kg_id, limit=limit, min_confidence=min_confidence)


@router.delete("/{kg_id}/graph", status_code=204, summary="清除 KG 的 SVO 知識層（Entity + RELATION）")
async def clear_graph(kg_id: UUID):
    """刪除此 KG 的所有 Entity 與 RELATION 節點，保留路由層 ConceptNode 與 Document 不動。"""
    from services.svo_service import _clear_kg_entities
    repo = KnowledgeGraphRepository(get_driver())
    kg = await repo.get_by_id(kg_id)
    if not kg:
        raise HTTPException(status_code=404, detail=f"KG 不存在：{kg_id}")
    await _clear_kg_entities(kg_id, kg.db_name)
    await repo.refresh_counts(kg_id)


# ── 自動分群 ──────────────────────────────────────────────────────────────────

class ClusterItem(BaseModel):
    name: str
    description: str = ""
    files: list[str] = []
    doc_ids: list[str] = []


@router.get("/auto-cluster/preview", summary="預覽自動分群方案（LLM 分析暫存區文件）")
async def auto_cluster_preview(
    min_docs: int = Query(default=3, ge=2, description="最少幾份文件才啟動"),
):
    """
    讀取 _staging/ 下所有 .txt，交由 LLM 分群並命名。
    只回傳建議方案，不建立任何 KG。使用者確認後再呼叫 /auto-cluster/confirm。
    """
    try:
        clusters = await auto_cluster_kgs(min_docs=min_docs)
        total = sum(len(c.get("files", [])) + len(c.get("doc_ids", [])) for c in clusters)
        return {"clusters": clusters, "total_files": total}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/auto-cluster/confirm", summary="確認分群方案並建立 KG")
async def auto_cluster_confirm(clusters: list[ClusterItem]):
    """
    接收使用者（可能已編輯過的）分群方案，逐一建立 KG 並分配文件。
    """
    payload = [c.model_dump() for c in clusters]
    results = await confirm_auto_cluster(payload)
    created = sum(1 for r in results if "kg_id" in r)
    return {"created": created, "results": results}
