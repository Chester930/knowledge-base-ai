from __future__ import annotations
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from neo4j import AsyncDriver

from core.config import settings
from models.kb_skill import ConceptScore, KBRegistry, KBSkill
from repositories.concept_repo import ConceptRepository
from repositories.knowledge_graph_repo import KnowledgeGraphRepository

logger = logging.getLogger(__name__)


# ── Registry I/O ──────────────────────────────────────────────────────────────

def _registry_path() -> Path:
    return Path(settings.registry_path)


def load_registry() -> KBRegistry:
    p = _registry_path()
    if p.exists():
        try:
            return KBRegistry(**json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning(f"registry.json 讀取失敗，重置：{e}")
    return KBRegistry(updated_at=_now())


def save_registry(registry: KBRegistry) -> None:
    """
    原子寫入 registry.json：先寫暫存檔再 os.replace()，避免程式在寫入中途
    崩潰（斷電、OOM-kill）留下截斷的 JSON，導致 load_registry() 靜默重置為空
    registry，讓 world federation 探索、跨 instance KB Skill 同步等功能全部失效。
    """
    registry.updated_at = _now()
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(registry.model_dump_json(indent=2, exclude_none=True))
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── 單筆 Skill 操作 ───────────────────────────────────────────────────────────

def upsert_skill(skill: KBSkill) -> None:
    registry = load_registry()
    registry.skills = [s for s in registry.skills if s.kb_id != skill.kb_id]
    registry.skills.append(skill)
    save_registry(registry)
    logger.info(f"KB Skill 已更新：{skill.name}（{skill.kb_id}）")


def remove_skill(kb_id: str) -> None:
    registry = load_registry()
    before = len(registry.skills)
    registry.skills = [s for s in registry.skills if s.kb_id != kb_id]
    if len(registry.skills) < before:
        save_registry(registry)
        logger.info(f"KB Skill 已移除：{kb_id}")


# ── 從本機 KG 生成描述檔 ──────────────────────────────────────────────────────

async def generate_skill(kg_id: UUID, driver: AsyncDriver) -> KBSkill:
    """從本機 Neo4j 讀取 KG 資訊，生成 KB Skill 描述檔。"""
    kg_repo = KnowledgeGraphRepository(driver)
    concept_repo = ConceptRepository(driver)

    kg = await kg_repo.get_by_id(kg_id)
    if not kg:
        raise ValueError(f"KG 不存在：{kg_id}")

    # 讀取 top ConceptNode（含 embedding 向量，用於計算指紋）
    raw_concepts = await concept_repo.get_kg_concepts(kg_id)
    top_concepts: list[ConceptScore] = []
    all_vectors: list[list[float]] = []

    for c in raw_concepts[:20]:
        score = (
            (c.get("interest_score") or 0) + (c.get("professional_score") or 0)
        ) / 2
        top_concepts.append(ConceptScore(
            name=c["name"],
            score=round(score, 4),
        ))
        vec = c.get("q_vector") or []
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        if vec:
            all_vectors.append(list(vec))

    top_concepts.sort(key=lambda x: x.score, reverse=True)

    # 計算指紋向量：所有 concept 向量的平均（4 位小數，大幅縮減 registry 大小）
    fingerprint: list[float] = []
    if all_vectors:
        dim = len(all_vectors[0])
        fingerprint = [
            round(sum(v[i] for v in all_vectors) / len(all_vectors), 4)
            for i in range(dim)
        ]

    return KBSkill(
        instance_id=settings.instance_id,
        kb_id=str(kg_id),
        name=kg.name,
        description=kg.description or "",
        language="zh-TW",
        last_sync=_now(),
        is_local=True,
        db_name=kg.db_name or None,
        tags=[],
        top_concepts=top_concepts,
        fingerprint_vector=fingerprint,
        entity_count=kg.entity_count,
        relation_count=kg.relation_count,
        document_count=kg.doc_count,
    )


# ── 批次同步所有公開 KG ────────────────────────────────────────────────────────

async def sync_public_kgs(driver: AsyncDriver) -> dict:
    """
    掃描所有公開 KG，生成並更新 registry.json 中的 KB Skill 描述檔。
    私有化的 KG 會從 registry 中移除。
    回傳 { synced, removed, errors }。
    """
    kg_repo = KnowledgeGraphRepository(driver)
    all_kgs = await kg_repo.list_all(include_private=True)
    public_kgs = [kg for kg in all_kgs if kg.is_public]
    public_ids = {str(kg.id) for kg in public_kgs}

    synced, removed, errors = 0, 0, []

    # 同步公開 KG
    for kg in public_kgs:
        try:
            skill = await generate_skill(kg.id, driver)
            upsert_skill(skill)
            synced += 1
        except Exception as e:
            msg = f"{kg.name}：{e}"
            logger.warning(f"KB Skill 生成失敗 — {msg}")
            errors.append(msg)

    # 移除已私有化的本機 KG
    registry = load_registry()
    for skill in registry.skills:
        if skill.is_local and skill.kb_id not in public_ids:
            remove_skill(skill.kb_id)
            removed += 1

    logger.info(f"sync_public_kgs 完成：同步 {synced}，移除 {removed}，錯誤 {len(errors)}")
    return {"synced": synced, "removed": removed, "errors": errors}
