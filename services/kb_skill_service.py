from __future__ import annotations
import json
import logging
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
    registry.updated_at = _now()
    _registry_path().write_text(
        registry.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )


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

    # 讀取 top ConceptNode（含 embedding 向量）
    raw_concepts = await concept_repo.get_kg_concepts(kg_id)
    top_concepts: list[ConceptScore] = []
    for c in raw_concepts[:20]:
        score = (
            (c.get("interest_score") or 0) + (c.get("professional_score") or 0)
        ) / 2
        vec = c.get("q_vector") or []
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        top_concepts.append(ConceptScore(
            name=c["name"],
            score=round(score, 4),
            vector=list(vec),
        ))
    top_concepts.sort(key=lambda x: x.score, reverse=True)

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
