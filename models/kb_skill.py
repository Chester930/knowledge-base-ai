from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class ConceptScore(BaseModel):
    name: str
    score: float
    # 注意：不在 registry 中儲存個別向量（太大），改用 fingerprint_vector 聚合


class KBSkill(BaseModel):
    """KB Skill 描述檔 — 記錄單一公開 KG 的路由 metadata 與連線資訊"""
    instance_id: str                 # 來源 instance 識別碼（防止實體命名衝突）
    kb_id: str                       # KG 的 UUID
    name: str
    description: str = ""
    language: str = "zh-TW"
    last_sync: str                   # ISO 8601

    # ── 連線資訊 ──────────────────────────────────────────────────────────────
    is_local: bool = True            # True = 本機模擬；False = 真實 AuraDB
    db_name: Optional[str] = None   # 本機模式：Neo4j 資料庫名稱（空白 = 主庫）
    aura_uri: Optional[str] = None  # 遠端模式：AuraDB URI
    read_token: Optional[str] = None # 遠端模式：read-only token

    # ── Agent 路由用 metadata（無需 DB 連線即可判斷相關性）────────────────────
    tags: list[str] = []
    top_concepts: list[ConceptScore] = []   # 名稱 + score（供人類閱讀與關鍵字比對）
    fingerprint_vector: list[float] = []    # 所有 concept 向量的平均，供 embedding 路由

    # ── 統計（讓 Agent 評估 KB 深度）─────────────────────────────────────────
    entity_count: int = 0
    relation_count: int = 0
    document_count: int = 0


class KBRegistry(BaseModel):
    """registry.json 的根結構"""
    version: str = "1.0"
    updated_at: str
    skills: list[KBSkill] = []
