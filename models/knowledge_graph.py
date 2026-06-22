from __future__ import annotations
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


# ── KnowledgeGraph CRUD ───────────────────────────────────────────────────────

class KnowledgeGraphCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    owner_id: str = "default"
    is_public: bool = True


class KnowledgeGraphUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = None
    is_public: bool | None = None


class KnowledgeGraph(BaseModel):
    id: UUID
    name: str
    description: str
    folder_path: str
    owner_id: str
    is_public: bool
    db_name: str = ""          # Neo4j 專用資料庫名稱（空白 = 使用主資料庫）
    doc_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    created_at: datetime
    updated_at: datetime


class KnowledgeGraphDetail(KnowledgeGraph):
    top_concepts: list[str] = []
    top_entities: list[str] = []


# ── 文件分配 ──────────────────────────────────────────────────────────────────

class KGCandidate(BaseModel):
    kg_id: UUID
    kg_name: str
    score: float
    top_matched_concepts: list[str] = []


class ClassifyRequest(BaseModel):
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    auto_assign: bool = False
    owner_id: str = "default"


class ClassifyResult(BaseModel):
    txt_filename: str
    candidates: list[KGCandidate] = []
    matched_kg_id: UUID | None = None
    matched_kg_name: str | None = None
    score: float = 0.0
    auto_assigned: bool = False
    status: str = "pending"   # pending | assigned | unmatched


class AssignRequest(BaseModel):
    kg_id: UUID


class MoveDocumentRequest(BaseModel):
    doc_id: UUID
    from_kg_id: UUID | None = None
    to_kg_id: UUID
    move_file: bool = True


# ── Staging 分群建議 ──────────────────────────────────────────────────────────

class ClusterSuggestion(BaseModel):
    suggested_name: str
    suggested_description: str
    files: list[str]
    top_concepts: list[str]
    intra_similarity: float


class ApproveSuggestionRequest(BaseModel):
    suggested_name: str
    suggested_description: str = ""
    files: list[str]


# ── SVO 知識層 ────────────────────────────────────────────────────────────────

class SVOTriple(BaseModel):
    subject: str
    subject_type: str = "概念"
    rel_type: str = "RELATED_TO"   # 語意關係類別
    verb: str                       # 原始動詞描述（保留自然語言）
    object: str
    object_type: str = "概念"
    confidence: int = 1
    source_doc_id: UUID | None = None


class BuildGraphRequest(BaseModel):
    doc_ids: list[UUID] | None = None
    force_rebuild: bool = False
