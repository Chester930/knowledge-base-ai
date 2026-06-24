from __future__ import annotations
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal


class DocumentCreate(BaseModel):
    title: str
    content: str
    file_path: str | None = None
    file_type: Literal["md", "txt", "pdf", "manual"] = "manual"


class Document(BaseModel):
    id: UUID
    title: str
    content: str
    file_path: str | None = None
    file_type: str
    created_at: datetime
    updated_at: datetime


class DocumentConcept(BaseModel):
    concept_id: UUID
    name: str
    domain: str
    interest_score: float
    professional_score: float


class SearchRequest(BaseModel):
    text: str
    top_k: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    document: Document
    score: float
    matched_concepts: list[str]


class AgentQueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    include_content: bool = True
    max_content_chars: int = Field(default=2000, ge=100, le=10000)


class AgentContext(BaseModel):
    title: str
    content_snippet: str
    score: float
    file_path: str | None = None


class AgentQueryResponse(BaseModel):
    question: str
    context: list[AgentContext]
    sources: list[str]


class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=10)
    max_chars_per_doc: int = Field(default=8000, ge=500, le=12000)
    owner_id: str = "default"
    use_svo: bool = True          # False 可停用 SVO 知識層（純文件 RAG）
    svo_hops: int = Field(default=2, ge=1, le=3)
