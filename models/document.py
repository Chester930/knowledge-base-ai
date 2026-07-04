from __future__ import annotations
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    # 20,000,000 字元上限（約可容納大型 PDF/DOCX 轉出的純文字），
    # 防止 JSON body 直接繞過檔案上傳路徑的大小限制塞入超大內容
    content: str = Field(..., min_length=1, max_length=20_000_000)
    file_path: str | None = None
    file_type: Literal["md", "txt", "pdf", "manual"] = "manual"
    kg_id: UUID | None = None  # 建立後自動關聯到指定 KG


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
    text: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    document: Document
    score: float
    matched_concepts: list[str]


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
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


class ChatMessage(BaseModel):
    role: str    # "user" | "assistant"
    content: str = Field(..., max_length=20_000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)
    max_chars_per_doc: int = Field(default=2000, ge=500, le=12000)
    owner_id: str = "default"
    use_svo: bool = True
    svo_hops: int = Field(default=2, ge=1, le=3)
    history: list[ChatMessage] | None = Field(default=None, max_length=50)  # 多輪對話歷史（☆10）
    kg_id: UUID | None = None  # 指定時強制路由到此 KG，跳過全域路由
