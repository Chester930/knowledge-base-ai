"""
models/document.py 輸入驗證測試。

防禦縱深：DocumentCreate.content 等欄位原本完全無長度上限，
POST /documents（JSON body，非檔案上傳）可繞過上傳路徑的大小限制
塞入任意大小字串。這裡驗證各欄位的長度邊界。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.document import (
    AgentQueryRequest,
    ChatMessage,
    ChatRequest,
    DocumentCreate,
    SearchRequest,
)


class TestDocumentCreate:
    def test_valid_document_accepted(self):
        doc = DocumentCreate(title="標題", content="內容")
        assert doc.title == "標題"

    def test_empty_title_rejected(self):
        with pytest.raises(ValidationError):
            DocumentCreate(title="", content="內容")

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            DocumentCreate(title="標題", content="")

    def test_title_over_limit_rejected(self):
        with pytest.raises(ValidationError):
            DocumentCreate(title="x" * 501, content="內容")

    def test_title_at_limit_accepted(self):
        doc = DocumentCreate(title="x" * 500, content="內容")
        assert len(doc.title) == 500

    def test_content_over_limit_rejected(self):
        with pytest.raises(ValidationError):
            DocumentCreate(title="標題", content="x" * 20_000_001)


class TestSearchRequest:
    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(text="")

    def test_text_over_limit_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(text="x" * 4001)

    def test_valid_text_accepted(self):
        req = SearchRequest(text="查詢")
        assert req.text == "查詢"


class TestAgentQueryRequest:
    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            AgentQueryRequest(question="")

    def test_question_over_limit_rejected(self):
        with pytest.raises(ValidationError):
            AgentQueryRequest(question="x" * 4001)


class TestChatRequest:
    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="")

    def test_question_over_limit_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(question="x" * 4001)

    def test_history_over_length_limit_rejected(self):
        history = [ChatMessage(role="user", content="嗨") for _ in range(51)]
        with pytest.raises(ValidationError):
            ChatRequest(question="問題", history=history)

    def test_history_at_limit_accepted(self):
        history = [ChatMessage(role="user", content="嗨") for _ in range(50)]
        req = ChatRequest(question="問題", history=history)
        assert len(req.history) == 50

    def test_chat_message_content_over_limit_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="x" * 20_001)
