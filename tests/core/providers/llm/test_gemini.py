"""
GeminiLLMProvider 測試。

重點迴歸案例：stream() 過去只用 asyncio.to_thread 包住「取得 response 物件」
這一步，實際逐 chunk 迭代仍是同步阻塞呼叫（generate_content(stream=True)
回傳的同步 generator），會讓事件迴圈整段卡住。修復後改用
generate_content_async(stream=True) 搭配 `async for`，本檔案驗證：
1. generate()/stream() 都呼叫非同步 API（*_async），不是同步版本
2. stream() 正確透過 async iteration 逐 chunk yield
"""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.providers.llm.gemini import GeminiLLMProvider


class _AsyncChunkIterator:
    """模擬 google-generativeai 的 AsyncGenerateContentResponse 串流回應。"""

    def __init__(self, texts: list[str]):
        self._texts = texts

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for t in self._texts:
            chunk = MagicMock()
            chunk.text = t
            yield chunk


@pytest.fixture
def mock_genai_model():
    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        yield mock_model


class TestGenerate:
    async def test_uses_async_api_not_sync(self, mock_genai_model):
        response = MagicMock()
        response.text = "回答內容"
        mock_genai_model.generate_content_async = AsyncMock(return_value=response)

        provider = GeminiLLMProvider(api_key="fake-key", model="gemini-1.5-flash")
        result = await provider.generate("問題")

        assert result == "回答內容"
        mock_genai_model.generate_content_async.assert_awaited_once_with("問題")
        mock_genai_model.generate_content.assert_not_called()


class TestStream:
    async def test_yields_chunks_via_async_iteration(self, mock_genai_model):
        mock_genai_model.generate_content_async = AsyncMock(
            return_value=_AsyncChunkIterator(["這是", "串流", "回應"])
        )

        provider = GeminiLLMProvider(api_key="fake-key", model="gemini-1.5-flash")
        chunks = [c async for c in provider.stream("問題")]

        assert chunks == ["這是", "串流", "回應"]

    async def test_calls_async_api_with_stream_true_not_sync_generate_content(self, mock_genai_model):
        """回歸測試：不可再呼叫同步的 generate_content(stream=True)。"""
        mock_genai_model.generate_content_async = AsyncMock(
            return_value=_AsyncChunkIterator(["x"])
        )

        provider = GeminiLLMProvider(api_key="fake-key", model="gemini-1.5-flash")
        _ = [c async for c in provider.stream("問題")]

        mock_genai_model.generate_content_async.assert_awaited_once_with("問題", stream=True)
        mock_genai_model.generate_content.assert_not_called()

    async def test_skips_empty_chunks(self, mock_genai_model):
        mock_genai_model.generate_content_async = AsyncMock(
            return_value=_AsyncChunkIterator(["有內容", "", "也有內容"])
        )

        provider = GeminiLLMProvider(api_key="fake-key", model="gemini-1.5-flash")
        chunks = [c async for c in provider.stream("問題")]

        assert chunks == ["有內容", "也有內容"]
