from __future__ import annotations
import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.providers.llm.anthropic import AnthropicLLMProvider


def _make_provider():
    """
    AnthropicLLMProvider.__init__ 內部使用 `from anthropic import AsyncAnthropic`。
    `anthropic` 是選用套件（不在 requirements.txt，CI 環境未安裝），因此用假模組
    注入 sys.modules，測試不依賴該套件是否實際安裝。
    """
    mock_cls = MagicMock()
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    fake_module = types.ModuleType("anthropic")
    fake_module.AsyncAnthropic = mock_cls

    with patch.dict(sys.modules, {"anthropic": fake_module}):
        provider = AnthropicLLMProvider(api_key="sk-test", model="claude-x")
    return provider, mock_client


class TestGenerate:
    async def test_returns_first_content_block_text(self):
        provider, mock_client = _make_provider()
        message = MagicMock()
        message.content = [MagicMock(text="回應內容")]
        mock_client.messages.create = AsyncMock(return_value=message)

        result = await provider.generate("問題")

        assert result == "回應內容"

    async def test_passes_model_and_max_tokens(self):
        provider, mock_client = _make_provider()
        message = MagicMock()
        message.content = [MagicMock(text="x")]
        mock_client.messages.create = AsyncMock(return_value=message)

        await provider.generate("問題")

        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["model"] == "claude-x"
        assert kwargs["messages"] == [{"role": "user", "content": "問題"}]
        assert kwargs["max_tokens"] == 4096


class TestStream:
    async def test_yields_text_stream_chunks(self):
        provider, mock_client = _make_provider()

        class _FakeStreamCtx:
            async def __aenter__(self):
                async def _gen():
                    for tok in ["a", "b", "c"]:
                        yield tok
                self.text_stream = _gen()
                return self
            async def __aexit__(self, *args):
                return False

        mock_client.messages.stream = MagicMock(return_value=_FakeStreamCtx())

        tokens = [t async for t in provider.stream("問題")]

        assert tokens == ["a", "b", "c"]
