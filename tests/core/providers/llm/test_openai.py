from __future__ import annotations
import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.providers.llm.openai import OpenAILLMProvider


def _make_provider():
    """
    OpenAILLMProvider.__init__ 內部使用 `from openai import AsyncOpenAI`。
    `openai` 是選用套件（不在 requirements.txt，CI 環境未安裝），因此用假模組
    注入 sys.modules，測試不依賴該套件是否實際安裝。
    """
    mock_cls = MagicMock()
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    fake_module = types.ModuleType("openai")
    fake_module.AsyncOpenAI = mock_cls

    with patch.dict(sys.modules, {"openai": fake_module}):
        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-4o-mini")
    return provider, mock_client, mock_cls


class TestInit:
    def test_constructs_client_with_api_key_only(self):
        provider, mock_client, mock_cls = _make_provider()
        mock_cls.assert_called_once_with(api_key="sk-test")
        assert provider.model == "gpt-4o-mini"


class TestGenerate:
    async def test_returns_message_content(self):
        provider, mock_client, _ = _make_provider()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="回應內容"))]
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        result = await provider.generate("問題")

        assert result == "回應內容"

    async def test_none_content_returns_empty_string(self):
        provider, mock_client, _ = _make_provider()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        result = await provider.generate("問題")

        assert result == ""

    async def test_passes_model_and_messages(self):
        provider, mock_client, _ = _make_provider()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="x"))]
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        await provider.generate("我的問題")

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["messages"] == [{"role": "user", "content": "我的問題"}]


class TestStream:
    async def test_yields_delta_content_tokens(self):
        provider, mock_client, _ = _make_provider()

        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="a"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="b"))]),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        mock_client.chat.completions.create = AsyncMock(return_value=_fake_stream())

        tokens = [t async for t in provider.stream("問題")]

        assert tokens == ["a", "b"]

    async def test_passes_stream_true(self):
        provider, mock_client, _ = _make_provider()

        async def _fake_stream():
            return
            yield  # pragma: no cover

        mock_client.chat.completions.create = AsyncMock(return_value=_fake_stream())

        async for _ in provider.stream("問題"):
            pass

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["stream"] is True
