from __future__ import annotations
import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.providers.llm.grok import GrokLLMProvider, _GROK_BASE_URL


def _make_provider():
    """
    GrokLLMProvider 相容 OpenAI SDK，內部用 `from openai import AsyncOpenAI`。
    `openai` 是選用套件（不在 requirements.txt，CI 環境未安裝），因此用假模組
    注入 sys.modules，測試不依賴該套件是否實際安裝。
    """
    mock_cls = MagicMock()
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    fake_module = types.ModuleType("openai")
    fake_module.AsyncOpenAI = mock_cls

    with patch.dict(sys.modules, {"openai": fake_module}):
        provider = GrokLLMProvider(api_key="grok-key", model="grok-x")
    return provider, mock_client, mock_cls


class TestInit:
    def test_uses_grok_base_url(self):
        provider, mock_client, mock_cls = _make_provider()
        mock_cls.assert_called_once_with(api_key="grok-key", base_url=_GROK_BASE_URL)
        assert provider.model == "grok-x"


class TestGenerate:
    async def test_returns_message_content(self):
        provider, mock_client, _ = _make_provider()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="grok 回應"))]
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        result = await provider.generate("問題")

        assert result == "grok 回應"

    async def test_none_content_returns_empty_string(self):
        provider, mock_client, _ = _make_provider()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        result = await provider.generate("問題")

        assert result == ""


class TestStream:
    async def test_yields_delta_content_tokens(self):
        provider, mock_client, _ = _make_provider()

        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="x"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="y"))]),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        mock_client.chat.completions.create = AsyncMock(return_value=_fake_stream())

        tokens = [t async for t in provider.stream("問題")]

        assert tokens == ["x", "y"]
