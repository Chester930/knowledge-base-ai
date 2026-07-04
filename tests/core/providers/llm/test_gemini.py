from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch

from core.providers.llm.gemini import GeminiLLMProvider


def _make_provider():
    with patch("google.generativeai.configure") as mock_configure, \
         patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        provider = GeminiLLMProvider(api_key="g-key", model="gemini-x")
    return provider, mock_model, mock_configure, mock_model_cls


class TestInit:
    def test_configures_api_key_and_builds_model(self):
        provider, mock_model, mock_configure, mock_model_cls = _make_provider()
        mock_configure.assert_called_once_with(api_key="g-key")
        mock_model_cls.assert_called_once_with("gemini-x")


class TestGenerate:
    async def test_returns_response_text(self):
        provider, mock_model, _, _ = _make_provider()
        response = MagicMock(text="生成的內容")
        mock_model.generate_content.return_value = response

        result = await provider.generate("問題")

        assert result == "生成的內容"
        mock_model.generate_content.assert_called_once_with("問題")


class TestStream:
    async def test_yields_chunk_text_when_present(self):
        provider, mock_model, _, _ = _make_provider()
        chunks = [MagicMock(text="a"), MagicMock(text=""), MagicMock(text="b")]
        mock_model.generate_content.return_value = chunks

        tokens = [t async for t in provider.stream("問題")]

        assert tokens == ["a", "b"]

    async def test_passes_stream_true(self):
        provider, mock_model, _, _ = _make_provider()
        mock_model.generate_content.return_value = []

        async for _ in provider.stream("問題"):
            pass

        _, kwargs = mock_model.generate_content.call_args
        assert kwargs["stream"] is True
