from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.providers.llm.ollama import OllamaLLMProvider


def _make_provider(base_url="http://localhost:11434/"):
    return OllamaLLMProvider(base_url=base_url, model="qwen2.5:7b")


class _FakeAsyncClientCtx:
    """模擬 `async with httpx.AsyncClient(...) as client:` 的 context manager。"""
    def __init__(self, client):
        self._client = client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, *args):
        return False


class TestInit:
    def test_strips_trailing_slash_from_base_url(self):
        provider = _make_provider("http://localhost:11434/")
        assert provider.base_url == "http://localhost:11434"


class TestGenerate:
    async def test_returns_response_field(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "生成內容"}
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("core.providers.llm.ollama.httpx.AsyncClient",
                   return_value=_FakeAsyncClientCtx(mock_client)):
            result = await provider.generate("問題")

        assert result == "生成內容"

    async def test_posts_to_generate_endpoint_with_stream_false(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "x"}
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("core.providers.llm.ollama.httpx.AsyncClient",
                   return_value=_FakeAsyncClientCtx(mock_client)):
            await provider.generate("問題")

        args, kwargs = mock_client.post.call_args
        assert args[0] == "http://localhost:11434/api/generate"
        assert kwargs["json"]["stream"] is False
        assert kwargs["json"]["prompt"] == "問題"

    async def test_missing_response_field_returns_empty_string(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("core.providers.llm.ollama.httpx.AsyncClient",
                   return_value=_FakeAsyncClientCtx(mock_client)):
            result = await provider.generate("問題")

        assert result == ""


class TestGenerateJson:
    async def test_uses_json_format_mode(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": '{"a": 1}'}
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("core.providers.llm.ollama.httpx.AsyncClient",
                   return_value=_FakeAsyncClientCtx(mock_client)):
            result = await provider.generate_json("問題")

        assert result == '{"a": 1}'
        _, kwargs = mock_client.post.call_args
        assert kwargs["json"]["format"] == "json"


class TestStream:
    async def test_yields_tokens_until_done(self):
        provider = _make_provider()
        lines = [
            json.dumps({"response": "a"}),
            json.dumps({"response": "b"}),
            json.dumps({"done": True}),
        ]

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = MagicMock()
        mock_stream_resp.aiter_lines = _aiter_lines

        class _FakeStreamCtx:
            async def __aenter__(self_inner):
                return mock_stream_resp
            async def __aexit__(self_inner, *args):
                return False

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_FakeStreamCtx())

        with patch("core.providers.llm.ollama.httpx.AsyncClient",
                   return_value=_FakeAsyncClientCtx(mock_client)):
            tokens = [t async for t in provider.stream("問題")]

        assert tokens == ["a", "b"]

    async def test_skips_empty_lines_and_invalid_json(self):
        provider = _make_provider()
        lines = ["", "not valid json{{", json.dumps({"response": "x"}), json.dumps({"done": True})]

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = MagicMock()
        mock_stream_resp.aiter_lines = _aiter_lines

        class _FakeStreamCtx:
            async def __aenter__(self_inner):
                return mock_stream_resp
            async def __aexit__(self_inner, *args):
                return False

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=_FakeStreamCtx())

        with patch("core.providers.llm.ollama.httpx.AsyncClient",
                   return_value=_FakeAsyncClientCtx(mock_client)):
            tokens = [t async for t in provider.stream("問題")]

        assert tokens == ["x"]
