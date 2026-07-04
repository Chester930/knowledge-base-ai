from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch

from core.providers.embedding.ollama import OllamaEmbeddingProvider


def _mock_probe_response(dim=768):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"embedding": [0.0] * dim}
    return resp


class TestInit:
    def test_strips_trailing_slash(self):
        with patch("core.providers.embedding.ollama.httpx.post", return_value=_mock_probe_response()):
            provider = OllamaEmbeddingProvider(base_url="http://localhost:11434/", model="nomic-embed-text")
        assert provider.base_url == "http://localhost:11434"

    def test_probes_dim_from_real_embedding_call(self):
        with patch("core.providers.embedding.ollama.httpx.post",
                   return_value=_mock_probe_response(dim=1024)) as mock_post:
            provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")
        assert provider.dim == 1024
        mock_post.assert_called_once()

    def test_probe_failure_defaults_to_768(self):
        with patch("core.providers.embedding.ollama.httpx.post", side_effect=RuntimeError("連線失敗")):
            provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")
        assert provider.dim == 768


class TestEncode:
    def test_returns_embedding_from_response(self):
        with patch("core.providers.embedding.ollama.httpx.post", return_value=_mock_probe_response()):
            provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        with patch("core.providers.embedding.ollama.httpx.post", return_value=resp) as mock_post:
            result = provider.encode("測試文字")

        assert result == [0.1, 0.2, 0.3]
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["prompt"] == "測試文字"
        assert kwargs["json"]["model"] == "nomic-embed-text"
