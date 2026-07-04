from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch

from core.providers.embedding.openai import OpenAIEmbeddingProvider


def _make_provider(model="text-embedding-3-small"):
    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        provider = OpenAIEmbeddingProvider(api_key="sk-test", model=model)
    return provider, mock_client, mock_cls


class TestDim:
    def test_known_model_small(self):
        provider, _, _ = _make_provider("text-embedding-3-small")
        assert provider.dim == 1536

    def test_known_model_large(self):
        provider, _, _ = _make_provider("text-embedding-3-large")
        assert provider.dim == 3072

    def test_ada_002(self):
        provider, _, _ = _make_provider("text-embedding-ada-002")
        assert provider.dim == 1536

    def test_unknown_model_defaults_to_1536(self):
        provider, _, _ = _make_provider("some-future-model")
        assert provider.dim == 1536


class TestEncode:
    def test_returns_first_embedding(self):
        provider, mock_client, _ = _make_provider()
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1, 0.2])]
        mock_client.embeddings.create.return_value = response

        result = provider.encode("測試文字")

        assert result == [0.1, 0.2]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="測試文字"
        )


class TestEncodeBatch:
    def test_returns_all_embeddings_in_order(self):
        provider, mock_client, _ = _make_provider()
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1]), MagicMock(embedding=[0.2])]
        mock_client.embeddings.create.return_value = response

        result = provider.encode_batch(["a", "b"])

        assert result == [[0.1], [0.2]]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=["a", "b"]
        )
