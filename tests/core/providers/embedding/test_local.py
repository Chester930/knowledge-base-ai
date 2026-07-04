from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch

from core.providers.embedding.local import LocalEmbeddingProvider


def _make_provider(dim=384):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dim
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_cls:
        provider = LocalEmbeddingProvider("some/model-name")
    return provider, mock_model, mock_cls


class TestInit:
    def test_loads_model_and_caches_dim(self):
        provider, mock_model, mock_cls = _make_provider(dim=768)
        mock_cls.assert_called_once_with("some/model-name")
        assert provider.dim == 768


class TestEncode:
    def test_encodes_single_text_normalized(self):
        provider, mock_model, _ = _make_provider()
        fake_vector = MagicMock()
        fake_vector.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = fake_vector

        result = provider.encode("測試文字")

        assert result == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with("測試文字", normalize_embeddings=True)


class TestEncodeBatch:
    def test_encodes_multiple_texts_normalized(self):
        provider, mock_model, _ = _make_provider()
        fake_vectors = MagicMock()
        fake_vectors.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_model.encode.return_value = fake_vectors

        result = provider.encode_batch(["a", "b"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_model.encode.assert_called_once_with(["a", "b"], normalize_embeddings=True)
