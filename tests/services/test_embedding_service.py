from __future__ import annotations
from unittest.mock import MagicMock, patch

from services.embedding_service import (
    EmbeddingService,
    init_embedding_service,
    get_embedding_service,
)


def _mock_provider(dim=384):
    provider = MagicMock()
    provider.dim = dim
    provider.encode.return_value = [0.1] * dim
    provider.encode_batch.return_value = [[0.1] * dim, [0.2] * dim]
    return provider


class TestEmbeddingService:
    def test_dim_delegates_to_provider(self):
        provider = _mock_provider(dim=768)
        svc = EmbeddingService(provider)
        assert svc.dim == 768

    def test_encode_delegates_to_provider(self):
        provider = _mock_provider()
        svc = EmbeddingService(provider)
        result = svc.encode("測試文字")
        provider.encode.assert_called_once_with("測試文字")
        assert result == [0.1] * 384

    def test_encode_batch_delegates_to_provider(self):
        provider = _mock_provider()
        svc = EmbeddingService(provider)
        result = svc.encode_batch(["a", "b"])
        provider.encode_batch.assert_called_once_with(["a", "b"])
        assert len(result) == 2


class TestInitEmbeddingService:
    def test_returns_embedding_service_wrapping_factory_provider(self):
        provider = _mock_provider()
        with patch("services.embedding_service.get_embedding_provider", return_value=provider):
            svc = init_embedding_service("some-model")
        assert isinstance(svc, EmbeddingService)
        assert svc.dim == provider.dim


class TestGetEmbeddingService:
    def test_returns_embedding_service_wrapping_factory_provider(self):
        provider = _mock_provider()
        with patch("services.embedding_service.get_embedding_provider", return_value=provider):
            svc = get_embedding_service()
        assert isinstance(svc, EmbeddingService)
        assert svc.encode("x") == provider.encode.return_value
