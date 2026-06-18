"""
向下相容的 embedding 服務包裝層。
新程式碼請直接使用 core.providers.factory.get_embedding_provider()。
"""
from __future__ import annotations
from core.providers.factory import get_embedding_provider
from core.providers.base import EmbeddingProvider


def init_embedding_service(model_name: str) -> "EmbeddingService":
    """保留舊介面供 main.py lifespan 呼叫；實際初始化已移至 init_providers()。"""
    return EmbeddingService(get_embedding_provider())


def get_embedding_service() -> "EmbeddingService":
    return EmbeddingService(get_embedding_provider())


class EmbeddingService:
    """薄包裝，保持舊呼叫介面（.encode / .encode_batch / .dim）不變。"""

    def __init__(self, provider: EmbeddingProvider):
        self._provider = provider

    @property
    def dim(self) -> int:
        return self._provider.dim

    def encode(self, text: str) -> list[float]:
        return self._provider.encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return self._provider.encode_batch(texts)
