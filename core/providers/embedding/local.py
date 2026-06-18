from __future__ import annotations
import logging

from core.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers 本地 Embedding，完全離線。"""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        logger.info(f"載入本地 Embedding 模型：{model_name}")
        self._model = SentenceTransformer(model_name)
        self._dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding 模型載入完成，維度={self._dim}")

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()
