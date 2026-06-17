from __future__ import annotations
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def init_embedding_service(model_name: str) -> "EmbeddingService":
    global _model
    _model = SentenceTransformer(model_name)
    svc = EmbeddingService(_model)
    logger.info(f"Embedding 模型載入完成：{model_name}，維度={svc.dim}")
    return svc


def get_embedding_service() -> "EmbeddingService":
    if _model is None:
        raise RuntimeError("Embedding 服務未初始化")
    return EmbeddingService(_model)


class EmbeddingService:
    def __init__(self, model: SentenceTransformer):
        self._model = model
        self.dim = model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()
