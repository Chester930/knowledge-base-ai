from __future__ import annotations
import logging

import httpx

from core.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """使用 Ollama 本地模型（nomic-embed-text 等）產生 Embedding。"""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = self._probe_dim()

    def _probe_dim(self) -> int:
        try:
            res = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": "dim probe"},
                timeout=30.0,
            )
            res.raise_for_status()
            return len(res.json()["embedding"])
        except Exception as e:
            logger.warning(f"OllamaEmbedding 維度探測失敗，預設 768：{e}")
            return 768

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> list[float]:
        res = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60.0,
        )
        res.raise_for_status()
        return res.json()["embedding"]
