from __future__ import annotations
import logging

from core.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# text-embedding-3-small=1536, text-embedding-3-large=3072, ada-002=1536
_DIM_MAP: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self._dim = _DIM_MAP.get(model, 1536)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
