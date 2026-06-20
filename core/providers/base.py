from __future__ import annotations
from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMProvider(ABC):
    """LLM 統一介面，所有 provider 必須實作 generate 與 stream。"""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """一次性生成回應，回傳完整字串。"""

    @abstractmethod
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """串流生成，逐 token yield 字串。"""

    async def generate_json(self, prompt: str) -> str:
        """強制 JSON 輸出模式，預設 fallback 至 generate（子類別可覆寫）。"""
        return await self.generate(prompt)


class EmbeddingProvider(ABC):
    """Embedding 統一介面，所有 provider 必須實作 encode 與 dim。"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量維度，建立 Neo4j vector index 時使用。"""

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """將單一文字編碼為向量。"""

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批次編碼，預設逐一呼叫 encode；provider 可覆寫以提升效率。"""
        return [self.encode(t) for t in texts]
