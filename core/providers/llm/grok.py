from __future__ import annotations
import logging
from typing import AsyncIterator

from core.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_GROK_BASE_URL = "https://api.x.ai/v1"


class GrokLLMProvider(LLMProvider):
    """xAI Grok — 相容 OpenAI SDK，只需換 base_url 與 api_key。"""

    def __init__(self, api_key: str, model: str):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key, base_url=_GROK_BASE_URL)
        self.model = model

    async def generate(self, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            if token := chunk.choices[0].delta.content:
                yield token
