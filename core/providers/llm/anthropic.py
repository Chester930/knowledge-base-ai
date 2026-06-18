from __future__ import annotations
import logging
from typing import AsyncIterator

from core.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_MAX_TOKENS = 4096


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str) -> str:
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            async for text in s.text_stream:
                yield text
