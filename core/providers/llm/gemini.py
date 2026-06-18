from __future__ import annotations
import asyncio
import logging
from typing import AsyncIterator

from core.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    async def generate(self, prompt: str) -> str:
        response = await asyncio.to_thread(self._model.generate_content, prompt)
        return response.text

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        def _sync_stream():
            return self._model.generate_content(prompt, stream=True)

        response = await asyncio.to_thread(_sync_stream)
        for chunk in response:
            if chunk.text:
                yield chunk.text
