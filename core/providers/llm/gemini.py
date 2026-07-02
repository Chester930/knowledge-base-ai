from __future__ import annotations
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
        response = await self._model.generate_content_async(prompt)
        return response.text

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        # generate_content_async(stream=True) 回傳真正的 async iterator，
        # 逐 chunk 的網路 I/O 不會阻塞事件迴圈（先前版本用 asyncio.to_thread
        # 只包住了取得 response 物件那一步，實際 for-loop 迭代仍是同步阻塞）。
        response = await self._model.generate_content_async(prompt, stream=True)
        async for chunk in response:
            if chunk.text:
                yield chunk.text
