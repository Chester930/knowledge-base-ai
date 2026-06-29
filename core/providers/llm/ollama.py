from __future__ import annotations
import json
import logging
from typing import AsyncIterator

import httpx

from core.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaLLMProvider(LLMProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    # RAG prompt 通常 8000-20000 字元，需要足夠的 context window
    _NUM_CTX = 8192

    async def generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=300.0) as client:
            res = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False,
                      "options": {"num_ctx": self._NUM_CTX, "temperature": 0.0, "num_predict": 1024}},
            )
            res.raise_for_status()
            return res.json().get("response", "")
 
    async def generate_json(self, prompt: str) -> str:
        """使用 Ollama format=json 模式，強制輸出合法 JSON。"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            res = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False,
                      "format": "json", "options": {"num_ctx": self._NUM_CTX, "temperature": 0.0, "num_predict": 1024}},
            )
            res.raise_for_status()
            return res.json().get("response", "")

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True,
                      "options": {"num_ctx": self._NUM_CTX}},
            ) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if token := data.get("response"):
                            yield token
                        if data.get("done"):
                            return
                    except json.JSONDecodeError:
                        continue
