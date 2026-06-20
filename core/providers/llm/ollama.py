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

    async def generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            res.raise_for_status()
            return res.json().get("response", "")

    async def generate_json(self, prompt: str) -> str:
        """使用 Ollama format=json 模式，強制輸出合法 JSON。"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "format": "json"},
            )
            res.raise_for_status()
            return res.json().get("response", "")

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True},
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
