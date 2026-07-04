from __future__ import annotations
import pytest

from core.providers.base import LLMProvider, EmbeddingProvider


class _MinimalLLM(LLMProvider):
    async def generate(self, prompt: str) -> str:
        return f"generated:{prompt}"

    async def stream(self, prompt: str):
        for ch in prompt:
            yield ch


class _MinimalEmbedding(EmbeddingProvider):
    @property
    def dim(self) -> int:
        return 3

    def encode(self, text: str) -> list[float]:
        return [float(len(text))] * 3


class TestLLMProviderDefaults:
    async def test_generate_json_falls_back_to_generate(self):
        provider = _MinimalLLM()
        result = await provider.generate_json("hello")
        assert result == "generated:hello"

    def test_cannot_instantiate_without_implementing_abstract_methods(self):
        with pytest.raises(TypeError):
            LLMProvider()


class TestEmbeddingProviderDefaults:
    def test_encode_batch_calls_encode_for_each_text(self):
        provider = _MinimalEmbedding()
        result = provider.encode_batch(["a", "bb", "ccc"])
        assert result == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]

    def test_encode_batch_empty_list_returns_empty(self):
        provider = _MinimalEmbedding()
        assert provider.encode_batch([]) == []

    def test_cannot_instantiate_without_implementing_abstract_methods(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()
