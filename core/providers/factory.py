from __future__ import annotations
import logging

from core.providers.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

_llm: LLMProvider | None = None
_embedding: EmbeddingProvider | None = None


def init_providers() -> EmbeddingProvider:
    """
    根據 settings 初始化 LLM 與 Embedding provider。
    在 app lifespan 啟動時呼叫一次；回傳 EmbeddingProvider 供建立 vector index 使用。
    """
    global _llm, _embedding
    from core.config import settings

    # ── Embedding ──────────────────────────────────────────────────────────────
    match settings.embedding_provider:
        case "local":
            from core.providers.embedding.local import LocalEmbeddingProvider
            _embedding = LocalEmbeddingProvider(settings.local_embedding_model)
        case "openai":
            from core.providers.embedding.openai import OpenAIEmbeddingProvider
            _embedding = OpenAIEmbeddingProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
            )
        case "ollama":
            from core.providers.embedding.ollama import OllamaEmbeddingProvider
            _embedding = OllamaEmbeddingProvider(
                base_url=settings.ollama_base_url,
                model=settings.ollama_embedding_model,
            )
        case _:
            raise ValueError(f"不支援的 embedding_provider：{settings.embedding_provider}")

    # ── LLM ───────────────────────────────────────────────────────────────────
    match settings.llm_provider:
        case "ollama":
            from core.providers.llm.ollama import OllamaLLMProvider
            _llm = OllamaLLMProvider(
                base_url=settings.ollama_base_url,
                model=settings.ollama_llm_model,
            )
        case "openai":
            from core.providers.llm.openai import OpenAILLMProvider
            _llm = OpenAILLMProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_llm_model,
            )
        case "anthropic":
            from core.providers.llm.anthropic import AnthropicLLMProvider
            _llm = AnthropicLLMProvider(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
            )
        case "gemini":
            from core.providers.llm.gemini import GeminiLLMProvider
            _llm = GeminiLLMProvider(
                api_key=settings.google_api_key,
                model=settings.gemini_model,
            )
        case "grok":
            from core.providers.llm.grok import GrokLLMProvider
            _llm = GrokLLMProvider(
                api_key=settings.grok_api_key,
                model=settings.grok_model,
            )
        case _:
            raise ValueError(f"不支援的 llm_provider：{settings.llm_provider}")

    logger.info(f"LLM Provider：{settings.llm_provider}")
    logger.info(f"Embedding Provider：{settings.embedding_provider}")
    return _embedding


def get_llm_provider() -> LLMProvider:
    if _llm is None:
        raise RuntimeError("Provider 尚未初始化，請先呼叫 init_providers()")
    return _llm


def get_embedding_provider() -> EmbeddingProvider:
    if _embedding is None:
        raise RuntimeError("Provider 尚未初始化，請先呼叫 init_providers()")
    return _embedding
