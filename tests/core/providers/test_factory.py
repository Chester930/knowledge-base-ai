from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch

import core.providers.factory as factory


@pytest.fixture(autouse=True)
def _reset_singletons():
    """每個測試前後重置 factory 的模組級單例，避免測試互相汙染。"""
    factory._llm = None
    factory._embedding = None
    yield
    factory._llm = None
    factory._embedding = None


def _mock_settings(**overrides):
    settings = MagicMock()
    settings.embedding_provider = "local"
    settings.llm_provider = "ollama"
    settings.local_embedding_model = "model-x"
    settings.ollama_base_url = "http://localhost:11434"
    settings.ollama_llm_model = "qwen"
    settings.ollama_embedding_model = "nomic"
    settings.openai_api_key = "sk-x"
    settings.openai_llm_model = "gpt-4o-mini"
    settings.openai_embedding_model = "text-embedding-3-small"
    settings.anthropic_api_key = "sk-ant"
    settings.anthropic_model = "claude-x"
    settings.google_api_key = "g-key"
    settings.gemini_model = "gemini-x"
    settings.grok_api_key = "grok-key"
    settings.grok_model = "grok-x"
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


class TestInitProvidersEmbeddingSelection:
    def test_local_embedding_selected(self):
        settings = _mock_settings(embedding_provider="local")
        mock_provider = MagicMock()
        with patch("core.config.settings", settings), \
             patch("core.providers.embedding.local.LocalEmbeddingProvider", return_value=mock_provider) as mock_cls, \
             patch("core.providers.llm.ollama.OllamaLLMProvider"):
            result = factory.init_providers()
        mock_cls.assert_called_once_with("model-x")
        assert result is mock_provider

    def test_openai_embedding_selected(self):
        settings = _mock_settings(embedding_provider="openai")
        mock_provider = MagicMock()
        with patch("core.config.settings", settings), \
             patch("core.providers.embedding.openai.OpenAIEmbeddingProvider", return_value=mock_provider) as mock_cls, \
             patch("core.providers.llm.ollama.OllamaLLMProvider"):
            factory.init_providers()
        mock_cls.assert_called_once_with(api_key="sk-x", model="text-embedding-3-small")

    def test_ollama_embedding_selected(self):
        settings = _mock_settings(embedding_provider="ollama")
        with patch("core.config.settings", settings), \
             patch("core.providers.embedding.ollama.OllamaEmbeddingProvider") as mock_cls, \
             patch("core.providers.llm.ollama.OllamaLLMProvider"):
            factory.init_providers()
        mock_cls.assert_called_once_with(base_url="http://localhost:11434", model="nomic")

    def test_unsupported_embedding_provider_raises(self):
        settings = _mock_settings(embedding_provider="unknown")
        with patch("core.config.settings", settings):
            with pytest.raises(ValueError, match="不支援的 embedding_provider"):
                factory.init_providers()


class TestInitProvidersLlmSelection:
    def test_ollama_llm_selected(self):
        settings = _mock_settings(llm_provider="ollama")
        with patch("core.config.settings", settings), \
             patch("core.providers.embedding.local.LocalEmbeddingProvider"), \
             patch("core.providers.llm.ollama.OllamaLLMProvider") as mock_cls:
            factory.init_providers()
        mock_cls.assert_called_once_with(base_url="http://localhost:11434", model="qwen")

    def test_openai_llm_selected(self):
        settings = _mock_settings(llm_provider="openai")
        with patch("core.config.settings", settings), \
             patch("core.providers.embedding.local.LocalEmbeddingProvider"), \
             patch("core.providers.llm.openai.OpenAILLMProvider") as mock_cls:
            factory.init_providers()
        mock_cls.assert_called_once_with(api_key="sk-x", model="gpt-4o-mini")

    def test_anthropic_llm_selected(self):
        settings = _mock_settings(llm_provider="anthropic")
        with patch("core.config.settings", settings), \
             patch("core.providers.embedding.local.LocalEmbeddingProvider"), \
             patch("core.providers.llm.anthropic.AnthropicLLMProvider") as mock_cls:
            factory.init_providers()
        mock_cls.assert_called_once_with(api_key="sk-ant", model="claude-x")

    def test_gemini_llm_selected(self):
        settings = _mock_settings(llm_provider="gemini")
        with patch("core.config.settings", settings), \
             patch("core.providers.embedding.local.LocalEmbeddingProvider"), \
             patch("core.providers.llm.gemini.GeminiLLMProvider") as mock_cls:
            factory.init_providers()
        mock_cls.assert_called_once_with(api_key="g-key", model="gemini-x")

    def test_grok_llm_selected(self):
        settings = _mock_settings(llm_provider="grok")
        with patch("core.config.settings", settings), \
             patch("core.providers.embedding.local.LocalEmbeddingProvider"), \
             patch("core.providers.llm.grok.GrokLLMProvider") as mock_cls:
            factory.init_providers()
        mock_cls.assert_called_once_with(api_key="grok-key", model="grok-x")

    def test_unsupported_llm_provider_raises(self):
        settings = _mock_settings(llm_provider="unknown")
        with patch("core.config.settings", settings), \
             patch("core.providers.embedding.local.LocalEmbeddingProvider"):
            with pytest.raises(ValueError, match="不支援的 llm_provider"):
                factory.init_providers()


class TestGetProviderAccessors:
    def test_get_llm_provider_raises_before_init(self):
        with pytest.raises(RuntimeError, match="尚未初始化"):
            factory.get_llm_provider()

    def test_get_embedding_provider_raises_before_init(self):
        with pytest.raises(RuntimeError, match="尚未初始化"):
            factory.get_embedding_provider()

    def test_get_llm_provider_returns_initialized_instance(self):
        settings = _mock_settings(llm_provider="ollama", embedding_provider="local")
        with patch("core.config.settings", settings), \
             patch("core.providers.embedding.local.LocalEmbeddingProvider"), \
             patch("core.providers.llm.ollama.OllamaLLMProvider") as mock_cls:
            factory.init_providers()
        assert factory.get_llm_provider() is mock_cls.return_value

    def test_get_embedding_provider_returns_initialized_instance(self):
        settings = _mock_settings(llm_provider="ollama", embedding_provider="local")
        with patch("core.config.settings", settings), \
             patch("core.providers.embedding.local.LocalEmbeddingProvider") as mock_cls, \
             patch("core.providers.llm.ollama.OllamaLLMProvider"):
            factory.init_providers()
        assert factory.get_embedding_provider() is mock_cls.return_value
