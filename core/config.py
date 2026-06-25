from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Neo4j ──────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "knowledge123"

    # ── Provider 選擇 ──────────────────────────────────────────────────────────
    llm_provider: str = "ollama"        # ollama | openai | anthropic | gemini | grok
    embedding_provider: str = "local"   # local | openai | ollama

    # ── Ollama（本地）─────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "qwen2.5:7b"
    ollama_embedding_model: str = "nomic-embed-text"

    # ── 本地 Embedding（sentence-transformers）────────────────────────────────
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Anthropic ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # ── Google Gemini ──────────────────────────────────────────────────────────
    google_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # ── xAI Grok ───────────────────────────────────────────────────────────────
    grok_api_key: str = ""
    grok_model: str = "grok-2"

    # ── Whisper 語音/影片轉譯 ──────────────────────────────────────────────────
    whisper_model_size: str = "base"   # tiny | base | small | medium | large-v3

    # ── 系統行為 ───────────────────────────────────────────────────────────────
    concept_extraction_max: int = 8
    score_threshold: float = 0.70
    docs_dir: str = "./docs"
    workspace_dir: str = "./workspace"
    chunk_store_dir: str = "./chunk_store"

    # ── 聯邦識別 ───────────────────────────────────────────────────────────────
    instance_id: str = "local"          # 用於 KB Skill 描述檔的 instance 命名空間
    registry_path: str = "./registry.json"   # 本機 registry 路徑
    github_registry_url: str = ""       # GitHub Raw URL（Phase 2b 遠端分片發現）

    # 向下相容舊 .env（mapping 舊欄位名稱）
    @property
    def llm_model(self) -> str:
        return self.ollama_llm_model

    @property
    def embedding_model(self) -> str:
        return self.local_embedding_model

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
