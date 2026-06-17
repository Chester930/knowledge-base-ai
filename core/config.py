from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "knowledge123"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "llama3.2"
    concept_extraction_max: int = 8
    score_threshold: float = 0.70
    docs_dir: str = "./docs"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
