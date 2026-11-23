import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/jobmatch"
    )

    # Embeddings run locally (no API key needed). If you change the model,
    # also change the vector(384) size in sql/schema.sql.
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_dim: int = 384

    # Any OpenAI-compatible chat API works (OpenAI, Gemini, Groq, ...).
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "120"))

    # how many jobs vector search hands to the LLM for re-ranking
    candidates_for_rerank: int = int(os.getenv("CANDIDATES_FOR_RERANK", "20"))


settings = Settings()
