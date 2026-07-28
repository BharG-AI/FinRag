import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("FINRAG_INDEX_DIR", "data/index")))
    raw_dir: Path = field(default_factory=lambda: Path(os.getenv("FINRAG_RAW_DIR", "data/raw")))

    # Any OpenAI-compatible endpoint works. Point base_url at Ollama
    # (http://localhost:11434/v1) to run fully local.
    llm_base_url: str | None = field(default_factory=lambda: os.getenv("FINRAG_LLM_BASE_URL"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("FINRAG_LLM_MODEL", "gpt-4o-mini"))
    embed_model: str = field(
        default_factory=lambda: os.getenv("FINRAG_EMBED_MODEL", "text-embedding-3-small")
    )

    # SEC asks for a descriptive User-Agent with contact info on all requests.
    sec_user_agent: str = field(
        default_factory=lambda: os.getenv("FINRAG_SEC_USER_AGENT", "finrag dev@example.com")
    )

    chunk_size: int = int(os.getenv("FINRAG_CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("FINRAG_CHUNK_OVERLAP", "150"))


def get_settings() -> Settings:
    return Settings()
