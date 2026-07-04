"""Application configuration via environment variables."""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_ENV_FILE = str(_ENV_PATH) if _ENV_PATH.exists() else ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_model_path: str = ""  # Local model path for offline loading
    vector_store_path: str = "./chroma_data"
    repo_cache_path: str = "./repo_cache"
    log_level: str = "INFO"

    @property
    def llm_config(self) -> dict:
        match self.llm_provider:
            case "openai": return {"api_key": self.openai_api_key, "base_url": self.openai_base_url, "model": self.openai_model}
            case "anthropic": return {"api_key": self.anthropic_api_key, "model": self.anthropic_model}
            case "deepseek": return {"api_key": self.deepseek_api_key, "base_url": self.deepseek_base_url, "model": self.deepseek_model}
            case _: raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    @property
    def embedding_dim(self) -> int:
        dims = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "BAAI/bge-small-zh-v1.5": 512, "BAAI/bge-large-zh-v1.5": 1024}
        return dims.get(self.embedding_model, 768)

_settings: "Settings | None" = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
