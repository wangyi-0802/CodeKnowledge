"""Multi-provider embedding pipeline for code and text."""

from __future__ import annotations

from typing import Any

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """Produces embeddings for code chunks and queries.

    Supports OpenAI API embeddings and local HuggingFace models,
    allowing users to choose based on cost/quality needs.
    """

    def __init__(self, provider: str | None = None, model: str | None = None):
        settings = get_settings()
        self.provider = provider or settings.embedding_provider
        self.model = model or settings.embedding_model
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the embedding client."""
        match self.provider:
            case "openai":
                from openai import OpenAI
                settings = get_settings()
                self._client = OpenAI(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                )
                logger.info("OpenAI embedder ready: %s", self.model)
            case "huggingface":
                from sentence_transformers import SentenceTransformer
                self._client = SentenceTransformer(self.model)
                logger.info("HuggingFace embedder ready: %s", self.model)
            case _:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""
        if not texts:
            return []

        match self.provider:
            case "openai":
                response = self._client.embeddings.create(
                    input=texts,
                    model=self.model,
                )
                return [item.embedding for item in response.data]
            case "huggingface":
                embeddings = self._client.encode(texts, show_progress_bar=False)
                return embeddings.tolist()
            case _:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        settings = get_settings()
        return settings.embedding_dim

    def __repr__(self) -> str:
        return f"Embedder(provider={self.provider}, model={self.model})"
