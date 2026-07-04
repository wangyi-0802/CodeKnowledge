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
                import os
                from pathlib import Path
                from sentence_transformers import SentenceTransformer

                settings = get_settings()

                # 1) Explicit local model path (zero network, deployment-ready)
                model_path = settings.embedding_model_path
                if model_path:
                    path = Path(model_path)
                    if not path.is_absolute():
                        # Resolve relative to project root
                        path = (Path(__file__).resolve().parent.parent.parent / path).resolve()
                    if path.exists():
                        os.environ.setdefault("HF_HUB_OFFLINE", "1")
                        self._client = SentenceTransformer(str(path))
                        logger.info("Embedder loaded from local path: %s", path)
                        return

                # 2) ModelScope cache
                ms_cache = Path.home() / ".cache" / "modelscope" / self.model
                if ms_cache.exists():
                    try:
                        os.environ.setdefault("HF_HUB_OFFLINE", "1")
                        self._client = SentenceTransformer(str(ms_cache))
                        logger.info("Embedder loaded from ModelScope cache: %s", ms_cache)
                        return
                    except Exception:
                        pass

                # 3) HuggingFace cache (try offline first)
                hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
                if hf_cache.exists():
                    try:
                        os.environ.setdefault("HF_HUB_OFFLINE", "1")
                        self._client = SentenceTransformer(self.model)
                        logger.info("Embedder loaded from HuggingFace cache: %s", self.model)
                        return
                    except Exception:
                        pass

                # 4) Try online download via ModelScope then HuggingFace
                os.environ.pop("HF_HUB_OFFLINE", None)
                errors = []
                try:
                    from modelscope import snapshot_download
                    model_dir = snapshot_download(self.model)
                    self._client = SentenceTransformer(model_dir)
                    logger.info("Embedder ready via ModelScope: %s", self.model)
                    return
                except Exception as e:
                    errors.append(f"ModelScope: {e}")

                try:
                    self._client = SentenceTransformer(self.model)
                    logger.info("Embedder ready via HuggingFace: %s", self.model)
                    return
                except Exception as e:
                    errors.append(f"HuggingFace: {e}")

                raise RuntimeError(
                    f"Failed to load model '{self.model}'. "
                    f"Set EMBEDDING_MODEL_PATH in .env to a local model directory.\n"
                    + "\n".join(f"  [{i+1}] {err}" for i, err in enumerate(errors))
                )
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
