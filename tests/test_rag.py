"""Tests for RAG pipeline components."""
import tempfile
import shutil
from pathlib import Path
from src.rag.embedder import Embedder


def test_embedder_initialization():
    embedder = Embedder(provider="huggingface", model="all-MiniLM-L6-v2")
    assert embedder.provider == "huggingface"
    assert embedder.dimension > 0


def test_embedder_embed_query():
    embedder = Embedder(provider="huggingface", model="all-MiniLM-L6-v2")
    vec = embedder.embed_query("hello world")
    assert len(vec) == embedder.dimension
    assert all(isinstance(v, float) for v in vec)


def test_embedder_batch():
    embedder = Embedder(provider="huggingface", model="all-MiniLM-L6-v2")
    texts = ["hello", "world", "code knowledge"]
    vecs = embedder.embed(texts)
    assert len(vecs) == 3
    assert all(len(v) == embedder.dimension for v in vecs)


def test_embedder_empty():
    embedder = Embedder(provider="huggingface", model="all-MiniLM-L6-v2")
    result = embedder.embed([])
    assert result == []


def test_tokenizer():
    from src.rag.retriever import Retriever
    tokens = Retriever._tokenize("def hello_world(): return x == y")
    assert "def" in tokens
    assert "hello_world" in tokens
    assert "return" in tokens


def test_tokenizer_code():
    from src.rag.retriever import Retriever
    tokens = Retriever._tokenize("class MyClass(BaseClass):\n    def method(self):")
    assert "class" in tokens
    assert "myclass" in tokens
    assert "baseclass" in tokens
    assert "method" in tokens


def test_embedder_different_models():
    small = Embedder(provider="huggingface", model="all-MiniLM-L6-v2")
    assert small.dimension == 384
