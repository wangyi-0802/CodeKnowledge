"""Tests for AST parser and chunk processor."""
from pathlib import Path
import tempfile
from src.ingestion.ast_parser import ASTParser
from src.ingestion.chunk_processor import ChunkProcessor


def create_temp_py(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


def test_parse_function():
    code = """
def greet(name: str) -> str:
    \"\"\"Say hello to someone.\"\"\"
    return f"Hello, {name}!"
"""
    parser = ASTParser()
    file_path = create_temp_py(code)
    symbols = parser.parse_file(file_path)
    assert len(symbols) == 1
    assert symbols[0].name == "greet"
    assert symbols[0].kind == "function"
    assert "Hello" in symbols[0].source
    Path(file_path).unlink()


def test_parse_class_with_method():
    code = """
class Calculator:
    \"\"\"A simple calculator.\"\"\"

    def add(self, a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b
"""
    parser = ASTParser()
    file_path = create_temp_py(code)
    symbols = parser.parse_file(file_path)
    names = [s.name for s in symbols]
    kinds = [s.kind for s in symbols]
    assert "Calculator" in names
    assert "add" in names
    assert "subtract" in names
    assert kinds.count("class") == 1
    assert kinds.count("function") == 2
    Path(file_path).unlink()


def test_chunk_processor():
    code = """
import os
import sys


def main():
    \"\"\"Entry point.\"\"\"
    print("hello")


class Helper:
    pass
"""
    parser = ASTParser()
    processor = ChunkProcessor(ast_parser=parser)
    file_path = create_temp_py(code)
    chunks = processor.process_file(file_path)
    assert len(chunks) == 2
    chunk_names = [c.symbol_name for c in chunks]
    assert "main" in chunk_names
    assert "Helper" in chunk_names
    Path(file_path).unlink()


def test_empty_file():
    parser = ASTParser()
    file_path = create_temp_py("")
    symbols = parser.parse_file(file_path)
    assert len(symbols) == 0
    Path(file_path).unlink()


def test_import_tracking():
    code = """
import json
from typing import Optional, List


def process(data: str) -> dict:
    return json.loads(data)
"""
    parser = ASTParser()
    file_path = create_temp_py(code)
    symbols = parser.parse_file(file_path)
    assert len(symbols) == 1
    assert any("import json" in imp for imp in symbols[0].imports)
    Path(file_path).unlink()
