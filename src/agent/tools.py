"""Agent tools for code repository interaction via RAG."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from src.rag.retriever import Retriever
from src.utils.logger import get_logger
logger = get_logger(__name__)
class CodeKnowledgeTools:
    """Tools that the agent can call to interact with the codebase."""
    def __init__(self, retriever: Retriever, repo_path: str):
        self.retriever = retriever
        self.repo_path = Path(repo_path)
    def search_code(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Search the codebase for relevant code.
        Uses semantic search + optional BM25 for the best results.
        Args:
            query: Natural language description of what to find.
            top_k: Number of results to return.
        Returns:
            List of relevant code chunks with file paths and line numbers.
        """
        logger.info("Searching codebase: %s", query[:80])
        results = self.retriever.retrieve(query, top_k=top_k)
        return [
            {
                "file_path": r["metadata"].get("file_path", "unknown"),
                "symbol_name": r["metadata"].get("symbol_name", "unknown"),
                "symbol_kind": r["metadata"].get("symbol_kind", "code"),
                "start_line": r["metadata"].get("start_line", 0),
                "end_line": r["metadata"].get("end_line", 0),
                "content": r["content"][:800],
                "relevance_score": r.get("relevance_score", 0.0),
            }
            for r in results
        ]
    def read_file(self, file_path: str, start_line: int = 0, end_line: int = 0) -> str:
        """Read a specific file from the repository.
        Args:
            file_path: Relative path within the repo (e.g. src/main.py).
            start_line: Optional start line (1-indexed, 0 for beginning).
            end_line: Optional end line (0 for end of file).
        Returns:
            File content as a string with line numbers.
        """
        full_path = self.repo_path / file_path
        if not full_path.exists():
            # Try to find the file by name
            matches = list(self.repo_path.rglob(file_path))
            if not matches:
                return f"Error: File not found: {file_path}"
            full_path = matches[0]
        try:
            lines = full_path.read_text(encoding="utf-8", errors="replace").split("\n")
            if end_line > 0:
                lines = lines[start_line:end_line]
            elif start_line > 0:
                lines = lines[start_line - 1:]
            numbered = [f"{i + 1}: {line}" for i, line in enumerate(lines, start=start_line or 1)]
            return "\n".join(numbered)
        except Exception as e:
            logger.error("Failed to read file %s: %s", file_path, e)
            return f"Error reading file: {e}"
    def list_files(self, directory: str = "") -> str:
        """List files in a directory within the repository.
        Args:
            directory: Relative directory path (empty for repo root).
        Returns:
            Formatted list of files and subdirectories.
        """
        target = self.repo_path / directory if directory else self.repo_path
        if not target.exists() or not target.is_dir():
            return f"Error: Directory not found: {directory}"
        items = []
        for item in sorted(target.iterdir()):
            if item.name.startswith("."):
                continue
            icon = "/" if item.is_dir() else ""
            size = item.stat().st_size if item.is_file() else 0
            size_str = f" ({size:,} bytes)" if size else ""
            items.append(f"  {item.name}{icon}{size_str}")
        header = f"Contents of /{directory}\n" if directory else "Repository root\n"
        return header + "\n".join(items)
    def get_repo_structure(self) -> str:
        """Get a high-level overview of the repository structure."""
        return self.list_files("")
    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions for LLM function calling."""
        return [
            {
                "name": "search_code",
                "description": "Search the codebase for relevant code using semantic search. Use this to find functions, classes, or patterns related to the user's question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of what to find",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default 8)",
                            "default": 8,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "read_file",
                "description": "Read the content of a specific file from the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Relative path to the file (e.g. src/main.py)",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Start line (1-indexed, 0 for beginning)",
                            "default": 0,
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "End line (0 for end of file)",
                            "default": 0,
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "list_files",
                "description": "List files and directories in the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory path relative to repo root (empty for root)",
                            "default": "",
                        },
                    },
                },
            },
        ]
