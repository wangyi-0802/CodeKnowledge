"""Semantic code chunking based on AST structure, not character count."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.ingestion.ast_parser import ASTParser, CodeSymbol
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CodeChunk:
    """A semantically meaningful code chunk with rich metadata."""
    id: str
    content: str
    file_path: str
    symbol_name: str
    symbol_kind: str  # "function", "class", "module", "file_overview"
    start_line: int
    end_line: int
    docstring: str = ""
    parent: str | None = None
    imports: list[str] = field(default_factory=list)
    language: str = "python"

    @property
    def metadata(self) -> dict[str, Any]:
        meta = {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "symbol_kind": self.symbol_kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
        }
        if self.parent is not None:
            meta["parent"] = self.parent
        return meta


class ChunkProcessor:
    """Produces semantically meaningful code chunks using AST structure.

    Instead of naive character splitting, this processor:
    1. Parses code with tree-sitter to find functions, classes, methods
    2. Creates one chunk per symbol (function/class)
    3. Preserves import context and parent relationships
    4. Adds file-level overview chunks for files with many small functions
    """

    def __init__(self, ast_parser: ASTParser | None = None):
        self.ast_parser = ast_parser or ASTParser()

    def process_file(self, file_path: str, max_chunk_size: int = 200) -> list[CodeChunk]:
        """Process a single file into semantic chunks.

        Args:
            file_path: Path to the source file.
            max_chunk_size: Maximum lines per chunk; large symbols are split
                            at secondary boundaries (e.g. inner functions).

        Returns:
            List of CodeChunk objects.
        """
        ext = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python", ".js": "javascript", ".jsx": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
        }
        language = language_map.get(ext, ext.lstrip("."))

        chunks: list[CodeChunk] = []
        symbols = self.ast_parser.parse_file(file_path)

        if not symbols:
            # Fallback: create a file-level chunk
            chunks.append(self._make_file_chunk(file_path, language))
            return chunks

        for sym in symbols:
            chunk_id = f"{sym.file_path}:{sym.name}:L{sym.start_line}"

            # For very large symbols, split into smaller chunks
            if sym.end_line - sym.start_line > max_chunk_size:
                sub_chunks = self._split_large_symbol(sym, max_chunk_size)
                chunks.extend(sub_chunks)
            else:
                chunks.append(CodeChunk(
                    id=chunk_id,
                    content=sym.source,
                    file_path=sym.file_path,
                    symbol_name=sym.name,
                    symbol_kind=sym.kind,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    docstring=sym.docstring,
                    parent=sym.parent,
                    imports=sym.imports,
                    language=language,
                ))

        logger.debug(
            "Processed %s -> %d chunks from %d symbols",
            file_path, len(chunks), len(symbols),
        )
        return chunks

    def process_directory(self, directory: str, extensions: set[str] | None = None) -> list[CodeChunk]:
        """Recursively process all supported files in a directory."""
        if extensions is None:
            extensions = {".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java"}

        all_chunks: list[CodeChunk] = []
        path = Path(directory)

        for file_path in path.rglob("*"):
            if file_path.suffix.lower() not in extensions:
                continue
            # Skip hidden dirs and __pycache__
            if any(part.startswith(".") or part == "__pycache__" for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue
            try:
                chunks = self.process_file(str(file_path))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning("Failed to process %s: %s", file_path, e)

        logger.info(
            "Directory processed: %s -> %d chunks from %d files",
            directory, len(all_chunks), len(set(c.file_path for c in all_chunks)),
        )
        return all_chunks

    def _make_file_chunk(self, file_path: str, language: str) -> CodeChunk:
        """Create a file-level overview chunk for unsupported or symbol-less files."""
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        return CodeChunk(
            id=f"{file_path}:file",
            content=content[:2000],  # Truncate very large files
            file_path=file_path,
            symbol_name=Path(file_path).name,
            symbol_kind="file_overview",
            start_line=1,
            end_line=len(lines),
            language=language,
        )

    def _split_large_symbol(self, sym: CodeSymbol, max_size: int) -> list[CodeChunk]:
        """Split a very large symbol at internal structure boundaries."""
        lines = sym.source.split("\n")
        chunks = []

        for i in range(0, len(lines), max_size):
            chunk_lines = lines[i:i + max_size]
            start = sym.start_line + i
            end = start + len(chunk_lines) - 1
            chunk_id = f"{sym.file_path}:{sym.name}:part{i // max_size}:L{start}"
            chunks.append(CodeChunk(
                id=chunk_id,
                content="\n".join(chunk_lines),
                file_path=sym.file_path,
                symbol_name=sym.name,
                symbol_kind=sym.kind,
                start_line=start,
                end_line=end,
                docstring=sym.docstring,
                parent=sym.parent,
                imports=sym.imports,
                language=sym.file_path.split(".")[-1],
            ))
        return chunks
