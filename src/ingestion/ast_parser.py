"""AST-aware code parser using tree-sitter for semantic understanding."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CodeSymbol:
    """Represents a parsed code symbol (function, class, etc.)."""
    name: str
    kind: str  # "function", "class", "method", "variable"
    file_path: str
    start_line: int
    end_line: int
    docstring: str = ""
    source: str = ""
    parent: str | None = None  # parent class name for methods
    imports: list[str] = field(default_factory=list)


class ASTParser:
    """Multi-language AST parser using tree-sitter.

    Extracts functions, classes, imports with precise line ranges
    for semantic chunking rather than naive character splitting.
    """

    def __init__(self):
        self._parsers: dict[str, Any] = {}
        self._languages: dict[str, Any] = {}
        self._init_languages()

    def _init_languages(self) -> None:
        """Initialize tree-sitter language parsers for all supported languages."""
        import tree_sitter
        
        languages = [
            ("Python", "tree_sitter_python", ".py"),
            ("JavaScript", "tree_sitter_javascript", ".js", ".jsx"),
            ("TypeScript", "tree_sitter_typescript", ".ts", ".tsx"),
            ("Rust", "tree_sitter_rust", ".rs"),
            ("Go", "tree_sitter_go", ".go"),
            ("Java", "tree_sitter_java", ".java"),
        ]
        
        for lang_name, module_name, *extensions in languages:
            try:
                lang_module = __import__(module_name, fromlist=["language"])
                lang = lang_module.language()
                for ext in extensions:
                    self._languages[ext] = lang
                    self._parsers[ext] = tree_sitter.Parser(lang)
                logger.info("tree-sitter %s parser initialized (%s)", lang_name, ", ".join(extensions))
            except ImportError:
                logger.debug("tree-sitter-%s not installed", lang_name.lower())
            except Exception as e:
                logger.warning("Failed to init %s parser: %s", lang_name, e)

    def supports_language(self, file_path: str) -> bool:
        """Check if the file extension is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in self._parsers

    def parse_file(self, file_path: str) -> list[CodeSymbol]:
        """Parse a single file and extract all code symbols."""
        ext = Path(file_path).suffix.lower()
        if ext not in self._parsers:
            return []

        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
        tree = self._parsers[ext].parse(bytes(content, "utf-8"))

        if ext == ".py":
            return self._parse_python(tree, content, str(file_path))
        elif ext in (".js", ".jsx"):
            return self._parse_javascript(tree, content, str(file_path))
        elif ext in (".ts", ".tsx"):
            return self._parse_typescript(tree, content, str(file_path))
        elif ext == ".rs":
            return self._parse_rust(tree, content, str(file_path))
        elif ext == ".go":
            return self._parse_go(tree, content, str(file_path))
        elif ext == ".java":
            return self._parse_java(tree, content, str(file_path))
        return []

    def _parse_python(
        self, tree: Any, content: str, file_path: str
    ) -> list[CodeSymbol]:
        """Extract symbols from Python AST."""
        symbols: list[CodeSymbol] = []
        lines = content.split("\n")
        root = tree.root_node

        # Collect module-level imports
        imports: list[str] = []

        for child in root.children:
            if child.type == "import_statement":
                import_text = content[child.start_byte:child.end_byte]
                imports.append(import_text)
            elif child.type == "import_from_statement":
                import_text = content[child.start_byte:child.end_byte]
                imports.append(import_text)
            elif child.type in ("class_definition", "function_definition"):
                sym = self._extract_python_def(child, content, lines, file_path)
                if sym:
                    sym.imports = imports.copy()
                    symbols.append(sym)

        return symbols

    def _extract_python_def(
        self, node: Any, content: str, lines: list[str], file_path: str
    ) -> CodeSymbol | None:
        """Extract a Python function or class definition."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        name = content[name_node.start_byte:name_node.end_byte]
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        source = "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])

        kind = "class" if node.type == "class_definition" else "function"

        # Extract docstring
        docstring = ""
        body = node.child_by_field_name("body")
        if body and body.children:
            first_child = body.children[0]
            if first_child.type == "expression_statement":
                expr = first_child.child_by_field_name("expression")
                if expr and expr.type == "string":
                    docstring = content[expr.start_byte:expr.end_byte].strip('"\'')

        return CodeSymbol(
            name=name,
            kind=kind,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            source=source,
        )

    def _parse_javascript(
        self, tree: Any, content: str, file_path: str
    ) -> list[CodeSymbol]:
        """Extract symbols from JavaScript/JSX AST."""
        symbols: list[CodeSymbol] = []
        lines = content.split("\n")
        root = tree.root_node

        def _walk(node: Any, parent_class: str | None = None) -> None:
            if node.type in ("function_declaration", "method_definition", "arrow_function"):
                name_node = node.child_by_field_name("name")
                name = content[name_node.start_byte:name_node.end_byte] if name_node else "anonymous"
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                source = "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])
                symbols.append(CodeSymbol(
                    name=name, kind="method" if parent_class else "function",
                    file_path=file_path, start_line=start_line, end_line=end_line,
                    source=source, parent=parent_class,
                ))
            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                name = content[name_node.start_byte:name_node.end_byte] if name_node else "anonymous"
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                source = "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])
                symbols.append(CodeSymbol(
                    name=name, kind="class",
                    file_path=file_path, start_line=start_line, end_line=end_line,
                    source=source,
                ))
                for child in node.children:
                    _walk(child, parent_class=name)
            else:
                for child in node.children:
                    _walk(child, parent_class)

        _walk(root)
        return symbols

    def _parse_rust(
        self, tree: Any, content: str, file_path: str
    ) -> list[CodeSymbol]:
        """Extract symbols from Rust AST."""
        symbols: list[CodeSymbol] = []
        lines = content.split("\n")
        root = tree.root_node

        def _walk(node: Any, parent: str | None = None) -> None:
            if node.type == "function_item":
                name_node = node.child_by_field_name("name")
                if not name_node:
                    return
                name = content[name_node.start_byte:name_node.end_byte]
                sl = node.start_point[0] + 1
                el = node.end_point[0] + 1
                src = "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])
                symbols.append(CodeSymbol(name=name, kind="function", file_path=file_path, start_line=sl, end_line=el, source=src, parent=parent))
            elif node.type == "struct_item":
                name_node = node.child_by_field_name("name")
                if not name_node:
                    return
                name = content[name_node.start_byte:name_node.end_byte]
                sl = node.start_point[0] + 1
                el = node.end_point[0] + 1
                src = "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])
                symbols.append(CodeSymbol(name=name, kind="struct", file_path=file_path, start_line=sl, end_line=el, source=src))
                for child in node.children:
                    _walk(child, parent=name)
            elif node.type == "impl_item":
                for child in node.children:
                    _walk(child, parent)
            else:
                for child in node.children:
                    _walk(child, parent)
        _walk(root)
        return symbols

    def _parse_go(
        self, tree: Any, content: str, file_path: str
    ) -> list[CodeSymbol]:
        """Extract symbols from Go AST."""
        symbols: list[CodeSymbol] = []
        lines = content.split("\n")
        root = tree.root_node

        def _walk(node: Any, parent: str | None = None) -> None:
            if node.type in ("function_declaration", "method_declaration"):
                name_node = node.child_by_field_name("name")
                if not name_node:
                    return
                name = content[name_node.start_byte:name_node.end_byte]
                sl = node.start_point[0] + 1
                el = node.end_point[0] + 1
                src = "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])
                kind = "method" if node.type == "method_declaration" else "function"
                symbols.append(CodeSymbol(name=name, kind=kind, file_path=file_path, start_line=sl, end_line=el, source=src, parent=parent))
            elif node.type == "type_declaration":
                for child in node.children:
                    _walk(child, parent)
            elif node.type == "type_spec":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = content[name_node.start_byte:name_node.end_byte]
                    sl = node.start_point[0] + 1
                    el = node.end_point[0] + 1
                    src = "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])
                    symbols.append(CodeSymbol(name=name, kind="type", file_path=file_path, start_line=sl, end_line=el, source=src))
            else:
                for child in node.children:
                    _walk(child, parent)
        _walk(root)
        return symbols

    def _parse_java(
        self, tree: Any, content: str, file_path: str
    ) -> list[CodeSymbol]:
        """Extract symbols from Java AST."""
        symbols: list[CodeSymbol] = []
        lines = content.split("\n")
        root = tree.root_node

        def _walk(node: Any, parent: str | None = None) -> None:
            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if not name_node:
                    return
                name = content[name_node.start_byte:name_node.end_byte]
                sl = node.start_point[0] + 1
                el = node.end_point[0] + 1
                src = "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])
                symbols.append(CodeSymbol(name=name, kind="class", file_path=file_path, start_line=sl, end_line=el, source=src))
                for child in node.children:
                    _walk(child, parent=name)
            elif node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                if not name_node:
                    return
                name = content[name_node.start_byte:name_node.end_byte]
                sl = node.start_point[0] + 1
                el = node.end_point[0] + 1
                src = "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])
                symbols.append(CodeSymbol(name=name, kind="method", file_path=file_path, start_line=sl, end_line=el, source=src, parent=parent))
            elif node.type == "interface_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = content[name_node.start_byte:name_node.end_byte]
                    sl = node.start_point[0] + 1
                    el = node.end_point[0] + 1
                    src = "\n".join(lines[node.start_point[0]:node.end_point[0] + 1])
                    symbols.append(CodeSymbol(name=name, kind="interface", file_path=file_path, start_line=sl, end_line=el, source=src))
            else:
                for child in node.children:
                    _walk(child, parent)
        _walk(root)
        return symbols

    def _parse_typescript(
        self, tree: Any, content: str, file_path: str
    ) -> list[CodeSymbol]:
        """Extract symbols from TypeScript/TSX AST."""
        return self._parse_javascript(tree, content, file_path)
