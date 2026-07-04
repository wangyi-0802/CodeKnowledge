"""Orchestration pipeline: ingest repo -> build index -> serve agent."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from src.config.settings import get_settings
from src.ingestion.repo_cloner import RepoCloner
from src.ingestion.ast_parser import ASTParser
from src.ingestion.chunk_processor import ChunkProcessor, CodeChunk
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.agent.tools import CodeKnowledgeTools
from src.agent.code_agent import CodeKnowledgeAgent
from src.utils.logger import get_logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
logger = get_logger(__name__)


class CodeKnowledgePipeline:
    """End-to-end pipeline for code repository analysis.

    Usage:
        pipeline = CodeKnowledgePipeline()
        pipeline.ingest("https://github.com/user/repo.git")
        answer = pipeline.ask("How does the main module work?")
    """

    def __init__(self, settings: Any | None = None):
        self.settings = settings or get_settings()
        self.repo_cloner = RepoCloner(cache_dir=self.settings.repo_cache_path)
        self.ast_parser = ASTParser()
        self.chunk_processor = ChunkProcessor(ast_parser=self.ast_parser)
        self.embedder = Embedder()
        self.vector_store = VectorStore(
            collection_name="code_knowledge",
            persist_dir=self.settings.vector_store_path,
        )
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            enable_bm25=True,
        )
        self.agent: CodeKnowledgeAgent | None = None
        self.tools: CodeKnowledgeTools | None = None
        self.repo_path: str | None = None
        self.repo_name: str | None = None

    def ingest(self, repo_url: str, branch: str | None = None) -> dict[str, Any]:
        """Clone a repository, parse it, and build the vector index."""
        logger.info("Starting ingestion for %s", repo_url)
        repo_path = self.repo_cloner.clone(repo_url, branch=branch)
        self.repo_path = str(repo_path)
        self.repo_name = repo_url.rstrip("/").split("/")[-1]
        if self.repo_name.endswith(".git"):
            self.repo_name = self.repo_name[:-4]
        self.vector_store.clear()
        chunks = self.chunk_processor.process_directory(
            str(repo_path),
            extensions={".py", ".js", ".jsx", ".ts", ".tsx"},
        )
        if not chunks:
            return {"status": "warning", "message": "No supported source files found.", "chunks_count": 0, "repo_name": self.repo_name}
        contents = [c.content for c in chunks]
        metadata = [c.metadata for c in chunks]
        ids = [c.id for c in chunks]
        self.vector_store.add_chunks(contents=contents, metadata=metadata, ids=ids)
        self.tools = CodeKnowledgeTools(retriever=self.retriever, repo_path=self.repo_path)
        self.agent = CodeKnowledgeAgent.create_with_defaults(self.tools, self.settings)
        stats = {
            "status": "success",
            "repo_name": self.repo_name,
            "chunks_count": len(chunks),
            "files_count": len(set(c.file_path for c in chunks)),
            "symbols_count": len(set(c.symbol_name for c in chunks)),
            "languages": list(set(c.language for c in chunks)),
        }
        logger.info("Ingestion complete: %d chunks from %d files", stats["chunks_count"], stats["files_count"])
        return stats

    def ask(self, query: str) -> str:
        """Ask a question about the ingested repository."""
        if not self.agent:
            return "Please ingest a repository first using the ingest() method."
        return self.agent.run(query)

    def get_repo_info(self) -> dict[str, Any]:
        if not self.repo_path:
            return {"status": "no_repo"}
        return self.repo_cloner.get_repo_info(self.repo_name or "")

    def get_dependency_graph(self) -> str:
        """Build a dependency graph of the repository and return as a PNG file path.
        
        Analyzes import relationships between files to create a visual
        architecture diagram showing module dependencies.
        """
        if not self.repo_path:
            return ""
        import re
        from pathlib import Path
        
        repo = Path(self.repo_path)
        G = nx.DiGraph()
        module_files: dict[str, list[str]] = defaultdict(list)
        
        # Find all source files and their imports
        for ext in [".py", ".js", ".ts", ".rs", ".go", ".java"]:
            for f in repo.rglob(f"*{ext}"):
                if any(p.startswith(".") or p == "__pycache__" or p == "node_modules" for p in f.parts):
                    continue
                rel_path = str(f.relative_to(repo))
                module = f.parent.name or "root"
                module_files[module].append(rel_path)
                G.add_node(rel_path, module=module, size=1)
                
                # Detect imports
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                    if ext == ".py":
                        imports = re.findall(r"^import (\S+)|^from (\S+) import", text, re.MULTILINE)
                        for imp in imports:
                            target = imp[0] or imp[1]
                            if target:
                                G.add_edge(rel_path, target.split(".")[0])
                    elif ext in (".js", ".ts"):
                        imports = re.findall(r'(?:import|require)\s*\(?\s*["'"'"']\.\.?/([^"'"'"']+)', text)
                        for imp in imports:
                            G.add_edge(rel_path, imp)
                    elif ext == ".rs":
                        imports = re.findall(r"^use\s+([^;]+)", text, re.MULTILINE)
                        for imp in imports:
                            target = imp.split("::")[0]
                            G.add_edge(rel_path, target)
                    elif ext == ".go":
                        imports = re.findall(r"^import\s+"(?:[\w/]+/)?(\w+)"", text, re.MULTILINE)
                        for imp in imports:
                            G.add_edge(rel_path, imp)
                    elif ext == ".java":
                        imports = re.findall(r"^import\s+(?:[\w.]+\.)*(\w+);", text, re.MULTILINE)
                        for imp in imports:
                            G.add_edge(rel_path, imp)
                except Exception:
                    continue

        if len(G.nodes) == 0:
            return ""
        
        # Remove isolated nodes (keep only nodes with edges)
        isolated = [n for n in G.nodes() if G.degree(n) == 0]
        G.remove_nodes_from(isolated)
        
        if len(G.nodes) == 0:
            return ""
        
        # Layout and render
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        pos = nx.spring_layout(G, k=0.6, iterations=30, seed=42)
        modules = list(set(nx.get_node_attributes(G, "module").values()))
        colors = plt.cm.tab20(range(len(modules)))
        module_color = {m: colors[i] for i, m in enumerate(modules)}
        
        node_colors = [module_color.get(G.nodes[n].get("module", "root"), "gray") for n in G.nodes()]
        node_sizes = [300 + 50 * G.out_degree(n) for n in G.nodes()]
        
        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                with_labels=True, font_size=7, font_family="sans-serif",
                edge_color="gray", alpha=0.7, arrows=True, arrowsize=12,
                arrowstyle="->")
        
        ax.set_title("Code Repository Dependency Graph", fontsize=14, pad=20)
        ax.axis("off")
        
        # Save to temp file
        out_path = os.path.join(os.environ.get("TEMP", "/tmp"), "codeknowledge_depgraph.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        
        return out_path
    
    def get_index_stats(self) -> dict[str, Any]:
        all_metadata = self.vector_store.get_all_metadata()
        if not all_metadata:
            return {"total_chunks": 0}
        kinds: dict[str, int] = {}
        languages: dict[str, int] = {}
        files: set[str] = set()
        for m in all_metadata:
            kind = m.get("symbol_kind", "unknown")
            kinds[kind] = kinds.get(kind, 0) + 1
            lang = m.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
            fp = m.get("file_path", "")
            if fp:
                files.add(fp)
        return {"total_chunks": len(all_metadata), "total_files": len(files), "by_kind": kinds, "by_language": languages}

    def reset(self) -> None:
        self.vector_store.clear()
        self.agent = None
        self.tools = None
        self.repo_path = None
        self.repo_name = None
        logger.info("Pipeline reset")

    def cleanup(self) -> None:
        self.reset()
        import shutil
        shutil.rmtree(self.settings.vector_store_path, ignore_errors=True)
        shutil.rmtree(self.settings.repo_cache_path, ignore_errors=True)
        logger.info("Cleanup complete")

    @property
    def is_ready(self) -> bool:
        return self.agent is not None and self.vector_store.count() > 0
