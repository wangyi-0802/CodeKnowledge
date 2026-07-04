"""Clone and manage Git repositories for analysis."""

from __future__ import annotations

import shutil
from pathlib import Path
from git import Repo, GitCommandError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RepoCloner:
    """Clone and manage repositories for code analysis."""

    def __init__(self, cache_dir: str = "./repo_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clone(self, repo_url: str, branch: str | None = None) -> Path:
        """Clone a repository and return the local path.

        If already cached, pull latest instead of re-cloning.
        """
        repo_name = self._extract_repo_name(repo_url)
        local_path = self.cache_dir / repo_name

        if local_path.exists():
            logger.info("Repo already cached at %s, pulling latest...", local_path)
            try:
                repo = Repo(local_path)
                origin = repo.remotes.origin
                origin.pull()
                logger.info("Pulled latest for %s", repo_name)
            except GitCommandError as e:
                logger.warning("Pull failed, using cached version: %s", e)
            finally:
                repo.close()
        else:
            logger.info("Cloning %s into %s...", repo_url, local_path)
            try:
                repo = Repo.clone_from(repo_url, local_path, branch=branch, depth=1)
                logger.info("Clone complete: %s", repo_name)
                repo.close()
            except GitCommandError as e:
                logger.error("Failed to clone %s: %s", repo_url, e)
                raise

        return local_path

    def get_repo_info(self, repo_url: str) -> dict:
        """Get basic repository information."""
        repo_name = self._extract_repo_name(repo_url)
        local_path = self.cache_dir / repo_name

        if not local_path.exists():
            return {"url": repo_url, "name": repo_name, "cloned": False}

        try:
            repo = Repo(local_path)
            info = {
                "url": repo_url,
                "name": repo_name,
                "cloned": True,
                "branch": repo.active_branch.name,
                "commit": str(repo.head.commit.hexsha[:8]) if repo.head.is_valid() else "N/A",
                "files_count": len(list(local_path.rglob("*.py"))) if repo_name else 0,
            }
            repo.close()
            return info
        except Exception as e:
            logger.warning("Failed to get repo info: %s", e)
            return {"url": repo_url, "name": repo_name, "error": str(e)}

    def clean_cache(self, repo_url: str | None = None) -> None:
        """Remove cached repository data."""
        if repo_url:
            repo_name = self._extract_repo_name(repo_url)
            path = self.cache_dir / repo_name
            if path.exists():
                shutil.rmtree(path)
                logger.info("Removed cache for %s", repo_name)
        else:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared entire repo cache")

    @staticmethod
    def _extract_repo_name(repo_url: str) -> str:
        """Extract repository name from URL."""
        repo_url = repo_url.rstrip("/")
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]
        return repo_url.split("/")[-2] + "_" + repo_url.split("/")[-1]
