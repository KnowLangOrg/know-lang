import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Set, Tuple

from git import InvalidGitRepositoryError, Repo

from knowlang.configs import AppConfig, CodebaseSource # Added CodebaseSource
from knowlang.indexing.file_utils import compute_file_hash # Removed get_relative_path
from knowlang.indexing.state_store.base import FileState
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)

class CodebaseManager:
    """Manages file-level operations and state creation for a single codebase source."""

    def __init__(self, codebase_source: CodebaseSource, app_config: AppConfig):
        self.codebase_source = codebase_source
        self.app_config = app_config # Renamed from self.config
        self.temp_dir = None # For cloned repos
        self.repo_path: Path = Path() # Effective path to the codebase
        self.repo: Repo | None = self._init_git_repo()
        
    def _init_git_repo(self) -> Repo | None:
        """
        Initialize git repo. Uses self.codebase_source.path.
        If self.codebase_source.url is provided, may clone into a temporary directory.
        Sets self.repo_path to the actual path of the codebase (original or temp).
        """
        source_path = self.codebase_source.path.resolve()
        source_url = self.codebase_source.url

        try:
            # Try to open existing repo at source_path
            if source_path.exists() and (source_path / '.git').is_dir():
                self.repo_path = source_path
                LOG.info(f"Using existing Git repository at: {self.repo_path} for alias '{self.codebase_source.alias}'")
                return Repo(self.repo_path)

            # If URL is provided, try cloning
            if source_url:
                LOG.info(f"Attempting to clone from URL: {source_url} for alias '{self.codebase_source.alias}'")
                self.temp_dir = tempfile.mkdtemp(prefix=f"repo_{self.codebase_source.alias}_")
                temp_path = Path(self.temp_dir)
                try:
                    repo = Repo.clone_from(source_url, temp_path)
                    self.repo_path = temp_path.resolve()
                    LOG.info(f"Successfully cloned repository to temporary directory: {self.repo_path}")
                    return repo
                except Exception as e:
                    LOG.warning(f"Failed to clone repository from {source_url}: {e}. Will try to use local path if exists.")
                    # If clone fails, and a temp_dir was created, clean it up if source_path doesn't exist either
                    if self.temp_dir and not source_path.exists():
                         shutil.rmtree(self.temp_dir)
                         self.temp_dir = None

            # If no repo at source_path and no URL or clone failed, check if source_path is a valid directory
            if source_path.is_dir():
                self.repo_path = source_path
                LOG.info(f"Using local directory (not a git repo): {self.repo_path} for alias '{self.codebase_source.alias}'")
                return None # Not a git repo, but a valid directory
            else:
                LOG.error(f"Codebase path {source_path} for alias '{self.codebase_source.alias}' is not a valid directory, and no usable git URL provided or clone failed.")
                self.repo_path = source_path # Set it anyway, subsequent operations might fail gracefully
                return None

        except InvalidGitRepositoryError:
            # Path exists but is not a git repo
            if source_path.is_dir():
                self.repo_path = source_path
                LOG.info(f"Path {self.repo_path} for alias '{self.codebase_source.alias}' is a directory but not a Git repository.")
                return None
            LOG.warning(f"Invalid Git repository at {source_path} for alias '{self.codebase_source.alias}'.")
            self.repo_path = source_path
            return None
        except Exception as e:
            LOG.error(f"Error initializing git repo for alias '{self.codebase_source.alias}' at {source_path}: {e}")
            self.repo_path = source_path # Fallback
            return None
        
    async def get_current_files(self) -> Set[Tuple[str, Path, Path]]:
        """
        Get set of current files in the codebase directory, properly filtered.
        Returns a set of tuples: (root_alias, relative_path, absolute_path).
        """
        current_files: Set[Tuple[str, Path, Path]] = set()
        
        if not self.repo_path or not self.repo_path.is_dir():
            LOG.error(f"Repository path {self.repo_path} for alias '{self.codebase_source.alias}' is not a valid directory. Cannot get current files.")
            return current_files

        try:
            root_dir_str = str(self.repo_path) # Use the effective repo_path
            
            for root, dirs, files in os.walk(root_dir_str):
                abs_root = Path(root)

                # Skip git-ignored directories early if it's a git repo
                if self.repo:
                    # Filter dirs in-place. self.repo.ignored expects paths relative to repo root or absolute.
                    # Here, Path(root) / d is absolute.
                    dirs[:] = [d for d in dirs if not self.repo.ignored(abs_root / d)]
                
                for file_name in files:
                    absolute_path = abs_root / file_name
                    
                    # Calculate relative path against the effective repo_path
                    try:
                        relative_path = absolute_path.relative_to(self.repo_path)
                    except ValueError:
                        LOG.warning(f"File {absolute_path} is not under repo_path {self.repo_path}. Skipping.")
                        continue

                    # Skip if path shouldn't be processed based on patterns
                    # Path patterns are typically relative to the source root.
                    if not self.app_config.parser.path_patterns.should_process_path(str(relative_path)):
                        continue
                        
                    # Skip if individual file is git-ignored (if it's a git repo)
                    if self.repo and self.repo.ignored(absolute_path):
                        continue
                        
                    current_files.add((self.codebase_source.alias, relative_path, absolute_path))
            
            return current_files
            
        except Exception as e:
            LOG.error(f"Error scanning directory {self.repo_path} for alias '{self.codebase_source.alias}': {e}")
            # It's often useful to re-raise or handle more specifically
            raise

    async def create_file_state(self, absolute_file_path: Path, relative_file_path: Path, chunk_ids: Set[str]) -> FileState:
        """Create a new FileState object for a file, using provided absolute and relative paths."""
        return FileState(
            root_alias=self.codebase_source.alias,
            file_path=str(relative_file_path), # Store relative path as string
            last_modified=datetime.fromtimestamp(absolute_file_path.stat().st_mtime),
            file_hash=compute_file_hash(absolute_file_path),
            chunk_ids=chunk_ids
        )
    
    def __del__(self):
        if self.temp_dir and Path(self.temp_dir).exists(): # Check if temp_dir was set and exists
            LOG.info(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None