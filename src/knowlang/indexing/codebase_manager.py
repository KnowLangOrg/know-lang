from pathlib import Path
import hashlib
from datetime import datetime
from typing import Set
from git import Repo, InvalidGitRepositoryError
import os

from knowlang.indexing.state_store.base import FileState
from knowlang.utils.chunking_util import convert_to_relative_path
from knowlang.configs.config import AppConfig
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__name__)

class CodebaseManager:
    """Manages file-level operations and state creation"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.repo = self._init_git_repo()
        
    def _init_git_repo(self) -> Repo | None:
        """Initialize git repo if the codebase directory is a git repository"""
        try:
            if (self.config.db.codebase_directory / '.git').exists():
                return Repo(self.config.db.codebase_directory)
            return None
        except InvalidGitRepositoryError:
            return None
        
    async def get_current_files(self) -> Set[Path]:
        """Get set of current files in directory with proper filtering"""
        current_files = set()
        
        try:
            # Convert to string for os.walk
            root_dir = str(self.config.db.codebase_directory)
            
            for root, dirs, files in os.walk(root_dir):
                # Skip git-ignored directories early
                if self.repo:
                    # Modify dirs in-place to skip ignored directories
                    dirs[:] = [d for d in dirs if not self.repo.ignored(Path(root) / d)]
                
                for file in files:
                    path = Path(root) / file
                    relative_path = convert_to_relative_path(path, self.config.db)
                    
                    # Skip if path shouldn't be processed based on patterns
                    if not self.config.parser.path_patterns.should_process_path(relative_path):
                        continue
                        
                    # Skip if individual file is git-ignored
                    if self.repo and self.repo.ignored(path):
                        continue
                        
                    current_files.add(path)
            
            return current_files
            
        except Exception as e:
            LOG.error(f"Error scanning directory {self.config.db.codebase_directory}: {e}")
            raise

    async def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def create_file_state(self, file_path: Path, chunk_ids: Set[str]) -> FileState:
        """Create a new FileState object for a file"""
        return FileState(
            file_path=str(file_path),
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
            file_hash=await self.compute_file_hash(file_path),
            chunk_ids=chunk_ids
        )