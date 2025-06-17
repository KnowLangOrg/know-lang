from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime # Added
from pathlib import Path
from typing import Dict, List, Tuple # Added Tuple
from rich.progress import track

from knowlang.configs import AppConfig
from knowlang.core.types import CodeChunk
from knowlang.indexing.chunk_indexer import ChunkIndexer
# Removed: from knowlang.indexing.codebase_manager import CodebaseManager
from knowlang.indexing.state_manager import StateManager
from knowlang.indexing.state_store.base import FileChange, FileState, StateChangeType # Added FileState
from knowlang.indexing.file_utils import compute_file_hash # Added
from knowlang.utils import FancyLogger # Removed convert_to_relative_path

LOG = FancyLogger(__name__)

@dataclass
class UpdateStats:
    """Statistics about the incremental update process"""
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    errors: int = 0

    def summary(self) -> str:
        """Get a human-readable summary of the update stats"""
        return (
            f"Update completed:\n"
            f"  Files: {self.files_added} added, {self.files_modified} modified, "
            f"{self.files_deleted} deleted\n"
            f"  Chunks: {self.chunks_added} added, {self.chunks_deleted} deleted\n"
            f"  Errors: {self.errors}"
        )

class IncrementalUpdater:
    """Orchestrates incremental updates to the vector store"""
    
    def __init__(
        self,
        app_config: AppConfig,
        alias_to_repo_path: Dict[str, Path] # Added
    ):
        self.app_config = app_config
        self.alias_to_repo_path = alias_to_repo_path # Stored
        # self.codebase_manager = CodebaseManager(app_config) # Removed
        self.state_manager = StateManager(app_config) # Stays
        self.chunk_indexer = ChunkIndexer(app_config) # Stays

    def _group_chunks_by_file(self, chunks: List[CodeChunk]) -> Dict[Tuple[str, str], List[CodeChunk]]:
        """Group chunks by their (root_alias, relative_file_path)"""
        chunks_by_file = defaultdict(list)
        for chunk in chunks:
            # chunk.location.file_path is already relative string
            key = (chunk.root_alias, chunk.location.file_path)
            chunks_by_file[key].append(chunk)
        return dict(chunks_by_file)

    async def process_changes(
        self,
        changes: List[FileChange],
        chunks: List[CodeChunk]
    ) -> UpdateStats:
        """Process detected changes and update vector store"""
        stats = UpdateStats()
        chunks_by_file = self._group_chunks_by_file(chunks) # Key is now (root_alias, relative_path_str)
        
        for change in track(changes, description="Processing code changes"):
            try:
                # Handle deletions and modifications (remove old chunks)
                # change.path for DELETED is relative, for MODIFIED is absolute
                # StateManager methods expect relative path for file_path arg

                path_for_state_ops: Path
                if change.change_type == StateChangeType.DELETED:
                    # change.path is already relative for DELETED
                    path_for_state_ops = change.path
                else: # ADDED or MODIFIED, change.path is absolute
                    path_for_state_ops = change.path.relative_to(self.alias_to_repo_path[change.root_alias])

                if change.change_type in (StateChangeType.MODIFIED, StateChangeType.DELETED):
                    # Pass root_alias to state_manager methods
                    old_state = await self.state_manager.get_file_state(path_for_state_ops, change.root_alias)
                    if old_state and old_state.chunk_ids:
                        stats.chunks_deleted += len(old_state.chunk_ids)
                    # delete_file_state needs relative path and root_alias
                    await self.state_manager.delete_file_state(path_for_state_ops, change.root_alias)
                
                # Handle additions and modifications (add new chunks)
                if change.change_type in (StateChangeType.ADDED, StateChangeType.MODIFIED):
                    # change.path is absolute for ADDED/MODIFIED
                    absolute_file_path = change.path

                    # Key for chunks_by_file is (root_alias, relative_path_str)
                    # relative_path_for_chunk_key is relative to its source root.
                    relative_path_for_chunk_key = str(absolute_file_path.relative_to(self.alias_to_repo_path[change.root_alias]))
                    lookup_key = (change.root_alias, relative_path_for_chunk_key)

                    if lookup_key in chunks_by_file:
                        file_chunks = chunks_by_file[lookup_key]
                        processed_chunk_ids = await self.chunk_indexer.process_file_chunks(file_chunks)
                        
                        if processed_chunk_ids:
                            # The file_path stored in FileState must be relative to its root_alias's path
                            relative_path_for_state = absolute_file_path.relative_to(self.alias_to_repo_path[change.root_alias])

                            # Construct FileState directly, no CodebaseManager.create_file_state
                            new_filestate_obj = FileState(
                                file_path=str(relative_path_for_state),
                                root_alias=change.root_alias,
                                last_modified=datetime.fromtimestamp(absolute_file_path.stat().st_mtime),
                                file_hash=compute_file_hash(absolute_file_path),
                                chunk_ids=processed_chunk_ids
                            )

                            # Call state_manager.update_file_state with relative path for file_path argument
                            # and absolute_file_path for hashing.
                            await self.state_manager.update_file_state(
                                file_path=relative_path_for_state, # This is Path object
                                root_alias=change.root_alias,
                                chunk_ids=list(processed_chunk_ids), # Ensure it's a list for DB
                                absolute_file_path=absolute_file_path # Absolute path for hashing
                            )
                            stats.chunks_added += len(processed_chunk_ids)
                
                # Update stats
                if change.change_type == StateChangeType.ADDED:
                    stats.files_added += 1
                elif change.change_type == StateChangeType.MODIFIED:
                    stats.files_modified += 1
                elif change.change_type == StateChangeType.DELETED:
                    stats.files_deleted += 1
                
            except Exception as e:
                LOG.error(f"Error processing change for {change.path}: {e}")
                stats.errors += 1
                continue
        
        LOG.info(stats.summary())
        return stats

    async def update_codebase(self, chunks: List[CodeChunk], file_changes: List[FileChange]) -> UpdateStats:
        """High-level method to update entire codebase incrementally"""
        try:
            if not file_changes:
                LOG.info("No changes detected in codebase")
                return UpdateStats()
            
            # Process changes
            return await self.process_changes(file_changes, chunks)
            
        except Exception as e:
            LOG.error(f"Error updating codebase: {e}")
            return UpdateStats(errors=1)