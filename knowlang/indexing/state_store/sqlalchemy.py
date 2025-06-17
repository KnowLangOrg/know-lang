import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import (Column, DateTime, ForeignKey, Integer, String,
                        UniqueConstraint, create_engine, select)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from knowlang.configs import AppConfig, DBConfig
from knowlang.core.types import StateStoreProvider
from knowlang.indexing.file_utils import (compute_file_hash) # get_absolute_path, get_relative_path removed as per plan
from knowlang.utils import FancyLogger

from .base import (FileChange, FileState, StateChangeType, StateStore,
                   register_state_store)

LOG = FancyLogger(__name__)
Base = declarative_base()

class FileStateModel(Base):
    """SQLAlchemy model for file states"""
    __tablename__ = 'file_states'
    __table_args__ = (UniqueConstraint('root_alias', 'file_path', name='_root_alias_file_path_uc'),)
    
    id = Column(Integer, primary_key=True)
    root_alias = Column(String, index=True)
    file_path = Column(String, index=True) # Removed unique=True
    last_modified = Column(DateTime)
    file_hash = Column(String)
    chunks = relationship(
        "ChunkStateModel", 
        back_populates="file", 
        cascade="all, delete-orphan"
    )

class ChunkStateModel(Base):
    """SQLAlchemy model for chunk states"""
    __tablename__ = 'chunk_states'
    
    id = Column(Integer, primary_key=True)
    chunk_id = Column(String, unique=True, index=True)
    file_id = Column(Integer, ForeignKey('file_states.id'))
    file = relationship("FileStateModel", back_populates="chunks")

@register_state_store(StateStoreProvider.SQLITE)
@register_state_store(StateStoreProvider.POSTGRES)
class SQLAlchemyStateStore(StateStore):
    """SQLAlchemy-based state storage implementation supporting both SQLite and PostgreSQL"""
    def __init__(self, config: AppConfig):
        """Initialize database with configuration and create schema if needed"""
        self.app_config = config
        self.config = DBConfig.model_validate(config.db)
        
        # Validate store type
        if self.config.state_store.provider not in (StateStoreProvider.SQLITE, StateStoreProvider.POSTGRES):
            raise ValueError(f"Invalid store type: {self.config.state_store.provider}")
            
        # Initialize database connection
        connection_args = self.config.state_store.get_connection_args()
        self.engine = create_engine(
            connection_args.pop('url'),
            **connection_args
        )
        self.Session = sessionmaker(bind=self.engine)

        # Create database schema if it doesn't exist
        Base.metadata.create_all(self.engine)
        
        LOG.info(f"Initialized {self.config.state_store.provider} state store schema at {self.config.state_store.store_path}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except IOError as e:
            LOG.error(f"Error computing hash for {file_path}: {e}")
            raise

    async def get_file_state(self, file_path: Path, root_alias: str) -> Optional[FileState]:
        """Get current state of a file. file_path is assumed to be relative to its root_alias."""
        try:
            with self.Session() as session:
                relative_path_str = str(file_path)
                stmt = select(FileStateModel).where(
                    FileStateModel.file_path == relative_path_str,
                    FileStateModel.root_alias == root_alias
                )
                result = session.execute(stmt).scalar_one_or_none()
                
                return (FileState(
                    root_alias=result.root_alias,
                    file_path=result.file_path,
                    last_modified=result.last_modified,
                    file_hash=result.file_hash,
                    chunk_ids={chunk.chunk_id for chunk in result.chunks}
                ) if result else None)
        except SQLAlchemyError as e:
            LOG.error(f"Database error getting file state for {root_alias}::{file_path}: {e}")
            raise

    async def update_file_state(
        self,
        file_path: Path, # Assumed to be relative to its root_alias
        root_alias: str,
        chunk_ids: List[str],
        # absolute_file_path is needed for hashing and mtime, will be passed by CodebaseManager
        absolute_file_path: Path
    ) -> None:
        """Update or create file state. file_path is relative, absolute_file_path is for disk access."""
        try:
            with self.Session() as session:
                relative_path_str = str(file_path)
                
                # Compute new file hash using the absolute path
                file_hash = compute_file_hash(absolute_file_path)
                current_mtime = datetime.fromtimestamp(absolute_file_path.stat().st_mtime)
                
                # Get or create file state
                file_state = session.execute(
                    select(FileStateModel).where(
                        FileStateModel.file_path == relative_path_str,
                        FileStateModel.root_alias == root_alias
                    )
                ).scalar_one_or_none()
                
                if not file_state:
                    file_state = FileStateModel(
                        root_alias=root_alias,
                        file_path=relative_path_str,
                        last_modified=current_mtime,
                        file_hash=file_hash
                    )
                    session.add(file_state)
                else:
                    file_state.last_modified = current_mtime
                    file_state.file_hash = file_hash
                
                # Update chunks
                # Ensure file_state.id is available if it's a new object
                session.flush()

                # Efficiently delete old chunks and add new ones
                # First, get existing chunk IDs for this file
                existing_chunk_stmt = select(ChunkStateModel.chunk_id).where(ChunkStateModel.file_id == file_state.id)
                existing_db_chunk_ids = {res[0] for res in session.execute(existing_chunk_stmt).all()}

                new_chunk_ids_set = set(chunk_ids)

                # Chunks to delete
                chunks_to_delete = existing_db_chunk_ids - new_chunk_ids_set
                if chunks_to_delete:
                    delete_stmt = ChunkStateModel.__table__.delete().where(
                        ChunkStateModel.file_id == file_state.id,
                        ChunkStateModel.chunk_id.in_(chunks_to_delete)
                    )
                    session.execute(delete_stmt)
                
                # Chunks to add
                chunks_to_add = new_chunk_ids_set - existing_db_chunk_ids
                for chunk_id in chunks_to_add:
                    chunk_state = ChunkStateModel(
                        chunk_id=chunk_id,
                        file_id=file_state.id # Directly use file_id
                    )
                    session.add(chunk_state)
                
                session.commit()
                
        except SQLAlchemyError as e:
            LOG.error(f"Database error updating file state for {root_alias}::{file_path}: {e}")
            raise

    async def delete_file_state(self, file_path: Path, root_alias: str) -> Set[str]:
        """Delete file state and return associated chunk IDs. file_path is relative."""
        try:
            with self.Session() as session:
                relative_path_str = str(file_path)
                
                file_state = session.execute(
                    select(FileStateModel).where(
                        FileStateModel.file_path == relative_path_str,
                        FileStateModel.root_alias == root_alias
                    )
                ).scalar_one_or_none()
                
                if file_state:
                    chunk_ids = {chunk.chunk_id for chunk in file_state.chunks}
                    session.delete(file_state) # Cascading delete should handle chunks
                    session.commit()
                    return chunk_ids
                
                return set()
                
        except SQLAlchemyError as e:
            LOG.error(f"Database error deleting file state for {root_alias}::{file_path}: {e}")
            raise

    async def get_all_file_states(self) -> List[FileState]:
        """Get all file states, with file_path being relative."""
        try:
            with self.Session() as session:
                stmt = select(FileStateModel).options(relationship(FileStateModel.chunks))
                results = session.execute(stmt).scalars().all()
                
                return [
                    FileState(
                        root_alias=state.root_alias,
                        file_path=state.file_path, # This is already relative
                        last_modified=state.last_modified,
                        file_hash=state.file_hash,
                        chunk_ids={chunk.chunk_id for chunk in state.chunks}
                    )
                    for state in results
                ]
                
        except SQLAlchemyError as e:
            LOG.error(f"Database error getting all file states: {e}")
            raise

    async def detect_changes(self, current_files: Set[Tuple[str, Path, Path]]) -> List[FileChange]:
        """
        Detect changes in files since last update.
        current_files is a set of (root_alias, relative_path, absolute_path) tuples.
        """
        try:
            changes = []
            all_db_states = await self.get_all_file_states() # List[FileState]

            # Create a map for easy lookup: (root_alias, relative_path_str) -> FileState
            existing_states_map: Dict[Tuple[str, str], FileState] = {
                (state.root_alias, state.file_path): state for state in all_db_states
            }

            current_files_map: Dict[Tuple[str, str], Path] = {
                (alias, str(rel_path)): abs_path for alias, rel_path, abs_path in current_files
            }
            
            # Check for new and modified files
            for (alias, rel_path_str), abs_path in current_files_map.items():
                if not abs_path.exists(): # Use absolute path for exists check
                    LOG.warning(f"File {abs_path} from current_files list does not exist. Skipping.")
                    continue
                    
                current_hash = compute_file_hash(abs_path) # Use absolute path for hash
                current_mtime = datetime.fromtimestamp(abs_path.stat().st_mtime) # Use absolute path
                
                state_key = (alias, rel_path_str)
                if state_key not in existing_states_map:
                    changes.append(FileChange(
                        root_alias=alias,
                        path=abs_path, # FileChange path should be absolute for processor
                        change_type=StateChangeType.ADDED,
                        old_chunks=None
                    ))
                else:
                    state = existing_states_map[state_key]
                    if (state.file_hash != current_hash or 
                        state.last_modified.replace(microsecond=0) != current_mtime.replace(microsecond=0)): # Compare mtime without microseconds
                        changes.append(FileChange(
                            root_alias=alias, # or state.root_alias, they should match
                            path=abs_path, # Absolute path
                            change_type=StateChangeType.MODIFIED,
                            old_chunks=state.chunk_ids
                        ))
            
            # Check for deleted files
            # Create a set of (alias, relative_path_str) from current_files for efficient lookup
            current_files_set_for_comparison = set(current_files_map.keys())

            for (alias, path_str), state in existing_states_map.items():
                # Construct the absolute path for the FileChange object if it's deleted.
                # This requires knowing the original base_path for this alias, which is not
                # directly available here. This points to a need for `CodebaseManager` to handle
                # constructing the absolute path for deleted files, or `FileChange` needs `root_alias`.
                # For now, we'll use a placeholder or relative path.
                # Decision: FileChange.path should be the absolute path.
                # This means `detect_changes` needs to be able to reconstruct it, or the caller does.
                # The current `current_files` tuple provides absolute_path. For deleted files,
                # we need to reconstruct it. This is tricky.
                #
                # Let's assume the caller (CodebaseManager) will handle resolving paths for deleted files.
                # So, FileChange.path for deleted files will be the *relative* path for now,
                # and the `root_alias` will be needed by the consumer to know *which* root it's relative to.
                # This implies FileChange itself might need root_alias.
                # For this subtask, we stick to Path for `FileChange.path`.
                # The problem is which path to use for deleted files.
                # The `config.codebase_directory` is gone.
                #
                # Option: The `FileState` object has `file_path` (relative) and `root_alias`.
                # The `FileChange` object for DELETED will carry this relative path.
                # The consumer (CodeProcessor) will need to be alias-aware.
                # For now, we'll pass the relative path.
                #
                # Revisit: `current_files` contains absolute paths. `FileChange.path` should be absolute.
                # For DELETED files, we need to reconstruct their absolute path.
                # This requires `CodebaseManager` to provide a way to get the root path for an alias.
                # Let's assume for now that `FileChange` for deleted files will store `Path(state.file_path)`
                # and it's understood to be relative, and the `root_alias` is available on `state`.
                # This is deferred to a later step where `FileChange` might be augmented or consumer logic updated.
                # For now, the path will be relative for DELETED, absolute for ADDED/MODIFIED. This is inconsistent.
                #
                # Let's make `FileChange.path` always relative and add `root_alias` to `FileChange`.
                # This is outside the scope of THIS subtask for `FileChange`.
                # So, for DELETED, `path` will be `Path(state.file_path)`.

                if (alias, path_str) not in current_files_set_for_comparison:
                    # This path is relative. The consumer needs to know its root via alias.
                    # This will be fixed when FileChange is updated or consumer is updated.
                    # For now, path will be `Path(state.file_path)` which is relative.
                    changes.append(FileChange(
                        root_alias=state.root_alias, # alias from existing_states_map key
                        path=Path(state.file_path), # This is relative path
                        change_type=StateChangeType.DELETED,
                        old_chunks=state.chunk_ids
                    ))
            
            return changes
            
        except Exception as e:
            LOG.error(f"Error detecting changes: {e}")
            # It's often better to log exception info for debugging
            import traceback
            LOG.error(traceback.format_exc())
            raise