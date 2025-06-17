"""Command implementation for parsing codebases."""
from pathlib import Path

from knowlang.cli.display.formatters import get_formatter
from knowlang.cli.types import ParseCommandArgs
from knowlang.cli.utils import create_config
from knowlang.indexing.codebase_manager import CodebaseManager
from knowlang.indexing.increment_update import IncrementalUpdater
from knowlang.indexing.indexing_agent import IndexingAgent
from knowlang.indexing.state_manager import StateManager
from knowlang.indexing.state_store.base import StateChangeType
from knowlang.parser.factory import CodeParserFactory
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)


async def parse_command(args: ParseCommandArgs) -> None:
    """Execute the parse command.

    Args:
        args: Typed command line arguments
    """
from typing import Dict, Set, Tuple # Added Set, Tuple

# ... (other imports remain the same)

LOG = FancyLogger(__name__)


async def parse_command(args: ParseCommandArgs) -> None:
    """Execute the parse command.
    
    Args:
        args: Typed command line arguments
    """
    # Load configuration
    config = create_config(args.config)
    
    # args.path might be used in the future to dynamically add a source
    # For now, we assume config.db.codebase_sources is populated.
    # config.db.codebase_directory = Path(args.path).resolve() # Removed
    # config.db.codebase_url = args.path # Removed
    if args.path:
        LOG.warning(f"CLI argument --path '{args.path}' is currently not used to dynamically add a codebase source. Please define sources in the configuration file.")

    config.extra_fields = args.extra_fields
    
    if not config.db.codebase_sources:
        LOG.error("No codebase_sources defined in the configuration. Exiting.")
        return

    all_codebase_files_with_details: Set[Tuple[str, Path, Path]] = set()
    alias_to_codebase_manager: Dict[str, CodebaseManager] = {}
    alias_to_repo_path_map: Dict[str, Path] = {} # Added map

    LOG.info(f"Processing {len(config.db.codebase_sources)} codebase source(s).")
    for codebase_source in config.db.codebase_sources:
        LOG.info(f"Initializing codebase manager for source: '{codebase_source.alias}' at path '{codebase_source.path}'")
        codebase_manager = CodebaseManager(codebase_source, config)
        alias_to_codebase_manager[codebase_source.alias] = codebase_manager
        alias_to_repo_path_map[codebase_source.alias] = codebase_manager.repo_path # Populate map

        source_files = await codebase_manager.get_current_files()
        LOG.info(f"Found {len(source_files)} files in source '{codebase_source.alias}'.")
        all_codebase_files_with_details.update(source_files)

    state_manager = StateManager(config) # Instantiated once
    
    LOG.info(f"Total files found across all sources: {len(all_codebase_files_with_details)}")
    file_changes = await state_manager.state_store.detect_changes(all_codebase_files_with_details)
    LOG.info(f"Detected {len(file_changes)} file changes across all sources.")

    code_parser_factory = CodeParserFactory(config) # Instantiated once
    total_chunks = []

    for change in file_changes:
        if change.change_type == StateChangeType.DELETED:
            LOG.info(f"File marked for deletion (chunks will be removed from DB): {change.root_alias}::{change.path}")
            continue

        # For ADDED or MODIFIED files, change.path is absolute
        absolute_path_to_parse = change.path
        current_alias = change.root_alias

        current_codebase_manager = alias_to_codebase_manager.get(current_alias)
        if not current_codebase_manager:
            LOG.error(f"Could not find CodebaseManager for alias '{current_alias}' from FileChange object. Skipping file {absolute_path_to_parse}.")
            continue
        
        # repo_path is the actual root path for this source (could be temp dir)
        current_source_root_path = current_codebase_manager.repo_path

        LOG.info(f"Parsing {change.change_type} file: {current_alias}::{absolute_path_to_parse.relative_to(current_source_root_path)}")

        parser = code_parser_factory.get_parser(absolute_path_to_parse, current_source_root_path)
        if parser:
            try:
                chunks = parser.parse_file(absolute_path_to_parse, current_alias)
                total_chunks.extend(chunks)
                LOG.debug(f"Found {len(chunks)} chunks in {current_alias}::{absolute_path_to_parse.relative_to(current_source_root_path)}")
            except Exception as e:
                LOG.error(f"Error parsing file {current_alias}::{absolute_path_to_parse.relative_to(current_source_root_path)}: {e}")
        else:
            LOG.warning(f"No parser found for file: {current_alias}::{absolute_path_to_parse.relative_to(current_source_root_path)}")


    updater = IncrementalUpdater(config, alias_to_repo_path_map) # Pass map to constructor
    # file_changes already contains root_alias for each change.
    # CodeChunk objects in total_chunks also contain their root_alias.
    await updater.update_codebase(
        chunks=total_chunks, 
        file_changes=file_changes
    )

    # Display results
    if total_chunks:
        LOG.info(f"Total code chunks processed from all sources: {len(total_chunks)}")
        formatter = get_formatter(args.output)
        formatter.display_chunks(total_chunks)
    else:
        LOG.warning("No code chunks found across all sources.")