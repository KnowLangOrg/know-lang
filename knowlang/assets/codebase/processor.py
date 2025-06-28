from typing import List, AsyncGenerator, TypeAlias
import os
from pathlib import Path
import uuid
import aiofiles
from knowlang.assets.processor import (
    DomainAssetSourceMixin,
    DomainAssetIndexingMixin,
    DomainAssetParserMixin,
    DomainContext,
)
from knowlang.assets.codebase.models import (
    CodebaseMetaData,
    CodeAssetMetaData,
    CodeAssetChunkMetaData,
    CodebaseManagerData,
    CodeAssetData,
    CodeAssetChunkData,
    CodeProcessorConfig,
)
from knowlang.assets.models import (
    GenericAssetData,
)
from knowlang.parser.factory import CodeParserFactory
from knowlang.utils.fancy_log import FancyLogger

LOG = FancyLogger(__file__)

# Type aliases to eliminate repetition
CodebaseConfigType: TypeAlias = CodeProcessorConfig

# Main context type alias
CodebaseDomainContext: TypeAlias = DomainContext[
        CodebaseManagerData,
        CodeAssetData, 
        CodeAssetChunkData,
    ]


class CodebaseAssetSource(DomainAssetSourceMixin):
    """Handles source management for codebase assets."""

    async def yield_all_assets(
        self,
        ctx: CodebaseDomainContext = None,
    ) -> AsyncGenerator[CodeAssetData, None]:
        """Get all assets for the codebase."""
        if ctx is not None:
            self.ctx = ctx
        ctx = self.ctx

        import zlib
        from git import Repo, InvalidGitRepositoryError
        assert isinstance(ctx.config, CodeProcessorConfig)

        domain = ctx.domain
        dir_path = ctx.config.directory_path
        path_pattern = ctx.config.path_patterns
        try:
            repo = Repo(dir_path)
        except InvalidGitRepositoryError:
            repo = None

        for top, dirs, files in os.walk(dir_path):
            # Skip git-ignored directories early
            if repo:
                # Modify dirs in-place to skip ignored directories
                dirs[:] = [d for d in dirs if not repo.ignored(os.path.join(top, d))]

            for file in files:
                file_path = os.path.join(top, file)

                if not path_pattern.should_process_path(file_path):
                    continue

                if repo and repo.ignored(file_path):
                    continue

                async with aiofiles.open(file_path, "rb") as f:
                    file_content = await f.read()
                    file_hash = zlib.crc32(file_content)

                relative_path = os.path.relpath(file_path, dir_path)
                asset_data = GenericAssetData(
                    domain_id=domain.id,
                    # id should be uuid, using relative path for name
                    id=str(uuid.uuid4()),
                    name=relative_path,
                    asset_manager_id=domain.id,
                    asset_hash=str(file_hash),
                    meta=CodeAssetMetaData(
                        file_path=file_path,
                    ),
                )
                yield asset_data


class CodebaseAssetIndexing(DomainAssetIndexingMixin):
    """Handles indexing of codebase assets."""

    def __init__(
        self,
        ctx: CodebaseDomainContext,
    ) -> None:
        super().__init__(ctx)
        from knowlang.vector_stores.factory import VectorStoreFactory
        self.vector_store = VectorStoreFactory.get(ctx.config.vector_store)

    async def index_chunks(
        self,
        chunks: List[CodeAssetChunkData],
        ctx: CodebaseDomainContext = None,
    ) -> None:
        """Index the given codebase assets."""
        if ctx is not None:
            self.ctx = ctx
            self.ctx.asset_chunks.extend(chunks)

        from knowlang.models import generate_embedding
        embedding_cfg = self.ctx.config.vector_store.embedding

        for chunk in chunks:
            assert isinstance(chunk.meta, CodeAssetChunkMetaData)
            embedding = await generate_embedding(chunk.meta.content, embedding_cfg)
            await self.vector_store.add_documents(
                documents=[chunk.meta.content],
                embeddings=[embedding],
                metadatas=[chunk.meta.model_dump()],
                ids=[chunk.id],
            )
        
        if chunks:
            LOG.debug(f"Indexed {len(chunks)} asset chunks from domain: {self.ctx.domain.name}")

    async def delete_chunks(self, chunks: List[CodeAssetChunkData], ctx: CodebaseDomainContext = None) -> None:
        """Delete the given asset chunks."""
        if ctx is not None:
            self.ctx = ctx
        
        LOG.debug(f"Deleting {len(chunks)} asset chunks from domain: {self.ctx.domain.name}")
        await self.vector_store.delete(ids=[chunk.id for chunk in chunks])


class CodebaseAssetParser(DomainAssetParserMixin):
    """Handles parsing of codebase assets."""

    def __init__(
        self,
        ctx: CodebaseDomainContext,
    ) -> None:
        super().__init__(ctx)
        self.code_parser_factory = CodeParserFactory(ctx.config)

    async def parse_assets(
        self,
        assets: List[CodeAssetData],
        ctx: CodebaseDomainContext = None,
    ) -> List[CodeAssetChunkData]:
        """Parse the given codebase assets."""
        if ctx is not None:
            self.ctx = ctx
            self.ctx.assets.extend(assets)

        chunks = []
        for asset in assets:
            assert isinstance(asset.meta, CodeAssetMetaData)

            file_path = Path(asset.meta.file_path)
            parser = self.code_parser_factory.get_parser(file_path)
            if parser is None:
                LOG.debug(f"No parser found for file: {file_path}, skipping.")
                continue

            LOG.debug(f"Parsing file: {file_path} with parser: {parser.__class__.__name__}")
            _chunks_raw = await parser.parse_file(file_path)
            curr_chunks = [CodeAssetChunkData(
                    id=str(uuid.uuid4()),
                    name=chunk.location.to_single_line(),
                    asset_id=asset.id,
                    content=chunk.content,
                    meta=CodeAssetChunkMetaData.from_code_chunk(chunk),
                ) for chunk in _chunks_raw]

            chunks.extend(curr_chunks)

        return chunks
