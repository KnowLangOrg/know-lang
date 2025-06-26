from typing import List, AsyncGenerator, TypeAlias
import os
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
        ctx: CodebaseDomainContext,
    ) -> AsyncGenerator[CodeAssetData, None]:
        """Get all assets for the codebase."""

        import zlib
        from git import Repo, InvalidGitRepositoryError
        assert isinstance(ctx.config, CodeProcessorConfig)

        domain = ctx.domain
        dir_path = ctx.config.directory_path
        try:
            repo = Repo(dir_path)
        except InvalidGitRepositoryError:
            repo = None

        for top, dirs, files in os.walk(dir_path):
            for file in files:
                if repo and repo.ignored(file):
                    continue

                async with aiofiles.open(os.path.join(top, file), "rb") as f:
                    file_content = await f.read()
                    file_hash = zlib.crc32(file_content)

                file_path = os.path.join(top, file)
                relative_path = os.path.relpath(file_path, dir_path)
                asset_data = GenericAssetData(
                    domain_id=domain.id,
                    id=relative_path,
                    name=file,
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

    async def index_assets(
        self,
        ctx: CodebaseDomainContext,
    ) -> None:
        """Index the given codebase assets."""
        from knowlang.models import generate_embedding

        for asset in ctx.assets:
            for chunk in ctx.asset_chunks:
                assert isinstance(chunk.meta, CodeAssetChunkMetaData)
                embedding = await generate_embedding(chunk)
                self.vector_store.add_documents(
                    documents=[chunk.meta.content],
                    embeddings=[embedding],
                    metadatas=[chunk.meta.model_dump()],
                    ids=[chunk.chunk_id],
                )

        pass


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
        ctx: CodebaseDomainContext,
    ) -> List[CodeAssetChunkData]:
        """Parse the given codebase assets."""

        chunks = []
        for asset in ctx.assets:
            assert isinstance(asset.meta, CodeAssetMetaData)

            file_path = asset.meta.file_path
            parser = self.code_parser_factory.get_parser(file_path)
            _chunks_raw = await parser.parse_file(file_path)
            chunks.extend(
                [CodeAssetChunkData(
                    chunk_id=chunk.location.to_single_line(),
                    asset_id=asset.id,
                    content=chunk.content,
                    meta=CodeAssetChunkMetaData.from_code_chunk(chunk),
                ) for chunk in _chunks_raw]
            )

        return chunks
