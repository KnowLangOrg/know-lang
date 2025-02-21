from typing import List, Set
from pathlib import Path

from knowlang.core.types import CodeChunk, DatabaseChunkMetadata
from knowlang.indexing.indexing_agent import IndexingAgent
from knowlang.models.embeddings import generate_embedding
from knowlang.vector_stores.base import VectorStore
from knowlang.configs.config import AppConfig
from knowlang.utils.fancy_log import FancyLogger
from knowlang.vector_stores.factory import VectorStoreFactory

LOG = FancyLogger(__name__)

class ChunkIndexer:
    """Handles processing of code chunks including summary and embedding generation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.vector_store = VectorStoreFactory.get(config.db)
        self.indexing_agent = IndexingAgent(config)

    async def process_chunk(self, chunk: CodeChunk) -> str:
        """Process a single chunk and store in vector store"""
        try:
            # Get summary from indexing agent
            summary = await self.indexing_agent.summarize_chunk(chunk)
            
            # Generate embedding
            embedding = generate_embedding(summary, self.config.embedding)
            
            # Create metadata
            metadata = DatabaseChunkMetadata.from_code_chunk(chunk)
            
            # Create unique ID
            chunk_id = chunk.location.to_single_line()
            
            # Store in vector store
            await self.vector_store.add_documents(
                documents=[summary],
                embeddings=[embedding],
                metadatas=[metadata.model_dump()],
                ids=[chunk_id]
            )
            
            return chunk_id
            
        except Exception as e:
            LOG.error(f"Error processing chunk {chunk.location}: {e}")
            raise

    async def process_file_chunks(self, file_path: Path, chunks: List[CodeChunk]) -> Set[str]:
        """Process all chunks from a single file"""
        chunk_ids = set()
        for chunk in chunks:
            try:
                chunk_id = await self.process_chunk(chunk)
                chunk_ids.add(chunk_id)
            except Exception as e:
                LOG.error(f"Error processing chunk in {file_path}: {e}")
                continue
        return chunk_ids