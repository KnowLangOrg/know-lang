"""
Indexing functionality for code search benchmark datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from rich.progress import track
from tqdm.asyncio import tqdm

from knowlang.configs import AppConfig
from knowlang.evaluations.base import QueryCodePair
from knowlang.models import generate_embedding
from knowlang.models.types import EmbeddingInputType
from knowlang.utils import FancyLogger
from knowlang.vector_stores.factory import VectorStoreFactory

LOG = FancyLogger(__name__)


class DatasetIndexer:
    """Handles indexing of benchmark datasets into the vector store."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.vector_store = VectorStoreFactory.get(config)
    
    async def initialize(self):
        """Initialize the vector store."""
        try:
            self.vector_store.initialize()
            LOG.info(f"Vector store initialized: {type(self.vector_store).__name__}")
        except Exception as e:
            LOG.error(f"Failed to initialize vector store: {e}")
            raise Exception("Failed to initialize vector store") from e
    
    async def index_code_snippets(self, pairs: List[QueryCodePair], batch_size: int = 100) -> Set[str]:
        """
        Index code snippets from query-code pairs into the vector store.
        
        Args:
            pairs: List of query-code pairs to index
            batch_size: Number of snippets to index in each batch
            
        Returns:
            Set of indexed code IDs
        """
        indexed_ids = set()
        batch_count = (len(pairs) + batch_size - 1) // batch_size
        
        LOG.info(f"Indexing {len(pairs)} code snippets in {batch_count} batches")
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            
            try:
                # Process batch
                documents = []
                embeddings = []
                metadatas = []
                ids = []
                
                # Create summaries and embeddings
                for pair in batch:
                    try:
                        # Content to embed (just the code itself for now)
                        content = pair.code
                        
                        # Generate embedding
                        embedding = generate_embedding(
                            input=content, 
                            config=self.config.embedding,
                            input_type=EmbeddingInputType.DOCUMENT
                        )
                        
                        # Skip if embedding generation failed
                        if not embedding or len(embedding) == 0:
                            LOG.warning(f"Failed to generate embedding for code ID: {pair.code_id}")
                            continue
                        
                        # Create metadata
                        metadata = {
                            **pair.metadata,
                            "language": pair.language,
                            "content": content,
                            "queries": [pair.query],  # Store the original query for reference
                            "dataset_type": "benchmark",
                            "id": pair.code_id  # Ensure ID is in metadata for retrieval
                        }
                        
                        documents.append(content)
                        embeddings.append(embedding)
                        metadatas.append(metadata)
                        ids.append(pair.code_id)
                        indexed_ids.add(pair.code_id)
                        
                    except Exception as e:
                        LOG.error(f"Error processing code snippet {pair.code_id}: {e}")
                        continue
                
                # Skip if batch is empty
                if not documents:
                    continue
                
                # Store in vector store
                await self.vector_store.add_documents(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                LOG.info(f"Indexed batch {i//batch_size + 1}/{batch_count} ({len(documents)} snippets)")
                
            except Exception as e:
                LOG.error(f"Error indexing batch {i//batch_size + 1}: {e}")
                continue
        
        LOG.info(f"Indexing complete. Total indexed: {len(indexed_ids)}")
        return indexed_ids
    
    async def index_dataset(self, pairs: List[QueryCodePair]) -> Set[str]:
        """
        Index an entire dataset into the vector store.
        
        Args:
            pairs: List of query-code pairs to index
            
        Returns:
            Set of indexed code IDs
        """
        await self.initialize()
        return await self.index_code_snippets(pairs)


class QueryManager:
    """Manages the query-code mappings for evaluation."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_query_mappings(self, pairs: List[QueryCodePair], dataset_name: str):
        """
        Save query-code mappings to a file for evaluation.
        
        Args:
            pairs: List of query-code pairs
            dataset_name: Name of the dataset
        """
        # Create dictionary mapping query IDs to relevant code IDs
        query_map = {}
        for pair in pairs:
            if pair.query_id not in query_map:
                query_map[pair.query_id] = {
                    "query": pair.query,
                    "language": pair.language,
                    "relevant_code": []
                }
            
            if pair.is_relevant:
                query_map[pair.query_id]["relevant_code"].append(pair.code_id)
        
        # Save as JSON
        output_path = self.output_dir / f"{dataset_name}_query_map.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(query_map, f, indent=2)
        
        LOG.info(f"Saved query mappings to {output_path}")
    
    def load_query_mappings(self, dataset_name: str) -> Dict:
        """
        Load query-code mappings from a file.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary mapping query IDs to relevant code IDs
        """
        input_path = self.output_dir / f"{dataset_name}_query_map.json"
        if not input_path.exists():
            LOG.warning(f"Query mappings file not found: {input_path}")
            return {}
        
        with open(input_path, "r", encoding="utf-8") as f:
            query_map = json.load(f)
        
        LOG.info(f"Loaded query mappings from {input_path}")
        return query_map