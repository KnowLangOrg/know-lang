from typing import List, Optional
from knowlang.models.types import EmbeddingInputType, EmbeddingVector
from knowlang.utils import FancyLogger


LOG = FancyLogger(__name__)

# Cache for model and tokenizer
_NOMIC_CACHE = {}

def generate_embeddings(inputs: List[str], model_name: str, input_type: Optional[EmbeddingInputType] = None) -> List[EmbeddingVector]:
    """
    Generate embeddings using Nomic Embed Code model with SentenceTransformers.
    
    Args:
        inputs: List of text or code inputs to embed
        model_name: Model identifier (e.g., "nomic-ai/nomic-embed-code")
        input_type: Type of input (document/query/code)
    
    Returns:
        List of embedding vectors
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            'SentenceTransformers is not installed. Please install it using `pip install "knowlang[nomic-ai]"`.'
        ) from e
    
    # Get or create SentenceTransformer model
    if model_name not in _NOMIC_CACHE:
        try:
            import os
            # Disable parallelism to avoid issues with tokenizers when using asyncio (asyncio spawns multiple threads by default)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            model = SentenceTransformer(model_name)
        except Exception as e:
            LOG.warning(f"Failed to load model {model_name}: {e}, try loading with trust_remote_code=True")
            model = SentenceTransformer(model_name, trust_remote_code=True)
        
        _NOMIC_CACHE[model_name] = model
    else:
        model = _NOMIC_CACHE[model_name]
    
    # Determine if inputs should be treated as queries
    is_query = input_type == EmbeddingInputType.QUERY
    
    # Generate embeddings
    if is_query:
        # Use the prompt_name="query" parameter for queries
        embeddings = model.encode(inputs, prompt_name="query", convert_to_numpy=True, normalize_embeddings=True)
    else:
        # Regular encoding for code/document
        embeddings = model.encode(inputs, convert_to_numpy=True, normalize_embeddings=True)
    
    # Convert numpy arrays to lists
    embeddings_list = embeddings.tolist()
    
    return embeddings_list