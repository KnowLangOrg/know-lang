from typing import Any, Callable, Dict, List, Optional 
from knowlang.configs import ModelProvider

from .types import EmbeddingInputType, EmbeddingVector

# Global registry for provider functions
EMBEDDING_PROVIDER_REGISTRY: Dict[ModelProvider, Callable[[List[str], str, Optional[EmbeddingInputType]], List[EmbeddingVector]]] = {}

def register_provider(provider: ModelProvider):
    """Decorator to register a provider function."""
    def decorator(func: Callable[[List[str], str, Optional[EmbeddingInputType]], List[EmbeddingVector]]):
        EMBEDDING_PROVIDER_REGISTRY[provider] = func
        return func
    return decorator


@register_provider(ModelProvider.GRAPH_CODE_BERT)
def _process_graph_code_bert_batch(
    inputs: List[str], 
    model_name: str, 
    input_type: Optional[EmbeddingInputType] = None,
) -> List[EmbeddingVector]:
    """
    Generate embeddings using GraphCodeBERT.
    
    Args:
        inputs: List of text inputs to embed
        model_name: Model identifier
        input_type: Type of input (document/query/code)
    
    Returns:
        List of embedding vectors
    """
    from knowlang.models.graph_code_bert import generate_embeddings
    
    # Generate embeddings
    return generate_embeddings(inputs)

@register_provider(ModelProvider.OLLAMA)
def _process_ollama_batch(inputs: List[str], model_name: str, _: Optional[EmbeddingInputType] = None) -> List[EmbeddingVector]:
    import ollama
    return ollama.embed(model=model_name, input=inputs)['embeddings']

@register_provider(ModelProvider.OPENAI)
def _process_openai_batch(inputs: List[str], model_name: str, _: Optional[EmbeddingInputType] = None) -> List[EmbeddingVector]:
    import openai
    response = openai.embeddings.create(input=inputs, model=model_name)
    return [item.embedding for item in response.data]

@register_provider(ModelProvider.VOYAGE)
def _process_voyage_batch(inputs: List[str], model_name: str, input_type: Optional[EmbeddingInputType]) -> List[EmbeddingVector]:
    import voyageai
    client = voyageai.Client()
    embeddings_obj = client.embed(model=model_name, texts=inputs, input_type=input_type.value)
    return embeddings_obj.embeddings