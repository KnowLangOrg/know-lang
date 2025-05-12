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


@register_provider(ModelProvider.NOMIC_AI)
def _process_nomic_sentence_batch(inputs: List[str], model_name: str, input_type: Optional[EmbeddingInputType] = None) -> List[EmbeddingVector]:
    from knowlang.models.nomic_ai.model import generate_embeddings
    return generate_embeddings(inputs, model_name=model_name, input_type=input_type)

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
    return generate_embeddings(inputs, input_type=input_type, model_name=model_name)

@register_provider(ModelProvider.OLLAMA)
def _process_ollama_batch(inputs: List[str], model_name: str, _: Optional[EmbeddingInputType] = None) -> List[EmbeddingVector]:
    try:
        import ollama
    except ImportError as e:
        raise ImportError(
            'Ollama is not installed. Please install it using `pip install "knowlang[ollama]"`.'
        ) from e

    return ollama.embed(model=model_name, input=inputs)['embeddings']

@register_provider(ModelProvider.OPENAI)
def _process_openai_batch(inputs: List[str], model_name: str, _: Optional[EmbeddingInputType] = None) -> List[EmbeddingVector]:
    import openai
    response = openai.embeddings.create(input=inputs, model=model_name)
    return [item.embedding for item in response.data]

@register_provider(ModelProvider.VOYAGE)
def _process_voyage_batch(inputs: List[str], model_name: str, input_type: Optional[EmbeddingInputType]) -> List[EmbeddingVector]:
    try:
        import voyageai
    except ImportError as e:
        raise ImportError(
            'VoyageAI is not installed. Please install it using `pip install "knowlang[voyage]"`.'
        ) from e
    client = voyageai.Client()
    embeddings_obj = client.embed(model=model_name, texts=inputs, input_type=input_type.value)
    return embeddings_obj.embeddings