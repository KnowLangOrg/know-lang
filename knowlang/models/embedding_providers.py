from typing import Awaitable, Callable, Dict, List, Union

from knowlang.core.types import ModelProvider

from .types import EmbeddingVector, EmbeddingParams

# Global registry for provider functions
EMBEDDING_PROVIDER_REGISTRY: Dict[
    ModelProvider,
    Union[
        # Synchronous function type
        Callable[[EmbeddingParams], List[EmbeddingVector]],
        # Asynchronous function type
        Callable[
            [EmbeddingParams],
            Awaitable[List[EmbeddingVector]],
        ],
    ],
] = {}


def register_provider(provider: ModelProvider):
    """Decorator to register a provider function."""

    def decorator(
        func: Callable[[EmbeddingParams], List[EmbeddingVector]],
    ):
        EMBEDDING_PROVIDER_REGISTRY[provider] = func
        return func

    return decorator


@register_provider(ModelProvider.NOMIC_AI)
def _process_nomic_sentence_batch(params: EmbeddingParams) -> List[EmbeddingVector]:
    from knowlang.models.nomic_ai.model import generate_embeddings

    inputs = params.inputs
    model_name = params.cfg.model_name
    input_type = params.input_type

    return generate_embeddings(inputs, model_name=model_name, input_type=input_type)


@register_provider(ModelProvider.OLLAMA)
def _process_ollama_batch(params: EmbeddingParams) -> List[EmbeddingVector]:
    inputs = params.inputs
    model_name = params.cfg.model_name

    try:
        import ollama
    except ImportError as e:
        raise ImportError(
            'Ollama is not installed. Please install it using `pip install "knowlang[ollama]"`.'
        ) from e

    return ollama.embed(model=model_name, input=inputs)["embeddings"]


@register_provider(ModelProvider.OPENAI)
def _process_openai_batch(params: EmbeddingParams) -> List[EmbeddingVector]:
    import openai

    inputs = params.inputs
    model_name = params.cfg.model_name

    response = openai.embeddings.create(input=inputs, model=model_name)
    return [item.embedding for item in response.data]


@register_provider(ModelProvider.VOYAGE)
def _process_voyage_batch(params: EmbeddingParams) -> List[EmbeddingVector]:
    inputs = params.inputs
    model_name = params.cfg.model_name
    input_type = params.input_type

    try:
        import voyageai
    except ImportError as e:
        raise ImportError(
            'VoyageAI is not installed. Please install it using `pip install "knowlang[voyage]"`.'
        ) from e
    client = voyageai.Client()
    embeddings_obj = client.embed(
        model=model_name, texts=inputs, input_type=input_type.value
    )
    return embeddings_obj.embeddings


@register_provider(ModelProvider.KNOWLANG)
async def _process_knowlang_batch(params: EmbeddingParams) -> List[EmbeddingVector]:
    """
    Generate embeddings using the KnowLang embedding model.
    Args:
        inputs: List of text inputs to embed
        model_name: Model identifier (not used in this case)
        input_type: Type of input (document/query/code)
    Returns:
        List of embedding vectors
    """
    import aiohttp

    inputs = params.inputs
    model_name = params.cfg.model_name
    input_type = params.input_type
    base_url = params.cfg.settings.get("base_url", None)
    if not base_url:
        raise ValueError("Base URL must be provided in the configuration settings.")

    embeddings = []
    async with aiohttp.ClientSession() as session:
        for item in inputs:
            async with session.post(
                f"{base_url}",
                json={
                    "model": model_name,
                    "input": item,
                    "input_type": input_type.value if input_type else "document",
                },
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get embeddings: {response.status}")
                data = await response.json()
                embeddings += [e["embedding"] for e in data["data"]]

    return embeddings
