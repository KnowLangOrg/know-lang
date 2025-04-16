from pydantic_ai.models import Model, KnownModelName
from knowlang.core.types import ModelProvider
from typing import get_args

def create_pydantic_model(
    model_provider: ModelProvider,
    model_name: str,
) -> Model | KnownModelName:
    model_str = f"{model_provider}:{model_name}"

    if model_str in get_args(KnownModelName):
        return model_str
    elif model_provider == ModelProvider.TESTING:
        # should be used for testing purposes only
        pass
    elif model_provider == ModelProvider.OLLAMA:
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider
        return OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        )
    else:
        raise NotImplementedError(f"Model {model_provider}:{model_name} is not supported")
