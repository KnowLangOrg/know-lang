from pydantic_ai.models import Model, KnownModelName
from knowlang.core.types import ModelProvider
from typing import get_args
from knowlang.configs.config import LLMConfig

def create_pydantic_model(
    config: LLMConfig,
) -> Model | KnownModelName:
    model_provider = config.model_provider
    model_name = config.model_name
    model_str = f"{model_provider}:{model_name}"

    known_model_names = get_args(KnownModelName.__value__)

    # raise error if the KnownModelName implementation in pydantic_ai has changed
    if not known_model_names:
        raise ValueError("KnownModelName has no known model names")
    
    if model_str in known_model_names:
        return model_str
    elif model_provider == ModelProvider.TESTING:
        from knowlang.utils.offline_model import OfflineModel
        return OfflineModel()
    elif model_provider == ModelProvider.OLLAMA:
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider
        return OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        )
    elif model_provider == ModelProvider.KNOWLANG:
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider
        base_url = config.model_settings.get("base_url", None)
        if not base_url:
            raise ValueError(
                "KnowLang model provider does not support base_url in model settings."
            )

        return OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url=base_url),
        )

    else:
        raise NotImplementedError(f"Model {model_provider}:{model_name} is not supported")
