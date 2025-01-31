from pydantic_ai.models import Model, KnownModelName
from know_lang_bot.core.types import ModelProvider
from know_lang_bot.models.huggingface import HuggingFaceModel
from typing import get_args

def create_pydantic_model(
    model_provider: ModelProvider,
    model_name: str,
) -> Model | KnownModelName:
    model_str = f"{model_provider}:{model_name}"

    if model_str in get_args(KnownModelName):
        return model_str
    elif model_provider == ModelProvider.HUGGINGFACE:
        return HuggingFaceModel(model_name=model_name)
    else:
        raise NotImplementedError(f"Model {model_provider}:{model_name} is not supported")
