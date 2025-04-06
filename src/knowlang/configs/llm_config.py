from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings
from typing import Any, Dict, Optional

from knowlang.core.types import ModelProvider
from knowlang.configs.base import _validate_api_key

class LLMConfig(BaseSettings):
    model_name: str = Field(
        default="llama3.2",
        description="Name of the LLM model to use"
    )
    model_provider: str = Field(
        default=ModelProvider.OLLAMA,
        description="Model provider (anthropic, openai, ollama, etc)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the model provider"
    )
    model_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model settings"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt for the model"
    )

    @field_validator('api_key', mode='after')
    @classmethod
    def validate_api_key(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        return _validate_api_key(v, info)

