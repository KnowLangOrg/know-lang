from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from enum import Enum
from knowlang.configs.llm_config import LLMConfig

class ChatConfig(BaseSettings):
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
    )
    interface_title: str = Field(
        default="KonwLang Codebase Assistant",
        description="Title shown in the chat interface"
    )
    interface_description: str = Field(
        default="Ask questions about the codebase and I'll help you understand it!",
        description="Description shown in the chat interface"
    )
    interface_placeholder: str = Field(
        default="Ask about the codebase",
        description="Placeholder text in the chat interface"
    )

class AnalyticsProvider(str, Enum):
    MIXPANEL = "mixpanel"

class ChatbotAnalyticsConfig(BaseSettings):
    enabled: bool = Field(
        default=False,
        description="Enable analytics tracking"
    )
    provider: AnalyticsProvider = Field(
        default=AnalyticsProvider.MIXPANEL,
        description="Analytics provider to use for tracking feedback"
    )

    api_key: Optional[str] = Field(
        default=None,
        description="api key for feedback tracking"
    )