from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel
from knowlang.configs.chat_config import ChatConfig


class UIGenerationResult(BaseModel):
    """Final result from the UI generation graph"""

    uxml_content: str
    uss_content: str
    csharp_content: str
    ui_description: str
    filename: str


@dataclass
class UIGenerationState:
    """State maintained throughout the UI generation graph execution"""

    ui_description: str
    filename: Optional[str] = None
    uxml_content: Optional[str] = None
    uss_content: Optional[str] = None
    csharp_content: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class UIGenerationDeps:
    """Dependencies required by the UI generation graph"""

    chat_config: ChatConfig = None
