from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import BaseModel
from knowlang.configs.chat_config import ChatConfig


class UIGenerationResult(BaseModel):
    """Final result from the UI generation graph"""
    
    uxml_content: str
    uss_content: str
    csharp_content: str
    ui_description: str


@dataclass
class UIGenerationState:
    """State maintained throughout the UI generation graph execution"""
    
    ui_description: str
    uxml_content: Optional[str] = None
    uss_content: Optional[str] = None
    csharp_content: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class UIGenerationDeps:
    """Dependencies required by the UI generation graph"""
    
    chat_config: ChatConfig = None
    unity_project_path: Optional[str] = None
    ui_style_preferences: Optional[dict] = None 