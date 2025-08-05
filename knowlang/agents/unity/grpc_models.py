from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel


class UIGenerationRequest(BaseModel):
    """Request model for UI generation via gRPC"""
    
    ui_description: str
    unity_project_path: Optional[str] = None
    ui_style_preferences: Optional[dict] = None
    chat_config_override: Optional[dict] = None


class UIGenerationResponse(BaseModel):
    """Response model for UI generation via gRPC"""
    
    success: bool
    uxml_content: Optional[str] = None
    uss_content: Optional[str] = None
    csharp_content: Optional[str] = None
    ui_description: str
    status: str
    progress_message: str
    error_message: Optional[str] = None


class UIGenerationStreamResponse(BaseModel):
    """Streaming response model for UI generation via gRPC"""
    
    uxml_content: Optional[str] = None
    uss_content: Optional[str] = None
    csharp_content: Optional[str] = None
    ui_description: str
    status: str
    progress_message: str
    error_message: Optional[str] = None
    is_complete: bool = False


class UIGenerationStatusResponse(BaseModel):
    """Status response model for checking UI generation progress"""
    
    request_id: str
    status: str
    progress_message: str
    is_complete: bool
    error_message: Optional[str] = None


class UIGenerationCancelRequest(BaseModel):
    """Request model for canceling UI generation"""
    
    request_id: str


class UIGenerationCancelResponse(BaseModel):
    """Response model for canceling UI generation"""
    
    success: bool
    message: str 