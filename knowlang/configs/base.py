import os
import sys
from pathlib import Path
from typing import Optional

from pydantic import ValidationInfo
from pydantic_settings import SettingsConfigDict


def get_resource_path(relative_path: str, default_path: Optional[str] = None) -> Path:
    """Get absolute path to resource, works for dev and PyInstaller

    Args:
        relative_path: The relative path to the resource
        default_path: Optional alternative relative path to use if the primary path doesn't exist

    Returns:
        Path: The absolute path to the resource (may not exist)
    """

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # Running in PyInstaller bundle
        base_path = Path(sys._MEIPASS)
    else:
        # Running in non-PyInstaller environment
        base_path = Path.cwd()

    # Define paths to try: primary path first, then default path if provided
    paths_to_try = [relative_path]
    if default_path:
        paths_to_try.append(default_path)

    for path_to_try in paths_to_try:
        full_path = base_path / path_to_try

        if full_path.exists():
            return full_path

    # If we get here, none of the paths exist
    attempted_paths = [str(Path.cwd() / path) for path in paths_to_try]
    raise FileNotFoundError(
        f"Resource not found. Attempted paths: {attempted_paths}"
    )


def generate_model_config(
    env_dir: Path = Path("settings"), 
    env_file: Path = ".env", 
    env_prefix: str = "",
    default_env_file: Optional[Path] = None
) -> SettingsConfigDict:
    # Use PyInstaller-aware path resolution with optional default path
    env_file_path = get_resource_path(
        str(env_dir / env_file), 
        default_path=str(env_dir / default_env_file) if default_env_file else None
    )

    return SettingsConfigDict(
        env_file=str(env_file_path),
        env_prefix=env_prefix,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )


def _validate_api_key(v: Optional[str], info: ValidationInfo) -> Optional[str]:
    """Validate API key is present when required"""
    from knowlang.core.types import ModelProvider

    if info.data["model_provider"] in [
        ModelProvider.OPENAI,
        ModelProvider.ANTHROPIC,
        ModelProvider.VOYAGE,
        ModelProvider.GOOGLE,
    ]:
        if not v:
            raise ValueError(f"API key required for {info.data['model_provider']}")
        elif info.data["model_provider"] == ModelProvider.ANTHROPIC:
            os.environ["ANTHROPIC_API_KEY"] = v
        elif info.data["model_provider"] == ModelProvider.OPENAI:
            os.environ["OPENAI_API_KEY"] = v
        elif info.data["model_provider"] == ModelProvider.VOYAGE:
            os.environ["VOYAGE_API_KEY"] = v
        elif info.data["model_provider"] == ModelProvider.GOOGLE:
            os.environ["GEMINI_API_KEY"] = v

    return v
