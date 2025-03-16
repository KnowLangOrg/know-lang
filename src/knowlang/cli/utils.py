from pathlib import Path
from typing import Optional
from knowlang.configs.config import AppConfig

def create_config(config_path: Optional[Path] = None) -> AppConfig:
    """Create configuration from file or defaults."""
    if config_path:
        with open(config_path, 'r') as file:
            config_data = file.read()
            return AppConfig.model_validate_json(config_data)
    return AppConfig()