from enum import Enum

class DatasetType(str, Enum):
    """Supported benchmark datasets."""
    CODESEARCHNET = "codesearchnet"
    COSQA = "cosqa"