from enum import Enum

class DatasetType(str, Enum):
    """Supported benchmark datasets."""
    CODESEARCHNET = "codesearchnet"
    COSQA = "cosqa"

class DatasetSplitType(str, Enum):
    """Dataset split types."""
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"