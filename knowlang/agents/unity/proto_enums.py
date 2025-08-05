"""
Protobuf-generated enums for Unity UI generation.
This file should be generated from the .proto file using protoc.
For now, we'll define them manually to match the protobuf schema.
"""

from enum import IntEnum
from typing import Optional


class UIGenerationStatus(IntEnum):
    """Enum for tracking UI generation progress status - matches protobuf definition"""
    
    UNSPECIFIED = 0
    STARTING = 1
    GENERATING_UXML = 2
    GENERATING_USS = 3
    GENERATING_CSHARP = 4
    COMPLETE = 5
    ERROR = 6
    
    @classmethod
    def from_string(cls, status_str: str) -> "UIGenerationStatus":
        """Convert string status to enum value"""
        status_map = {
            "starting": cls.STARTING,
            "generating_uxml": cls.GENERATING_UXML,
            "generating_uss": cls.GENERATING_USS,
            "generating_csharp": cls.GENERATING_CSHARP,
            "complete": cls.COMPLETE,
            "error": cls.ERROR,
        }
        return status_map.get(status_str.lower(), cls.UNSPECIFIED)
    
    def to_string(self) -> str:
        """Convert enum value to string"""
        status_map = {
            self.STARTING: "starting",
            self.GENERATING_UXML: "generating_uxml",
            self.GENERATING_USS: "generating_uss",
            self.GENERATING_CSHARP: "generating_csharp",
            self.COMPLETE: "complete",
            self.ERROR: "error",
        }
        return status_map.get(self, "unspecified")


# Type aliases for better code readability
UIGenerationStatusProto = UIGenerationStatus 