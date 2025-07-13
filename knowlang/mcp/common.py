from abc import abstractmethod

from knowlang.database.config import VectorStoreConfig
from knowlang.utils import FancyLogger

LOG = FancyLogger(__name__)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class KnowLangTool:
    @classmethod
    @abstractmethod
    def initialize(config: VectorStoreConfig) -> "KnowLangTool":
        pass
