from knowlang.configs.config import AppConfig
from knowlang.utils import FancyLogger
from abc import abstractmethod

LOG = FancyLogger(__name__)

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class KnowLangTool():
    @classmethod
    @abstractmethod
    def initialize(config: AppConfig) -> "KnowLangTool":
        pass
