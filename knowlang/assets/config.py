from pydantic import BaseModel
from knowlang.assets.models import DomainManagerData

class DomainMixinConfig(BaseModel):
    source_cls: str             # Class identifier (e.g., "CodebaseAssetSource")
    indexer_cls: str            # Class identifier
    parser_cls: str             # Class identifier

class BaseDomainConfig(BaseModel):
    domain_type: str
    domain_id: str
    enabled: bool = True
    manager_data: DomainManagerData
    mixins: DomainMixinConfig
    