# Domain Registry 
Each Domain can have assets, each asset can have several chunks.
Domain can be registered by adding yaml files in settings/assets/*.yml, example one can be found in (codebase.yml)


## Domain, Asset, Chunks
No matter of which domain, assets and chunks, the shared data structures are defined in `knowlang/assets/models.py`.
The domain speicifc information can be defined and stored into the metadata field to flexibily store more information.

# Asset Domain Registry

Configuration-driven registry system for managing heterogeneous assets across different domains (codebase, Unity, Unreal, etc.). Uses YAML configuration files to automatically instantiate and orchestrate domain processors.

## Architecture

```
Domain → Assets → Chunks
```
The entire processing flow is as below.

![Domain,Asset,Registry](knowlang/assets/DomainAssetChunk.jpg)

Each domain contains multiple assets, and each asset can be parsed into multiple chunks. The system provides a unified interface for:
- **Sourcing**: Discovery and enumeration of assets
- **Parsing**: Breaking assets into meaningful chunks  
- **Indexing**: Vector embedding and storage

## Core Components

### Models ([`models.py`](knowlang/assets/models.py))
- `DomainManagerData[MetaDataT]`: Domain configuration and metadata
- `GenericAssetData[MetaDataT]`: Individual asset within a domain
- `GenericAssetChunkData[MetaDataT]`: Parsed chunks from assets

### Processor Framework ([`processor.py`](knowlang/assets/processor.py))
- `DomainAssetSourceMixin`: Asset discovery and enumeration
- `DomainAssetParserMixin`: Asset-to-chunk parsing
- `DomainAssetIndexingMixin`: Vector embedding and storage

### Registry ([`registry.py`](knowlang/assets/registry.py))
- `DomainRegistry`: Main orchestrator that discovers YAML configs
- `TypeRegistry`: Maps domain types to their metadata classes
- `MixinRegistry`: Maps string identifiers to processor classes

## Configuration

### Domain Configuration (YAML)
see example in [codebase.yml](settings/assets/codebase.yml)

### Registry Configuration (`settings/registry.yaml`)
see example in [registry.yaml](settings/registry.yaml)

## Domain Implementation

### Codebase Domain
Located in `codebase/`:

**Models**:
- `CodebaseMetaData`: Git repository information
- `CodeAssetMetaData`: File path metadata
- `CodeAssetChunkMetaData`: Code chunk with location and content

**Processors**:
- `CodebaseAssetSource`: Walks directory, respects .gitignore
- `CodebaseAssetParser`: Uses tree-sitter for code parsing
- `CodebaseAssetIndexing`: Generates embeddings and stores in vector database

## Database Integration

### SQL Schema (`database/db.py`)
- **domains**: Domain manager configurations
- **assets**: Individual assets with file hashes for change detection
- **asset_chunks**: Parsed chunks linked to assets

### Vector Store (`database/config.py`)
Configurable vector storage for chunk embeddings:
- SQLite (default)
- ChromaDB
- Other providers via `VectorStoreProvider`

## Usage

### Processing All Domains
```python
from knowlang.assets.registry import DomainRegistry, RegistryConfig

config = RegistryConfig()
registry = DomainRegistry(config)
await registry.discover_and_register()
await registry.process_all_domains()
```

### Command Line
```bash
knowlang parse
```
or


## Features

- **Incremental Processing**: Only processes changed files using hash comparison
- **Batch Processing**: Configurable batch sizes for efficient database operations  
- **Cleanup Handling**: Automatically removes deleted assets and chunks
- **Type Safety**: Full generic typing with covariant/contravariant type variables
- **Configuration Validation**: pydantic-based validation for all YAML configs
- **Extensible**: Add new domains by implementing three mixins and a YAML config

## Adding New Domains

1. Create domain-specific models to make the metadata type concrete
2. Implement the three processor mixins
3. Register types and mixins in registry
4. Add YAML configuration file
5. The registry automatically discovers and loads the domain

## 

# TODOs
- Domain registry ymls in settings/assets should also be properly git ignored
- deprecate followings
    - AppConfig
    - All logic under knowlang/indexing folder, like, StateStore (which is replaced by `KnowledegSqlDatabase`)
- Testing for the Domain Registry Parsing
- `knowlang chat` is still configured in the old way to use the database specified in settings/.env.app, I guess we need to extract the ChatConfig from the AppConfig and use yml for chat interface setting management.