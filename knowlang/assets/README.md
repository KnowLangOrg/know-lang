# Domain Registry

Each domain can have assets, and each asset can have several chunks.
Domains can be registered by adding YAML files in settings/assets/\*.yml. An example can be found in (codebase.example.yml).

## Domain, Asset, Chunks

Regardless of the domain, assets, and chunks, the shared data structures are defined in `knowlang/assets/models.py`.
The domain-specific information can be defined and stored in the metadata field to flexibly store additional information.

# Asset Domain Registry

Configuration-driven registry system for managing heterogeneous assets across different domains (codebase, Unity, Unreal, etc.). Uses YAML configuration files to automatically instantiate and orchestrate domain processors.

## Architecture

```
Domain → Assets → Chunks
```

The entire processing flow is shown below.

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

See example in [codebase.yml](../../settings/assets/codebase.example.yaml)

### Registry Configuration (YAML)

See example in [registry.yaml](../../settings/registry.example.yaml)

## Domain Implementation

### Codebase Domain

Located in `codebase/`:

**Models**:

- `CodebaseMetaData`: Git repository information
- `CodeAssetMetaData`: File path metadata
- `CodeAssetChunkMetaData`: Code chunk with location and content

**Processors**:

- `CodebaseAssetSource`: Walks directory and respects .gitignore
- `CodebaseAssetParser`: Uses tree-sitter for code parsing
- `CodebaseAssetIndexing`: Generates embeddings and stores them in vector database

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

> ⚠️ Note that the `VectorStoreConfig.table_name` should be set differently for each domain in the corresponding YAML file so that the vector store for each domain is properly separated.

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

- Testing for the Domain Registry Parsing
