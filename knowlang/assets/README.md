# Domain Registry 
Each Domain can have assets, each asset can have several chunks.
Domain can be registered by adding yaml files in settings/assets/*.yml, example one can be found in (codebase.yml)

The entire processing flow is as below.

![Domain, Asset, Registry](knowlang/assets/DomainAssetChunk.jpg)

## Domain, Asset, Chunks
No matter of which domain, assets and chunks, the shared data structures are defined in `knowlang/assets/models.py`.
The domain speicifc information can be defined and stored into the metadata field to flexibily store more information.


## 

# TODOs
- Domain registry ymls in settings/assets should also be properly git ignored
- deprecate followings
    - AppConfig
    - All logic under knowlang/indexing folder, like, StateStore (which is replaced by `KnowledegSqlDatabase`)
- Testing for the Domain Registry Parsing
- `knowlang chat` is still configured in the old way to use the database specified in settings/.env.app, I guess we need to extract the ChatConfig from the AppConfig and use yml for chat interface setting management.