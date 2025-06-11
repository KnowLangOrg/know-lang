# LLM Configuration
# Model settings for the main language model
LLM__MODEL_NAME=llama3.2
LLM__MODEL_PROVIDER=ollama
LLM__API_KEY=your_api_key
LLM__MODEL_SETTINGS='{"base_url":"http://127.0.0.1:11434/v1"}'

# Evaluator Configuration
# Settings for the model evaluation
EVALUATOR__MODEL_NAME=llama3.2
EVALUATOR__MODEL_PROVIDER=ollama
EVALUATOR__API_KEY=your_api_key
EVALUATOR__EVALUATION_ROUNDS=1

# Reranker Configuration
# Settings for result reranking
RERANKER__ENABLED=false
RERANKER__MODEL_NAME=qwen3:30b
RERANKER__MODEL_PROVIDER=ollama
RERANKER__API_KEY=your_api_key
RERANKER__TOP_K=5
RERANKER__RELEVANCE_THRESHOLD=0.1

# Database Configuration
# ChromaDB and codebase settings
DB__PERSIST_DIRECTORY=./chromadb/mycode
DB__COLLECTION_NAME=code
DB__CODEBASE_DIRECTORY='.'
DB__DB_PROVIDER=postgres
DB__CONNECTION_URL=postgresql://postgres:postgres@localhost:5432/postgres
# State Store Configuration
DB__STATE_STORE__PROVIDER=postgres
DB__STATE_STORE__CONNECTION_URL=postgresql://postgres:postgres@localhost:5432/postgres

# Parser Configuration
# Settings for code parsing and file patterns
PARSER__LANGUAGES='{"python": {"enabled": true, "file_extensions": [".py"], "tree_sitter_language": "python", "chunk_types": ["class_definition", "function_definition"], "max_file_size": 1000000}, "cpp": {"enabled": true, "file_extensions": [".cpp", ".h", ".hpp", ".cc"], "tree_sitter_language": "cpp", "chunk_types": ["class_definition", "function_definition"], "max_file_size": 1000000}, "typescript": {"enabled": true, "file_extensions": [".ts", ".tsx"], "tree_sitter_language": "typescript", "chunk_types": ["class_definition", "function_definition"], "max_file_size": 1000000}, "csharp": {"enabled": true, "file_extensions": [".cs"], "tree_sitter_language": "csharp", "chunk_types": ["class_declaration", "method_declaration"], "max_file_size": 1000000}}'
PARSER__PATH_PATTERNS='{"include": ["**/*"], "exclude": ["**/venv/**", "**/.git/**", "**/__pycache__/**", "**/tests/**"]}'

# Chat Configuration
CHAT__LLM__MODEL_NAME=llama3.2
CHAT__LLM__MODEL_PROVIDER=ollama
CHAT__LLM__API_KEY=your_api_key

# Embedding Configuration
# Settings for text embedding generation
EMBEDDING__MODEL_NAME=nomic-ai/CodeRankEmbed
EMBEDDING__MODEL_PROVIDER=nomic-ai
EMBEDDING__API_KEY=your_api_key
EMBEDDING__DIMENSION=768 # default dimension of graphcodebert

# Chat Analytics Configuration
# Settings for analytics tracking
CHAT_ANALYTICS__ENABLED=false
CHAT_ANALYTICS__PROVIDER=mixpanel
CHAT_ANALYTICS__API_KEY=your_api_key

RETRIEVAL__VECTOR_SEARCH__TOP_K=10