# Configuration Guide

KnowLang uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration management. Settings can be provided through environment variables, `.env` files, JSON configuration files, or programmatically.

## Quick Start

1. Copy the example configuration files:
```bash
cp settings/.env.example.app settings/.env.app
cp settings/app_config.example.json settings/app_config.json
```

2. Modify settings as needed in `.env.app` or `app_config.json`

## Configuration Methods

KnowLang supports multiple configuration methods:

1. **Environment Variables**: Set directly in your environment
2. **`.env` Files**: Settings in `.env.app` file
3. **JSON Configuration**: Settings in `app_config.json` file 
4. **Programmatic**: Create config objects in code

Environment variables take precedence over .env files, which take precedence over JSON configuration.

## Core Settings

### LLM Settings
```env
# Default is Ollama with llama3.2
LLM__MODEL_NAME=llama3.2
LLM__MODEL_PROVIDER=ollama
LLM__API_KEY=your_api_key  # Required for providers like OpenAI
LLM__MODEL_SETTINGS='{"base_url":"http://127.0.0.1:11434/v1"}'
```

Supported providers:
- `ollama`: Local models through Ollama
- `openai`: OpenAI models (requires API key)
- `anthropic`: Anthropic models (requires API key)

### Embedding Settings
```env
# Default uses the nomic-ai CodeRankEmbed model
EMBEDDING__MODEL_NAME=nomic-ai/CodeRankEmbed
EMBEDDING__MODEL_PROVIDER=nomic-ai
EMBEDDING__API_KEY=your_api_key  # Required for certain providers
EMBEDDING__DIMENSION=768  # Default dimension of graphcodebert
```

### Database Settings
```env
# Database configuration
DB__PERSIST_DIRECTORY=./chromadb/mycode
DB__COLLECTION_NAME=code
DB__CODEBASE_DIRECTORY=.
DB__DB_PROVIDER=postgres
DB__CONNECTION_URL=postgresql://postgres:postgres@localhost:5432/postgres

# State Store Configuration
DB__STATE_STORE__PROVIDER=postgres
DB__STATE_STORE__CONNECTION_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

### Parser Settings
```env
# Language support and file patterns
PARSER__LANGUAGES='{"python": {"enabled": true, "file_extensions": [".py"], "tree_sitter_language": "python", "chunk_types": ["class_definition", "function_definition"], "max_file_size": 1000000}, "cpp": {"enabled": true, "file_extensions": [".cpp", ".h", ".hpp", ".cc"], "tree_sitter_language": "cpp", "chunk_types": ["class_definition", "function_definition"], "max_file_size": 1000000}, "typescript": {"enabled": true, "file_extensions": [".ts", ".tsx"], "tree_sitter_language": "typescript", "chunk_types": ["class_definition", "function_definition"], "max_file_size": 1000000}}'

PARSER__PATH_PATTERNS='{"include": ["**/*"], "exclude": ["**/venv/**", "**/.git/**", "**/__pycache__/**", "**/tests/**"]}'
```

### Two-Stage Retrieval Configuration
```env
# Retrieval Configuration
RETRIEVAL__ENABLED=true

# Vector Search Configuration
RETRIEVAL__VECTOR_SEARCH__ENABLED=true
RETRIEVAL__VECTOR_SEARCH__TOP_K=10
RETRIEVAL__VECTOR_SEARCH__SCORE_THRESHOLD=0.0
RETRIEVAL__VECTOR_SEARCH__MAX_RETRIES=5

# Keyword Search Configuration
RETRIEVAL__KEYWORD_SEARCH__ENABLED=true
RETRIEVAL__KEYWORD_SEARCH__TOP_K=10
RETRIEVAL__KEYWORD_SEARCH__SCORE_THRESHOLD=0.0
RETRIEVAL__KEYWORD_SEARCH__MAX_RETRIES=5
```

In the JSON configuration, this would look like:

```json
"retrieval": {
  "enabled": true,
  "keyword_search": {
    "enabled": true,
    "top_k": 10,
    "score_threshold": 0.0,
    "max_retries": 5
  },
  "vector_search": {
    "enabled": true,
    "top_k": 10,
    "score_threshold": 0.0,
    "max_retries": 5
  }
}
```

### Reranker Configuration
```env
# Reranker Configuration
RERANKER__ENABLED=true
RERANKER__MODEL_NAME=KnowLang/RerankerCodeBERT
RERANKER__MODEL_PROVIDER=graph_code_bert
RERANKER__API_KEY=your_api_key
RERANKER__TOP_K=5
RERANKER__RELEVANCE_THRESHOLD=0.1
```

The `graph_code_bert` provider implements the GraphCodeBERT model for more effective code search reranking.

## Advanced Configuration

### Using Multiple Models

You can configure different models for different purposes:
```env
# Main LLM for responses
LLM__MODEL_NAME=llama3.2
LLM__MODEL_PROVIDER=ollama

# Evaluation model
EVALUATOR__MODEL_NAME=llama3.2
EVALUATOR__MODEL_PROVIDER=ollama
EVALUATOR__EVALUATION_ROUNDS=1

# Chat-specific LLM settings
CHAT__LLM__MODEL_NAME=llama3.2
CHAT__LLM__MODEL_PROVIDER=ollama
```

### Analytics Integration
```env
CHAT_ANALYTICS__ENABLED=false
CHAT_ANALYTICS__PROVIDER=mixpanel
CHAT_ANALYTICS__API_KEY=your_api_key
```

## JSON Configuration Format

For detailed configuration, you can use the JSON format in `app_config.json`. Here's a simplified example:

```json
{
  "llm": {
    "model_name": "llama3.2",
    "model_provider": "ollama",
    "api_key": "your_api_key",
    "model_settings": {
      "base_url": "http://127.0.0.1:11434/v1"
    }
  },
  "db": {
    "db_provider": "postgres",
    "connection_url": "postgresql://postgres:postgres@localhost:5432/postgres",
    "persist_directory": "./chromadb/mycode",
    "collection_name": "code",
    "codebase_directory": ".",
    "state_store": {
      "provider": "postgres",
      "connection_url": "postgresql://postgres:postgres@localhost:5432/postgres"
    }
  },
  "parser": {
    "languages": {
      "python": {
        "enabled": true,
        "file_extensions": [".py"],
        "tree_sitter_language": "python",
        "chunk_types": ["class_definition", "function_definition"]
      }
    },
    "path_patterns": {
      "include": ["**/*"],
      "exclude": ["**/venv/**", "**/.git/**", "**/__pycache__/**", "**/tests/**"]
    }
  },
  "retrieval": {
    "enabled": true,
    "keyword_search": {
      "enabled": true,
      "top_k": 10
    },
    "vector_search": {
      "enabled": true,
      "top_k": 10
    }
  }
}
```

## Further Reading

- For detailed settings configuration options, see [pydantic-settings documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- For model-specific configuration, see provider documentation:
  - [Ollama Models](https://ollama.ai/library)
  - [OpenAI Models](https://platform.openai.com/docs/models)
  - [Anthropic Models](https://www.anthropic.com/models)
