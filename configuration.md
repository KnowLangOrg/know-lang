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

Please refer to [database_setup.md](database_setup.md).

### Parser Settings

```env
# Language support and file patterns
PARSER__LANGUAGES='{"python": {"enabled": true, "file_extensions": [".py"], "tree_sitter_language": "python", "chunk_types": ["class_definition", "function_definition"], "max_file_size": 1000000}, "cpp": {"enabled": true, "file_extensions": [".cpp", ".h", ".hpp", ".cc"], "tree_sitter_language": "cpp", "chunk_types": ["class_definition", "function_definition"], "max_file_size": 1000000}, "typescript": {"enabled": true, "file_extensions": [".ts", ".tsx"], "tree_sitter_language": "typescript", "chunk_types": ["class_definition", "function_definition"], "max_file_size": 1000000}}'

PARSER__PATH_PATTERNS='{"include": ["**/*"], "exclude": ["**/venv/**", "**/.git/**", "**/__pycache__/**", "**/tests/**"]}'
```

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
      "exclude": [
        "**/venv/**",
        "**/.git/**",
        "**/__pycache__/**",
        "**/tests/**"
      ]
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
