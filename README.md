# KnowLang: Comprehensive Understanding for Complex Codebase

KnowLang is an advanced codebase exploration tool that helps software engineers better understand complex codebases through semantic search and intelligent Q&A capabilities. Our first release focuses on providing RAG-powered search and Q&A for popular open-source libraries, with Hugging Face's repositories as our initial targets.

[![Official Website](https://img.shields.io/badge/🌐%20Official-Website-blue)](https://www.knowlang.dev)

> 🚀 **Try it yourself!** Want to see KnowLang in action? Visit our live demo at [www.knowlang.dev](https://www.knowlang.dev) and start exploring codebases today!

## Features

- 🔍 **Semantic Code Search**: Find relevant code snippets based on natural language queries
- 📚 **Contextual Q&A**: Get detailed explanations about code functionality and implementation details
- 🎯 **Smart Chunking**: Intelligent code parsing that preserves semantic meaning
- 🔄 **Two-Stage Retrieval**: Powerful multi-stage retrieval pipeline with keyword search, vector embedding search, and relevance reranking
- 🌐 **Multi-Language Support**: Support for Python, C++, TypeScript, with more languages on the roadmap
- 📈 **Incremental Updates**: Efficiently update your index when code changes without reprocessing the entire codebase

## Prerequisites

### LLM Provider

> Note: While Ollama is the default choice for easy setup, KnowLang supports other LLM providers through configuration. See our [Configuration Guide](configuration.md) for using alternative providers like OpenAI or Anthropic.
KnowLang uses [Ollama](https://ollama.com) as its default LLM and embedding provider. Before installing KnowLang:

1. Install Ollama:

```bash
# check the official download instructions from https://ollama.com/download
curl -fsSL https://ollama.com/install.sh | sh
```

2. Pull required models:

```bash
# For LLM responses
ollama pull llama3.2
```

3. Verify Ollama is running:

```bash
ollama list
```

You should see `llama3.2` in the list of available models.


### Database Setup

KnowLang uses PostgreSQL with pgvector extension for efficient vector storage and retrieval. You can easily set up the database using Docker:

1. Make sure you have Docker and Docker Compose installed:
   ```bash
   docker --version
   docker compose --version
   ```

2. Start the PostgreSQL database:
   ```bash
   # From the root of the know-lang repository
   docker compose -f docker/application/docker-compose.app.yml up -d
   ```

3. Verify the database is running:
   ```bash
   docker ps | grep pgvector
   ```

You should see the pgvector container running on port 5432.

> ⚠️ **Important**: The database must be running before you use any KnowLang commands like `parse` or `chat` that require database access.

## Quick Start

### Installation

You can install KnowLang via pip:

```bash
pip install knowlang
```

Alternatively, you can clone the repository and install it in editable mode:

```bash
git clone https://github.com/KnowLangOrg/know-lang.git
cd know-lang
pip install -e .

# if using Poetry
poetry install
poetry env activate
# poetry will output
# source path/activate <- run this command
source path_provided_by_poetry/activate
```

This allows you to make changes to the source code and have them immediately reflected without reinstalling the package.

### Basic Usage

1. Make sure the PostgreSQL database is running (see [Database Setup](#database-setup) above).

2. Parse and index your codebase:

```bash
# For a local codebase
knowlang parse ./my-project

# For verbose output
knowlang -v parse ./my-project
```

> ⚠️ Warning: Make sure to setup the correct paths to include and exclude for parsing. Please refer to "Parser Settings" section in [Configuration Guide](configuration.md) for more information

3. Launch the chat interface:

```bash
knowlang chat
```

That's it! The chat interface will open in your browser, ready to answer questions about your codebase.

![Chat Interface](chat.png)

### Advanced Usage

#### Custom Configuration

```bash
# Use custom configuration file
knowlang parse --config my_config.yaml ./my-project

# Output parsing results in JSON format
knowlang parse --output json ./my-project

# Incremental update of the codebase
knowlang parse --incremental ./my-project
```

#### Chat Interface Options

```bash
# Run on a specific port
knowlang chat --port 7860

# Create a shareable link
knowlang chat --share

# Run on custom server
knowlang chat --server-name localhost --server-port 8000
```

### Example Session

```bash
# Parse the transformers library
$ knowlang parse ./transformers
Found 1247 code chunks
Processing summaries... Done!

# Start chatting
$ knowlang chat

💡 Ask questions like:
- How is tokenization implemented?
- Explain the training pipeline
- Show me examples of custom model usage
```

## Architecture

KnowLang uses several key technologies:

- **Tree-sitter**: For robust, language-agnostic code parsing
- **PostgreSQL with pgvector**: For efficient vector storage and retrieval
- **PydanticAI**: For type-safe LLM interactions
- **Gradio**: For the interactive chat interface

## Technical Details

### Multi-Language Code Parsing

Our code parsing pipeline uses Tree-sitter to break down source code into meaningful chunks while preserving context:

1. Repository cloning and file identification
2. Language detection and routing to appropriate parsers (Python, C++, TypeScript)
3. Semantic parsing with Tree-sitter
4. Smart chunking based on language-specific AST structures
5. LLM-powered summarization
6. Embedding generation
7. Vector store indexing

### Incremental Updates

KnowLang supports efficient incremental updates to your code index:

1. Tracking file states (hash, modification time, chunk IDs)
2. Detecting changed files since last indexing
3. Only processing modified files rather than the entire codebase
4. Maintaining index consistency by removing outdated chunks
5. Adding new chunks for modified or added files

### Two-Stage Retrieval System

The RAG system uses a sophisticated multi-stage retrieval process:

1. **First Stage**: Recall relevant code chunks using:
   - Keyword-based search for exact matches
   - Vector embedding search for semantic similarity
   - Combined results from both approaches

2. **Second Stage**: Rerank results using:
   - GraphCodeBERT cross-encoder for more accurate relevance scoring
   - Filtering based on relevance threshold
   - Limited to top-K most relevant chunks

3. **Response Generation**:
   - Combine reranked chunks as context
   - Generate LLM response with the enhanced context

> ⚠️ Warning: the reranker is not yet fully implemented, hence reranking stage is disabled by default.

## MCP Integration

KnowLang supports the Model Context Protocol (MCP), which enables language models like Claude to directly access your codebase knowledge. This allows for more interactive and context-aware conversations about your code.

### What is MCP?

MCP (Model Context Protocol) is an open protocol that standardizes how applications provide context to LLMs. Think of it like a USB-C port for AI applications - it provides a standardized way to connect AI models to different data sources and tools.

### Using MCP with KnowLang

With MCP integration, you can connect KnowLang's search capabilities directly to Claude Desktop or other MCP-compatible applications.

For detailed instructions on setting up and using MCP with KnowLang, refer to the MCP README file in the source directory:
```
src/knowlang/mcp/README.md
```

## Roadmap
- [x] MCP support for LLM contexts
- [ ] Additional language support (Java, Ruby, Go, etc.)
- [ ] Inter-repository semantic search
- [ ] Automatic documentation maintenance
- [ ] Integration with popular IDEs
- [ ] Custom embedding model training
- [ ] Enhanced evaluation metrics

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. The Apache License 2.0 is a permissive license that enables broad use, modification, and distribution while providing patent rights and protecting trademark use.

## Citation

If you use KnowLang in your research, please cite:

```bibtex
@software{knowlang2025,
  author = KnowLang,
  title = {KnowLang: Comprehensive Understanding for Complex Codebase},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/KnowLangOrg/know-lang}
}
```

## Support

For support, please open an issue on GitHub or reach out to us directly through discussions. You can also visit our [official website](https://www.knowlang.dev) for more resources, documentation, and live demonstrations of KnowLang in action.

## Community

Wondering how KnowLang works in real-world scenarios? Curious about best practices? Join our growing community of developers at [www.knowlang.dev](https://www.knowlang.dev) to see examples, share your experiences, and learn from others.
