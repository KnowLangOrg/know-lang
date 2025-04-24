# Model Context Protocol (MCP) for Knowlang

This package provides an implementation of the Model Context Protocol (MCP) for Knowlang's search capabilities. It allows language models like Claude to access Knowlang's keyword and vector search features.

## Overview

The Model Context Protocol (MCP) standardizes how applications provide context to LLMs. This implementation exposes Knowlang's search capabilities as MCP tools that can be used by Claude or other MCP clients.

## Features

- **Keyword Search Tool**: Performs keyword-based search on your knowledge base
- **Vector Search Tool**: Performs semantic search using vector embeddings


## Usage

### Starting the MCP Server

You can start the MCP server using the Knowlang CLI:

```bash
knowlang mcp serve
```

By default, the server listens on `localhost:7773`. You can customize this with arguments:

```bash
knowlang mcp serve --host 0.0.0.0 --port 8080 --name my-knowlang-server
```

### Using with Claude for Desktop

To use the MCP server with Claude for Desktop:

1. Make sure you have Claude for Desktop installed and updated to the latest version
2. Copy the content in `claude_desktop_config.json` file to your Claude Desktop configuration and update the path to the knowlang directory as needed:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`
3. **Important**: Ensure the following is set in your knowlang installation's `settings/.env.logging` file:
   ```
   STDIO_ENABLED=False
   ```
   This is required because Claude Desktop only accepts stdio MCP servers, and the stdio MCP server should not emit messages other than MCP communication.
4. Restart Claude for Desktop
5. Look for the hammer icon in Claude for Desktop to confirm the tools are available

### Example Queries

Here are some example queries you can try with Claude once the MCP server is connected:

- "Search for information about keyword search in the codebase"
- "Find code related to vector embeddings"
- "What's the implementation of search results ranking?"

## Development

To add new MCP tools or enhance existing ones, see the implementation in `knowlang/mcp/tools/`.
