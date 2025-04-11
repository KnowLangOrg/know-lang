# Model Context Protocol (MCP) for Knowlang

This package provides an implementation of the Model Context Protocol (MCP) for Knowlang's search capabilities. It allows language models like Claude to access Knowlang's keyword and vector search features.

## Overview

The Model Context Protocol (MCP) standardizes how applications provide context to LLMs. This implementation exposes Knowlang's search capabilities as MCP tools that can be used by Claude or other MCP clients.

## Features

- **Keyword Search Tool**: Performs keyword-based search on your knowledge base
- **Vector Search Tool**: Performs semantic search using vector embeddings

## Installation

The MCP server is included as part of the Knowlang package. Make sure you have the `mcp-python` package installed:

```bash
pip install mcp-python>=1.2.0
```

## Usage

### Starting the MCP Server

You can start the MCP server using the CLI:

```bash
python -m knowlang.mcp.cli
```

By default, the server listens on `localhost:7773`. You can customize this with arguments:

```bash
python -m knowlang.mcp.cli --host 0.0.0.0 --port 8080 --name my-knowlang-server
```

### Using with Claude for Desktop

To use the MCP server with Claude for Desktop:

1. Make sure you have Claude for Desktop installed and updated to the latest version
2. Copy the `claude_desktop_config.json` file to your Claude Desktop configuration directory:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`
3. Start the MCP server using the CLI
4. Restart Claude for Desktop
5. Look for the hammer icon in Claude for Desktop to confirm the tools are available

### Example Queries

Here are some example queries you can try with Claude once the MCP server is connected:

- "Search for information about keyword search in the codebase"
- "Find code related to vector embeddings"
- "What's the implementation of search results ranking?"

## Advanced Configuration

You can customize the MCP server behavior by modifying the configuration in your Knowlang config file:

```yaml
mcp:
  host: localhost
  port: 7773
  server_name: knowlang-search
  
  # Search tool settings
  search:
    max_results: 10
    score_threshold: 0.5
```

## Troubleshooting

If you encounter issues:

1. Check that the MCP server is running (`python -m knowlang.mcp.cli --debug`)
2. Verify that the configuration in Claude for Desktop is correct
3. Make sure the `mcp-python` package is installed (version 1.2.0 or higher)
4. Look for error messages in the server logs

## Development

To add new MCP tools or enhance existing ones, see the implementation in `knowlang/mcp/tools/`.
