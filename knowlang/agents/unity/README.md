# Unity UI Generation Agent

A pydantic-ai graph-based system for automatic Unity UI generation with gRPC communication between C# frontend and Python backend.

## Overview

The Unity agent generates complete Unity UI Toolkit components:
- **UXML**: Unity UI XML markup
- **USS**: Unity Style Sheets for styling
- **C#**: Boilerplate code for event binding

## Architecture

```
┌─────────────────┐    gRPC    ┌─────────────────┐
│   Unity C#      │◄──────────►│  Python Backend │
│   Frontend      │            │                 │
└─────────────────┘            └─────────────────┘
        │                               │
        ▼                               ▼
┌─────────────────┐            ┌─────────────────┐
│ C# Enums        │            │ Python Enums    │
│ (Generated from │            │ (Generated from │
│  .proto)        │            │  .proto)        │
└─────────────────┘            └─────────────────┘
```

## Setup

### 1. Generate Protobuf Files

```bash
python -m grpc_tools.protoc \
  -Igrpc_stub=./knowlang-api/protos/ \
  --python_out=./ \
  --grpc_python_out=./ \
  --pyi_out=./ \
  protos/unity/ui_generation.proto
```

**Command Options Explained:**
- `-Igrpc_stub=./knowlang-api/protos/`: Sets the import path for protobuf files
- `--python_out=./`: Generates Python message classes in current directory
- `--grpc_python_out=./`: Generates gRPC service classes in current directory  
- `--pyi_out=./`: Generates Python type stubs for better IDE support

### 2. File Structure

```
knowlang/agents/unity/
├── __init__.py                 # Package exports
├── ui_generation_graph.py     # Main graph orchestration
├── serve.py                   # gRPC service implementation
└── nodes/
    ├── base.py                # Base types and state
    ├── uxml_generator.py      # UXML generation node
    ├── uss_generator.py       # USS generation node
    └── csharp_generator.py    # C# generation node
```

## Usage

### Python Backend

```python
from knowlang.agents.unity import stream_ui_generation_progress

# Streaming generation with progress updates
async for result in stream_ui_generation_progress(
    ui_description="Create a login form with username and password fields"
):
    print(f"Status: {result.status} - {result.progress_message}")
    if result.is_complete:
        print(f"UXML: {result.uxml_content}")
        print(f"USS: {result.uss_content}")
        print(f"C#: {result.csharp_content}")
```

### gRPC Service

```python
from knowlang.agents.unity.serve import serve

# Start gRPC server
await serve(port=50051)
```

## Graph Flow

```
User Description
       │
       ▼
┌─────────────────┐
│ UXML Generator  │ ──► Unity UI XML markup
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ USS Generator   │ ──► Unity Style Sheets
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ C# Generator    │ ──► Event binding code
└─────────────────┘
       │
       ▼
   Final Result
```

## gRPC Methods

- `GenerateUIStream`: Streaming generation with real-time progress
- `GetGenerationStatus`: Check status of ongoing generation  
- `CancelGeneration`: Cancel ongoing generation

## Benefits

- **Type Safety**: Compile-time guarantees across language boundaries
- **Maintainability**: Single source of truth for enum definitions
- **Scalability**: Easy to add new status types or modify existing ones
- **Reliability**: No runtime enum mapping errors 