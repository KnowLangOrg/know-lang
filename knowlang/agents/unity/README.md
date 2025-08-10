# Unity UI Generation Agent

A pydantic-ai graph-based system for automatic Unity UI generation with HTTP/WebSocket communication between C# frontend and Python backend.

## Overview

The Unity agent generates complete Unity UI Toolkit components:
- **UXML**: Unity UI XML markup
- **USS**: Unity Style Sheets for styling
- **C#**: Boilerplate code for event binding

## Architecture

```
┌─────────────────┐ HTTP/WebSocket ┌─────────────────┐
│   Unity C#      │◄──────────────►│  Python Backend │
│   Frontend      │                │                 │
└─────────────────┘                └─────────────────┘
        │                                   │
        ▼                                   ▼
┌─────────────────┐                ┌─────────────────┐
│ C# Classes      │                │ Python Classes  │
│ (Generated from │                │ (Generated from │
│  .proto)        │                │  .proto)        │
└─────────────────┘                └─────────────────┘
```

## Setup

### 1. Generate Protobuf Files

```bash
python -m grpc_tools.protoc \
  -Igrpc_stub=./knowlang-api/protos/ \
  --python_out=./ \
  --pyi_out=./ \
  ./knowlang-api/protos/unity/ui_generation.proto
```

**Command Options Explained:**
- `-Igrpc_stub=./knowlang-api/protos/`: Sets the import path for protobuf files
- `--python_out=./`: Generates Python message classes in current directory
- `--pyi_out=./`: Generates Python type stubs for better IDE support

**Note:** We use protobuf only for schema definition. Communication between Unity C# and Python backend uses HTTP or WebSocket due to Unity's limited gRPC support.


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

## Benefits

- **Type Safety**: Compile-time guarantees across language boundaries
- **Maintainability**: Single source of truth for enum definitions
- **Scalability**: Easy to add new status types or modify existing ones
- **Reliability**: No runtime enum mapping errors 