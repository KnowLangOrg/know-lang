# Unity UI Generation Agent

A pydantic-ai graph-based system for automatic Unity UI generation with gRPC communication between C# frontend and Python backend.

## Architecture Overview

### Protobuf-First Design

The system uses a **protobuf-first approach** for enum definitions to ensure type safety and consistency between C# frontend and Python backend:

```
┌─────────────────┐    gRPC    ┌─────────────────┐
│   Unity C#      │◄──────────►│  Python Backend │
│   Frontend      │            │                 │
└─────────────────┘            └─────────────────┘
        │                               │
        │                               │
        ▼                               ▼
┌─────────────────┐            ┌─────────────────┐
│ C# Enums        │            │ Python Enums    │
│ (Generated from │            │ (Generated from │
│  .proto)        │            │  .proto)        │
└─────────────────┘            └─────────────────┘
```

### Why Protobuf-First Enums?

**Problems with Python-only enums:**
- ❌ Type mismatches between frontend and backend
- ❌ Serialization issues over gRPC
- ❌ Manual enum duplication in C#
- ❌ Version drift between languages
- ❌ No compile-time type safety

**Benefits of protobuf-first enums:**
- ✅ Single source of truth for enum definitions
- ✅ Automatic code generation for both languages
- ✅ Compile-time type safety
- ✅ Version compatibility guarantees
- ✅ Proper gRPC serialization

## File Structure

```
knowlang/agents/unity/
├── __init__.py                 # Package exports
├── README.md                   # This file
├── proto_enums.py             # Python protobuf enum wrapper
├── csharp_enums.cs            # C# enum definitions
├── ui_generation_graph.py     # Main graph orchestration
├── grpc_models.py             # gRPC request/response models
├── grpc_service.py            # gRPC service implementation
├── protos/
│   └── ui_generation.proto    # Protobuf service definition
└── nodes/
    ├── __init__.py            # Node exports
    ├── base.py                # Base types and state
    ├── uxml_generator.py      # UXML generation node
    ├── uss_generator.py       # USS generation node
    └── csharp_generator.py    # C# generation node
```

## Graph Flow

```
User Description
       │
       ▼
┌─────────────────┐
│ UXML Generator  │ ──► Generates Unity UI XML markup
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ USS Generator   │ ──► Generates Unity Style Sheets
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ C# Generator    │ ──► Generates C# boilerplate code
└─────────────────┘
       │
       ▼
   Final Result
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

### C# Frontend (Unity)

```csharp
using UnityUIGeneration;

// The enum values are guaranteed to match the Python backend
UIGenerationStatus status = UIGenerationStatus.GeneratingUxml;
string displayText = status.ToDisplayString(); // "Generating UXML"

if (status.IsInProgress())
{
    // Show progress indicator
}
```

## gRPC Communication

### Service Methods

1. **GenerateUIStream**: Streaming generation with real-time progress
2. **GenerateUISync**: Synchronous generation (final result only)
3. **GetGenerationStatus**: Check status of ongoing generation
4. **CancelGeneration**: Cancel ongoing generation

### Enum Consistency

The `UIGenerationStatus` enum is defined in:
- **Protobuf**: `protos/ui_generation.proto`
- **Python**: `proto_enums.py` (matches protobuf)
- **C#**: `csharp_enums.cs` (matches protobuf)

All three definitions are guaranteed to be in sync.

## Benefits of This Architecture

1. **Type Safety**: Compile-time guarantees across language boundaries
2. **Maintainability**: Single source of truth for enum definitions
3. **Scalability**: Easy to add new status types or modify existing ones
4. **Reliability**: No runtime enum mapping errors
5. **Developer Experience**: IntelliSense support in both languages

## Future Improvements

1. **Auto-generation**: Use `protoc` to automatically generate enum files
2. **Versioning**: Add protobuf versioning for backward compatibility
3. **Validation**: Add runtime validation for enum values
4. **Documentation**: Auto-generate API documentation from protobuf 