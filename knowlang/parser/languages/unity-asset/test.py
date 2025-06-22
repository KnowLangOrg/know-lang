#!/usr/bin/env python3
"""
Test for UnityAssetParser chunking functionality
"""

import asyncio
import re
import json
from pathlib import Path
from parser import UnityAssetParser


class MockConfig:
    """Mock config for testing"""
    def __init__(self):
        self.parser = MockParserConfig()
        self.db = Path.cwd()


class MockParserConfig:
    """Mock parser config"""
    def __init__(self):
        self.languages = {
            "unity-asset": MockLanguageConfig()
        }


class MockLanguageConfig:
    """Mock language config"""
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.file_extensions = [".asset"]


def extract_ids_from_chunk(chunk_content):
    """Extract all IDs from a chunk's JSON content using regex"""
    # Parse the JSON array
    try:
        elements = json.loads(chunk_content)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {chunk_content[:100]}...")
        return {"node_ids": [], "connection_guids": [], "group_guids": []}
    
    node_ids = []
    connection_guids = []
    group_guids = []
    
    for element in elements:
        element_type = element.get("$type", "")
        
        if "Connection" in element_type:
            # Extract connection info
            guid = element.get("guid", "")
            source_ref = element.get("sourceUnit", {}).get("$ref", "")
            dest_ref = element.get("destinationUnit", {}).get("$ref", "")
            connection_guids.append({
                "guid": guid[:8] + "...",
                "flow": f"{source_ref} -> {dest_ref}",
                "type": element_type.replace("Bolt.", "")
            })
        elif "GraphGroup" in element_type:
            # Extract group info
            guid = element.get("guid", "")
            label = element.get("label", "No label")
            group_guids.append({
                "guid": guid[:8] + "...",
                "label": label
            })
        elif element.get("$id"):
            # Extract node info
            node_id = element.get("$id", "")
            node_type = element_type.replace("Bolt.", "").replace("Ludiq.", "")
            position = element.get("position", {})
            node_ids.append({
                "id": node_id,
                "type": node_type,
                "position": f"({position.get('x', 0)}, {position.get('y', 0)})"
            })
    
    return {
        "node_ids": node_ids,
        "connection_guids": connection_guids,
        "group_guids": group_guids
    }


async def test_chunking():
    """Test the Unity Asset Parser chunking functionality"""
    # Initialize parser
    config = MockConfig()
    parser = UnityAssetParser(config)
    parser.setup()
    
    # Path to test file
    test_file = Path(__file__).parent / "test.asset"
    
    print("Unity Visual Script Chunking Test")
    print("=" * 50)
    print(f"Testing file: {test_file}")
    print()
    
    # Parse the file and get chunks
    chunks = await parser.parse_file(test_file)
    
    print(f"Generated {len(chunks)} chunks:")
    print("=" * 50)
    
    # Analyze each chunk
    for i, chunk in enumerate(chunks):
        print(f"\nCHUNK {i + 1}:")
        print("-" * 30)
        
        # Extract IDs from the chunk
        ids_info = extract_ids_from_chunk(chunk.content)
        
        # Show chunk metadata
        print(f"Content length: {len(chunk.content)} characters")
        print(f"Language: {chunk.language}")
        print(f"Type: {chunk.type}")
        print()
        
        # Show nodes in this chunk
        if ids_info["node_ids"]:
            print("NODES:")
            for node in ids_info["node_ids"]:
                print(f"  - Node {node['id']}: {node['type']} at {node['position']}")
        else:
            print("NODES: None")
        print()
        
        # Show connections in this chunk
        if ids_info["connection_guids"]:
            print("CONNECTIONS:")
            for conn in ids_info["connection_guids"]:
                print(f"  - {conn['type']}: {conn['flow']} (GUID: {conn['guid']})")
        else:
            print("CONNECTIONS: None")
        print()
        
        # Show groups in this chunk
        if ids_info["group_guids"]:
            print("GROUPS:")
            for group in ids_info["group_guids"]:
                print(f"  - Group: {group['label']} (GUID: {group['guid']})")
        else:
            print("GROUPS: None")
        print()
        
        # Show raw chunk content for copy/paste
        print("RAW CHUNK CONTENT:")
        print("```")
        print(chunk.content)
        print("```")
        print()
        
        print("-" * 30)
    
    # Summary for manual verification
    print("\nSUMMARY FOR MANUAL VERIFICATION:")
    print("=" * 50)
    
    all_node_ids = set()
    all_connection_flows = []
    
    for i, chunk in enumerate(chunks):
        ids_info = extract_ids_from_chunk(chunk.content)
        
        chunk_node_ids = [node["id"] for node in ids_info["node_ids"]]
        chunk_connections = [conn["flow"] for conn in ids_info["connection_guids"]]
        
        if chunk_node_ids:
            all_node_ids.update(chunk_node_ids)
            print(f"Chunk {i+1} contains nodes: {', '.join(sorted(chunk_node_ids))}")
        
        if chunk_connections:
            all_connection_flows.extend(chunk_connections)
            print(f"Chunk {i+1} contains connections: {'; '.join(chunk_connections)}")
        
        if not chunk_node_ids and not chunk_connections:
            print(f"Chunk {i+1} contains only groups (no nodes/connections)")
    
    print()
    print("ALL NODE IDs FOUND:", sorted(all_node_ids))
    print()
    print("ALL CONNECTIONS:")
    for flow in sorted(set(all_connection_flows)):
        print(f"  {flow}")
    
    print()
    print("VERIFICATION GUIDE:")
    print("- Check that connected nodes are in the same chunk")
    print("- Check that nodes within the same visual group are in the same chunk") 
    print("- Check that isolated nodes are in separate chunks")


if __name__ == "__main__":
    asyncio.run(test_chunking())
