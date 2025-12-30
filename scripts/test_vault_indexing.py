#!/usr/bin/env python3
"""Test script specifically for vault indexing to exercise metadata validation and embedding paths."""

import asyncio
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
import sys
from pathlib import Path

# Add minime directory to path
minime_dir = Path(__file__).parent.parent
sys.path.insert(0, str(minime_dir))

from minime.memory.db import AsyncDatabase, init_db
from minime.memory.embeddings import EmbeddingModel
from minime.memory.vault import VaultIndexer


async def test_vault_indexing():
    """Test vault indexing with a test markdown file."""
    print("\n" + "="*60)
    print("TEST: Vault Indexing")
    print("="*60)
    
    # Create test directory structure
    test_dir = Path("./test_vault")
    test_dir.mkdir(exist_ok=True)
    
    # Clean up old test files
    for file in test_dir.glob("*.md"):
        file.unlink()
    
    # Create a test markdown file
    test_file = test_dir / "test_note.md"
    test_file.write_text("""---
title: Test Note
tags: [test, example]
---

# Test Note

This is a test note with some content. It has multiple sentences. 
The content should be chunked and embedded. This will test the 
metadata validation and embedding paths.

[[Another Note]]

#test #example
""")
    
    # Initialize database
    test_db = "./test_vault.db"
    if Path(test_db).exists():
        Path(test_db).unlink()
    
    print("\nInitializing database...")
    init_db(test_db)
    print("[OK] Database initialized")
    
    db = AsyncDatabase(test_db)
    model = EmbeddingModel()
    indexer = VaultIndexer(str(test_dir), db, model)
    
    print("\nIndexing vault...")
    nodes = await indexer.index()
    
    print(f"[OK] Indexed {len(nodes)} nodes")
    if nodes:
        node = nodes[0]
        print(f"[OK] Node: {node.title}")
        print(f"[OK] Node ID: {node.node_id}")
        
        # Get chunks to verify metadata
        chunks = await db.get_chunks_for_node(node.node_id)
        print(f"[OK] Created {len(chunks)} chunks")
        if chunks:
            chunk = chunks[0]
            print(f"[OK] Chunk has embedding: {chunk.embedding is not None}")
            if chunk.metadata:
                print(f"[OK] Chunk metadata keys: {list(chunk.metadata.keys())}")
                if "embedding" in chunk.metadata:
                    embed_meta = chunk.metadata["embedding"]
                    print(f"[OK] Embedding metadata keys: {list(embed_meta.keys())}")
    
    await db.close()
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    Path(test_db).unlink(missing_ok=True)
    
    print("\n[SUCCESS] Vault indexing test passed!")
    return True


if __name__ == "__main__":
    asyncio.run(test_vault_indexing())

