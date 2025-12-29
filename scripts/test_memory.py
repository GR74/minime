#!/usr/bin/env python3
"""Test script for memory layer: database, vault indexing, embeddings, and similarity."""

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
from minime.memory.chunk import chunk_note


async def test_chunking():
    """Test note chunking."""
    print("\n" + "="*60)
    print("TEST 1: Note Chunking")
    print("="*60)
    
    # Create a long note
    long_note = """
# Machine Learning Overview

Machine learning is a subset of artificial intelligence. It involves training models on data to make predictions. There are many types of machine learning algorithms.

## Supervised Learning

Supervised learning uses labeled data. The model learns from examples where the correct answer is known. Common algorithms include linear regression, decision trees, and neural networks.

## Unsupervised Learning

Unsupervised learning finds patterns without labels. The model discovers hidden structure in the data. Clustering and dimensionality reduction are common techniques.

## Reinforcement Learning

Reinforcement learning learns through trial and error. The agent receives rewards for good actions and penalties for bad ones. This is used in game playing and robotics.

## Deep Learning

Deep learning uses neural networks with many layers. These networks can learn complex patterns. They're inspired by the human brain's structure.
"""
    
    chunks = chunk_note(long_note, max_tokens=50, overlap=10)  # Small chunks for demo
    
    print(f"\nOriginal note: {len(long_note.split())} words")
    print(f"Number of chunks: {len(chunks)}")
    print("\nChunks:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n  Chunk {i+1} ({len(chunk.split())} words):")
        print(f"  {chunk[:100]}...")
    
    if len(chunks) > 3:
        print(f"\n  ... and {len(chunks) - 3} more chunks")
    
    return len(chunks) > 0


async def test_database():
    """Test database operations."""
    print("\n" + "="*60)
    print("TEST 2: Database Operations")
    print("="*60)
    
    # Create test database
    test_db_path = "./test_memory.db"
    if Path(test_db_path).exists():
        Path(test_db_path).unlink()  # Remove if exists
    
    # Initialize database
    init_db(test_db_path)
    print(f"\n✓ Database initialized: {test_db_path}")
    
    db = AsyncDatabase(test_db_path)
    
    try:
        # Test node operations
        from minime.minime.schemas import VaultNode
        
        test_node = VaultNode(
            node_id="test_node_123",
            path="test/note.md",
            title="Test Note",
            frontmatter={"tags": ["test"]},
            tags=["test", "demo"],
            domain="coding",
            scope="global",
            links=["other-note"],
            backlinks=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            embedding_ref="test_chunk_0"
        )
        
        await db.insert_node(test_node)
        print("✓ Node inserted")
        
        retrieved_node = await db.get_node("test_node_123")
        assert retrieved_node is not None
        assert retrieved_node.title == "Test Note"
        print("✓ Node retrieved")
        
        count = await db.count_nodes()
        assert count == 1
        print(f"✓ Node count: {count}")
        
        # Test chunk operations
        from minime.minime.schemas import MemoryChunk
        
        test_chunk = MemoryChunk(
            chunk_id="test_chunk_0",
            node_id="test_node_123",
            content="This is a test chunk",
            embedding=[0.1] * 384,  # Dummy embedding
            metadata={"test": True},
            position=0
        )
        
        await db.insert_chunk(test_chunk)
        print("✓ Chunk inserted")
        
        chunks = await db.get_chunks_for_node("test_node_123")
        assert len(chunks) == 1
        assert chunks[0].content == "This is a test chunk"
        print("✓ Chunks retrieved")
        
        # Test embeddings retrieval
        embeddings = await db.get_all_node_embeddings()
        assert len(embeddings) == 1
        assert len(embeddings[0][1]) == 384
        print("✓ Embeddings retrieved")
        
        await db.close()
        print("\n✓ All database tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await db.close()
        # Clean up
        if Path(test_db_path).exists():
            Path(test_db_path).unlink()


async def test_embeddings():
    """Test embedding model."""
    print("\n" + "="*60)
    print("TEST 3: Embedding Model")
    print("="*60)
    
    try:
        model = EmbeddingModel("all-MiniLM-L6-v2")
        
        # Test single encoding
        text1 = "Machine learning is a subset of artificial intelligence"
        embedding1 = model.encode_single(text1)
        print(f"\n✓ Single text encoded: {len(embedding1)} dimensions")
        print(f"  First 5 values: {embedding1[:5]}")
        
        # Test batch encoding
        texts = [
            "Python is a programming language",
            "Machine learning uses algorithms",
            "The weather is nice today"
        ]
        embeddings = model.encode(texts)
        assert len(embeddings) == 3
        assert len(embeddings[0]) == 384
        print(f"\n✓ Batch encoded {len(texts)} texts")
        
        # Test similarity
        import numpy as np
        vec1 = np.array(embedding1)
        vec2 = np.array(embeddings[1])  # "Machine learning uses algorithms"
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm1 * norm2)
        
        print(f"\n✓ Similarity test:")
        print(f"  Text 1: '{text1[:50]}...'")
        print(f"  Text 2: '{texts[1]}'")
        print(f"  Similarity: {similarity:.3f}")
        print(f"  (Expected: >0.7 for similar topics)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vault_indexing():
    """Test vault indexing with sample notes."""
    print("\n" + "="*60)
    print("TEST 4: Vault Indexing")
    print("="*60)
    
    # Create test vault directory
    test_vault = Path("./test_vault")
    if test_vault.exists():
        shutil.rmtree(test_vault)
    test_vault.mkdir()
    
    try:
        # Create sample notes
        note1_content = """---
title: Machine Learning Basics
tags: [ai, ml, tutorial]
domain: coding
---

# Machine Learning Basics

Machine learning is a method of data analysis that automates analytical model building. It uses algorithms to learn from data.

## Key Concepts

- **Supervised Learning**: Learning with labeled examples
- **Unsupervised Learning**: Finding patterns in unlabeled data
- **Reinforcement Learning**: Learning through rewards and penalties

See also: [[Deep Learning Overview]]
"""
        
        note2_content = """---
title: Deep Learning Overview
tags: [ai, deep-learning]
domain: coding
---

# Deep Learning Overview

Deep learning is a subset of machine learning that uses neural networks with multiple layers. These networks can learn complex patterns.

## Neural Networks

Neural networks consist of:
- Input layer
- Hidden layers
- Output layer

Related: [[Machine Learning Basics]]
"""
        
        note3_content = """---
title: Weather Forecast
tags: [weather]
---

# Weather Forecast

Today's weather is sunny with a high of 75°F. Perfect day for outdoor activities!

No need for an umbrella today.
"""
        
        # Write notes to vault
        (test_vault / "note1.md").write_text(note1_content)
        (test_vault / "note2.md").write_text(note2_content)
        (test_vault / "note3.md").write_text(note3_content)
        
        print(f"\n✓ Created test vault with 3 notes: {test_vault}")
        
        # Create database
        test_db_path = "./test_vault_indexing.db"
        if Path(test_db_path).exists():
            Path(test_db_path).unlink()
        init_db(test_db_path)
        
        db = AsyncDatabase(test_db_path)
        embedding_model = EmbeddingModel()
        indexer = VaultIndexer(str(test_vault), db, embedding_model)
        
        # Index vault
        print("\n📚 Indexing vault...")
        nodes = await indexer.index()
        
        print(f"\n✓ Indexed {len(nodes)} notes")
        for node in nodes:
            print(f"\n  Node: {node.title}")
            print(f"    Path: {node.path}")
            print(f"    Tags: {node.tags}")
            print(f"    Domain: {node.domain}")
            print(f"    Links: {node.links}")
        
        # Check chunks
        all_chunks = []
        for node in nodes:
            chunks = await db.get_chunks_for_node(node.node_id)
            all_chunks.extend(chunks)
            print(f"\n  Chunks for '{node.title}': {len(chunks)}")
        
        print(f"\n✓ Total chunks created: {len(all_chunks)}")
        
        # Check edges
        from minime.minime.schemas import GraphEdge
        # Note: We'd need a method to get edges, but for now just verify indexing worked
        
        # Check proposals
        proposals = await db.get_pending_proposals()
        print(f"\n✓ Similarity proposals created: {len(proposals)}")
        if proposals:
            print("\n  Proposals:")
            for prop in proposals[:5]:  # Show first 5
                print(f"    {prop.source_node_id[:8]}... → {prop.target_node_id[:8]}... (similarity: {prop.weight:.3f})")
        
        await db.close()
        
        # Clean up
        if Path(test_db_path).exists():
            Path(test_db_path).unlink()
        
        print("\n✓ Vault indexing test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Vault indexing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up vault
        if test_vault.exists():
            shutil.rmtree(test_vault)


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MiniMe Memory Layer Test Suite")
    print("="*60)
    print("\nTesting memory layer components:")
    print("  1. Note chunking")
    print("  2. Database operations")
    print("  3. Embedding model")
    print("  4. Vault indexing (full integration)")
    
    results = []
    
    # Run tests
    results.append(("Chunking", await test_chunking()))
    results.append(("Database", await test_database()))
    results.append(("Embeddings", await test_embeddings()))
    results.append(("Vault Indexing", await test_vault_indexing()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check output above.")
    
    print("\n" + "="*60)
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

