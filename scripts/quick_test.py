#!/usr/bin/env python3
"""Quick interactive test for memory layer components."""

import asyncio
import sys
from pathlib import Path

# Add minime directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that we can import all modules."""
    print("Testing imports...")
    try:
        from minime.memory.embeddings import EmbeddingModel
        from minime.memory.chunk import chunk_note
        from minime.memory.db import init_db, AsyncDatabase
        from minime.memory.vault import VaultIndexer
        print("[OK] All imports successful!")
        return True
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        print("\nMake sure dependencies are installed:")
        print("  pip install -r requirements.txt")
        return False


def test_embeddings_quick():
    """Quick embedding test."""
    print("\n" + "="*60)
    print("Testing Embeddings")
    print("="*60)
    
    try:
        from minime.memory.embeddings import EmbeddingModel
        
        print("\nLoading embedding model (this may take a moment on first run)...")
        model = EmbeddingModel()
        
        text = "Machine learning is fascinating"
        embedding = model.encode_single(text)
        
        print(f"[OK] Encoded text: '{text}'")
        print(f"[OK] Embedding dimensions: {len(embedding)}")
        print(f"[OK] First 5 values: {[round(x, 3) for x in embedding[:5]]}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunking_quick():
    """Quick chunking test."""
    print("\n" + "="*60)
    print("Testing Chunking")
    print("="*60)
    
    try:
        from minime.memory.chunk import chunk_note
        
        # Create a long note
        long_text = "This is a test sentence. " * 50  # ~50 sentences
        
        chunks = chunk_note(long_text, max_tokens=20, overlap=5)
        
        print(f"\n[OK] Created {len(chunks)} chunks from {len(long_text.split())} words")
        print(f"[OK] First chunk: {chunks[0][:50]}...")
        if len(chunks) > 1:
            print(f"[OK] Second chunk: {chunks[1][:50]}...")
        
        return True
    except Exception as e:
        print(f"[ERROR] Chunking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_database_quick():
    """Quick database test."""
    print("\n" + "="*60)
    print("Testing Database")
    print("="*60)
    
    try:
        from minime.memory.db import init_db, AsyncDatabase
        from minime.schemas import VaultNode
        from datetime import datetime
        
        test_db = "./quick_test.db"
        if Path(test_db).exists():
            Path(test_db).unlink()
        
        print("\nInitializing database...")
        init_db(test_db)
        print("[OK] Database initialized")
        
        db = AsyncDatabase(test_db)
        
        print("Creating test node...")
        node = VaultNode(
            node_id="test_quick",
            path="test/note.md",
            title="Quick Test Note",
            frontmatter={},
            tags=["test"],
            domain=None,
            scope="global",
            links=[],
            backlinks=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            embedding_ref=None
        )
        
        await db.insert_node(node)
        print("[OK] Node inserted")
        
        retrieved = await db.get_node("test_quick")
        print(f"[OK] Node retrieved: {retrieved.title}")
        
        count = await db.count_nodes()
        print(f"[OK] Node count: {count}")
        
        await db.close()
        
        # Clean up
        if Path(test_db).exists():
            Path(test_db).unlink()
        
        return True
    except Exception as e:
        print(f"[ERROR] Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run quick tests."""
    print("="*60)
    print("MiniMe Memory Layer - Quick Test")
    print("="*60)
    
    results = []
    
    # Test imports first
    if not test_imports():
        print("\n[WARNING] Cannot proceed without imports. Install dependencies first.")
        return False
    
    # Run quick tests
    results.append(("Embeddings", test_embeddings_quick()))
    results.append(("Chunking", test_chunking_quick()))
    results.append(("Database", await test_database_quick()))
    
    # Summary
    print("\n" + "="*60)
    print("QUICK TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"{name:20} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n[SUCCESS] Quick tests passed!")
        print("\nFor full test suite, run: python scripts\\test_memory.py")
    else:
        print("\n[WARNING] Some tests failed.")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

