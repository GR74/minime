"""Quick test script to verify FAISS is working."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from minime.memory import AsyncDatabase, EmbeddingModel, MemorySearch


async def test_faiss():
    """Test FAISS search."""
    db_path = "demo_memory.db"
    index_path = "demo_memory.faiss"
    
    print("Testing FAISS integration...\n")
    
    # Initialize
    db = AsyncDatabase(db_path)
    await db._session.connect()
    
    try:
        model = EmbeddingModel()
        search = MemorySearch(
            db,
            model,
            use_faiss=True,
            faiss_index_path=index_path
        )
        
        # Check if FAISS is enabled
        if not search.use_faiss:
            print("ERROR: FAISS not available")
            return
        
        print(f"FAISS enabled: {search.use_faiss}")
        if search._faiss_store:
            print(f"Index size: {search._faiss_store.size()} vectors")
        
        # Test search
        print("\nTesting search: 'Python async programming'")
        results = await search.search("Python async programming", k=3)
        
        print(f"\nFound {len(results)} results:")
        for i, (chunk, score) in enumerate(results, 1):
            node = await db.get_node(chunk.node_id)
            title = node.title if node else "Unknown"
            print(f"\n{i}. [{title}] (similarity: {score:.3f})")
            print(f"   {chunk.content[:100]}...")
        
        print("\nFAISS is working correctly!")
        
    finally:
        await db._session.close()


if __name__ == "__main__":
    asyncio.run(test_faiss())

