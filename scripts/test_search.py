"""Quick test for search functionality."""

import asyncio
import sys
from pathlib import Path

# Add minime directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from minime.memory import (
    AsyncDatabase,
    EmbeddingModel,
    MemorySearch,
    init_db,
)


async def test_search():
    """Test search functionality."""
    print("Testing MemorySearch...")
    
    # Initialize
    db_path = "test_search.db"
    init_db(db_path)  # init_db is not async
    db = AsyncDatabase(db_path)
    model = EmbeddingModel()
    search = MemorySearch(db, model)
    
    print("[OK] MemorySearch instantiated")
    
    # Test search on empty DB (should return empty list)
    results = await search.search("test query", k=5)
    print(f"[OK] Search on empty DB returned {len(results)} results")
    
    print("\n[SUCCESS] MemorySearch basic test passed!")


if __name__ == "__main__":
    asyncio.run(test_search())

