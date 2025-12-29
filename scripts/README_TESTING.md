# Testing the Memory Layer

This guide shows you how to test the memory/embedding functionality.

## Prerequisites

1. Install dependencies:
```bash
cd minime
pip install -r requirements.txt
```

2. Make sure you have the required packages:
- `sentence-transformers` - for embeddings
- `aiosqlite` - for async database
- `python-frontmatter` - for parsing frontmatter
- `numpy` - for similarity calculations

## Running the Test Suite

Run the comprehensive test script:

```bash
cd minime
python -m scripts.test_memory
```

This will test:
1. ✅ Note chunking (splitting long notes)
2. ✅ Database operations (storing nodes, chunks, edges)
3. ✅ Embedding model (converting text to vectors)
4. ✅ Vault indexing (full integration test)

## Quick Test

You can also test individual components interactively:

### Test Embeddings

```python
from minime.memory.embeddings import EmbeddingModel

model = EmbeddingModel()
embedding = model.encode_single("Machine learning is interesting")
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

### Test Chunking

```python
from minime.memory.chunk import chunk_note

long_text = "This is a very long note with lots of content..." * 100
chunks = chunk_note(long_text, max_tokens=100, overlap=20)
print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}: {len(chunk.split())} words")
```

### Test Database

```python
import asyncio
from minime.memory.db import init_db, AsyncDatabase
from minime.schemas import VaultNode
from datetime import datetime

async def test():
    # Initialize database
    init_db("./test.db")
    db = AsyncDatabase("./test.db")
    
    # Create a test node
    node = VaultNode(
        node_id="test_123",
        path="test/note.md",
        title="Test Note",
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
    
    # Insert and retrieve
    await db.insert_node(node)
    retrieved = await db.get_node("test_123")
    print(f"Retrieved: {retrieved.title}")
    
    count = await db.count_nodes()
    print(f"Total nodes: {count}")
    
    await db.close()

asyncio.run(test())
```

### Test Vault Indexing

1. Create a test vault directory:
```bash
mkdir test_vault
```

2. Create a sample note in `test_vault/sample.md`:
```markdown
---
title: Sample Note
tags: [test, example]
domain: coding
---

# Sample Note

This is a test note with some content.

See also: [[other-note]]
```

3. Index it:
```python
import asyncio
from pathlib import Path
from minime.memory.db import init_db, AsyncDatabase
from minime.memory.embeddings import EmbeddingModel
from minime.memory.vault import VaultIndexer

async def test_index():
    # Setup
    init_db("./test.db")
    db = AsyncDatabase("./test.db")
    embedding_model = EmbeddingModel()
    
    # Index vault
    indexer = VaultIndexer("./test_vault", db, embedding_model)
    nodes = await indexer.index()
    
    print(f"Indexed {len(nodes)} notes")
    for node in nodes:
        print(f"- {node.title}: {len(node.tags)} tags, {len(node.links)} links")
    
    # Check chunks
    for node in nodes:
        chunks = await db.get_chunks_for_node(node.node_id)
        print(f"{node.title}: {len(chunks)} chunks")
    
    # Check proposals
    proposals = await db.get_pending_proposals()
    print(f"Similarity proposals: {len(proposals)}")
    
    await db.close()

asyncio.run(test_index())
```

## Expected Output

When running the full test suite, you should see:

```
============================================================
MiniMe Memory Layer Test Suite
============================================================

Testing memory layer components:
  1. Note chunking
  2. Database operations
  3. Embedding model
  4. Vault indexing (full integration)

============================================================
TEST 1: Note Chunking
============================================================

Original note: 150 words
Number of chunks: 4

Chunks:
  Chunk 1 (50 words):
  Machine learning is a subset of artificial intelligence...

✓ Test passed

============================================================
TEST 2: Database Operations
============================================================

✓ Database initialized: ./test_memory.db
✓ Node inserted
✓ Node retrieved
✓ Node count: 1
✓ Chunk inserted
✓ Chunks retrieved
✓ Embeddings retrieved

✓ All database tests passed!

============================================================
TEST 3: Embedding Model
============================================================

✓ Single text encoded: 384 dimensions
  First 5 values: [0.123, 0.456, -0.789, 0.234, 0.567]

✓ Batch encoded 3 texts

✓ Similarity test:
  Text 1: 'Machine learning is a subset of artificial intel...'
  Text 2: 'Machine learning uses algorithms'
  Similarity: 0.856
  (Expected: >0.7 for similar topics)

============================================================
TEST 4: Vault Indexing
============================================================

✓ Created test vault with 3 notes: test_vault
📚 Indexing vault...

✓ Indexed 3 notes

  Node: Machine Learning Basics
    Path: note1.md
    Tags: ['ai', 'ml', 'tutorial']
    Domain: coding
    Links: ['Deep Learning Overview']

  Chunks for 'Machine Learning Basics': 1

✓ Total chunks created: 3

✓ Similarity proposals created: 2

  Proposals:
    abc12345... → def67890... (similarity: 0.823)

✓ Vault indexing test passed!

============================================================
TEST SUMMARY
============================================================
Chunking            ✓ PASSED
Database            ✓ PASSED
Embeddings          ✓ PASSED
Vault Indexing      ✓ PASSED

🎉 All tests passed!
```

## Troubleshooting

### ModuleNotFoundError

If you get `ModuleNotFoundError`, install dependencies:
```bash
pip install -r requirements.txt
```

### Model Download

On first run, the embedding model will be downloaded (~80MB). This is one-time only.

### Database Locked

If you get database locked errors, make sure to close connections:
```python
await db.close()
```

### Similarity Scores

Similarity scores range from -1.0 to 1.0:
- `>0.7`: Very similar (proposals created)
- `0.5-0.7`: Somewhat similar
- `<0.5`: Not very similar

Different topics (e.g., "machine learning" vs "weather") should have low similarity (<0.3).

