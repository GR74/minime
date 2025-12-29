# Testing the Memory Layer

This guide shows you how to test the memory/embedding functionality we just implemented.

## Quick Start

### 1. Install Dependencies

First, make sure all dependencies are installed:

```bash
cd minime
pip install -r requirements.txt
```

This will install:
- `sentence-transformers` - for embeddings (will download model on first use)
- `aiosqlite` - for async database
- `python-frontmatter` - for parsing markdown frontmatter
- `numpy` - for similarity calculations

### 2. Run Quick Test

Run the quick test script to verify everything works:

```bash
cd minime
python scripts/quick_test.py
```

This will test:
- ✅ Module imports
- ✅ Embedding model (converts text to vectors)
- ✅ Chunking (splits long notes)
- ✅ Database operations (stores data)

### 3. Run Full Test Suite

For comprehensive testing including vault indexing:

```bash
cd minime
python scripts/test_memory.py
```

This creates a test vault with sample notes and indexes them, testing the full pipeline.

## Interactive Testing

You can also test components interactively in Python:

### Test Embeddings

```python
from minime.memory.embeddings import EmbeddingModel

# Create model (downloads on first use, ~80MB)
model = EmbeddingModel()

# Encode a single text
text = "Machine learning is fascinating"
embedding = model.encode_single(text)
print(f"Dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# Encode multiple texts (batch)
texts = [
    "Python is a programming language",
    "Machine learning uses algorithms", 
    "The weather is nice today"
]
embeddings = model.encode(texts)
print(f"Encoded {len(embeddings)} texts")
```

### Test Similarity

```python
import numpy as np
from minime.memory.embeddings import EmbeddingModel

model = EmbeddingModel()

# Encode two texts
text1 = "Machine learning is a subset of artificial intelligence"
text2 = "Machine learning uses algorithms to learn from data"
text3 = "The weather is sunny today"

emb1 = model.encode_single(text1)
emb2 = model.encode_single(text2)
emb3 = model.encode_single(text3)

# Compute cosine similarity
def cosine_similarity(a, b):
    vec1 = np.array(a)
    vec2 = np.array(b)
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)

similarity_12 = cosine_similarity(emb1, emb2)
similarity_13 = cosine_similarity(emb1, emb3)

print(f"Similarity (ML vs ML): {similarity_12:.3f}")  # Should be >0.7
print(f"Similarity (ML vs Weather): {similarity_13:.3f}")  # Should be <0.3
```

### Test Chunking

```python
from minime.memory.chunk import chunk_note

# Create a long note
long_note = """
# Machine Learning Overview

Machine learning is a subset of artificial intelligence. It involves training models on data to make predictions.

## Supervised Learning

Supervised learning uses labeled data. The model learns from examples where the correct answer is known.

## Unsupervised Learning

Unsupervised learning finds patterns without labels. The model discovers hidden structure in the data.

## Deep Learning

Deep learning uses neural networks with many layers. These networks can learn complex patterns.
"""

# Chunk it
chunks = chunk_note(long_note, max_tokens=50, overlap=10)

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(chunk.split())} words):")
    print(chunk[:100] + "...")
```

### Test Database

```python
import asyncio
from minime.memory.db import init_db, AsyncDatabase
from minime.schemas import VaultNode, MemoryChunk
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
        frontmatter={"author": "test"},
        tags=["test", "demo"],
        domain="coding",
        scope="global",
        links=["other-note"],
        backlinks=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        embedding_ref="test_chunk_0"
    )
    
    # Insert node
    await db.insert_node(node)
    print("✓ Node inserted")
    
    # Retrieve node
    retrieved = await db.get_node("test_123")
    print(f"✓ Retrieved: {retrieved.title}")
    print(f"  Tags: {retrieved.tags}")
    print(f"  Links: {retrieved.links}")
    
    # Insert chunk
    chunk = MemoryChunk(
        chunk_id="test_chunk_0",
        node_id="test_123",
        content="This is test content",
        embedding=[0.1] * 384,
        metadata={"test": True},
        position=0
    )
    await db.insert_chunk(chunk)
    print("✓ Chunk inserted")
    
    # Get chunks
    chunks = await db.get_chunks_for_node("test_123")
    print(f"✓ Retrieved {len(chunks)} chunks")
    
    # Get embeddings
    embeddings = await db.get_all_node_embeddings()
    print(f"✓ Retrieved {len(embeddings)} node embeddings")
    
    await db.close()
    print("\n✓ All database operations worked!")

asyncio.run(test())
```

### Test Vault Indexing

1. Create a test vault directory:
```bash
mkdir test_vault
```

2. Create a sample note `test_vault/sample.md`:
```markdown
---
title: Machine Learning Basics
tags: [ai, ml, tutorial]
domain: coding
---

# Machine Learning Basics

Machine learning is a method of data analysis that automates analytical model building.

## Key Concepts

- Supervised Learning
- Unsupervised Learning  
- Reinforcement Learning

See also: [[Deep Learning]]
```

3. Index it:
```python
import asyncio
from minime.memory.db import init_db, AsyncDatabase
from minime.memory.embeddings import EmbeddingModel
from minime.memory.vault import VaultIndexer

async def test_index():
    # Setup
    init_db("./test.db")
    db = AsyncDatabase("./test.db")
    embedding_model = EmbeddingModel()
    
    # Index vault
    print("Indexing vault...")
    indexer = VaultIndexer("./test_vault", db, embedding_model)
    nodes = await indexer.index()
    
    print(f"\n✓ Indexed {len(nodes)} notes")
    for node in nodes:
        print(f"\n  {node.title}:")
        print(f"    Path: {node.path}")
        print(f"    Tags: {node.tags}")
        print(f"    Domain: {node.domain}")
        print(f"    Links: {node.links}")
        
        # Check chunks
        chunks = await db.get_chunks_for_node(node.node_id)
        print(f"    Chunks: {len(chunks)}")
    
    # Check similarity proposals
    proposals = await db.get_pending_proposals()
    print(f"\n✓ Created {len(proposals)} similarity proposals")
    for prop in proposals[:5]:
        print(f"  Similarity: {prop.weight:.3f}")
        print(f"  Rationale: {prop.rationale}")
    
    await db.close()

asyncio.run(test_index())
```

## What to Expect

### Embeddings
- **Dimensions**: 384 (for all-MiniLM-L6-v2 model)
- **Similar topics**: Similarity > 0.7
- **Different topics**: Similarity < 0.3
- **First run**: Downloads model (~80MB, one-time)

### Chunking
- Long notes split into overlapping chunks
- Default: 512 tokens per chunk, 128 overlap
- Sentence boundaries preserved when possible

### Database
- Creates SQLite database file
- Stores nodes, chunks, embeddings, edges, proposals
- All operations are async

### Vault Indexing
- Scans all `.md` files recursively
- Extracts frontmatter, tags, wikilinks
- Computes embeddings for chunks
- Creates explicit edges (wikilinks)
- Generates similarity proposals (>0.7 threshold)

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError`, make sure:
1. You're in the `minime/` directory
2. Dependencies are installed: `pip install -r requirements.txt`
3. Python can find the `minime` package (should work if running from `minime/`)

### Model Download
On first run, the embedding model downloads automatically (~80MB). This is normal and only happens once.

### Database Locked
If you get "database is locked" errors:
- Make sure to call `await db.close()` after use
- Don't open multiple connections to the same database simultaneously

### Low Similarity Scores
Different topics should have low similarity (<0.3). If similar topics show low similarity:
- Check that embeddings are being computed correctly
- Verify the model loaded properly

## Next Steps

After testing, you can:
1. Index your own Obsidian vault
2. View similarity proposals
3. Use the database for retrieval (when context manager is implemented)
4. Generate auto-notes (when orchestrator is implemented)

For more details, see:
- `scripts/README_TESTING.md` - Detailed test documentation
- `docs/EXPLANATION_MEMORY_DATABASE.md` - Database documentation
- `docs/EXPLANATION_VAULT_INDEXER.md` - Indexer documentation

