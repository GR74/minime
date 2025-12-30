# MemorySearch - Semantic Search Engine

## Overview

`search.py` provides the `MemorySearch` class, which implements semantic search over indexed memory chunks. It uses vector similarity (cosine similarity) to find relevant content based on meaning, not just keyword matching.

## Purpose

The MemorySearch class enables:
- **Semantic Search**: Find content by meaning, not keywords
- **FAISS Integration**: Fast approximate nearest neighbor search
- **Metadata Validation**: Ensures embedding version compatibility
- **Flexible Filtering**: Search within specific nodes or all nodes

## Key Components

### MemorySearch Class

```python
class MemorySearch:
    def __init__(
        self,
        db: AsyncDatabase,
        embedding_model: EmbeddingModel,
        use_faiss: bool = True,
        faiss_index_path: Optional[str] = None,
    )
    
    async def search(
        query: str,
        k: int = 5,
        threshold: float = 0.0,
        node_ids: List[str] = None,
    ) -> List[Tuple[MemoryChunk, float]]
    
    async def search_nodes(
        query: str,
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[VaultNode, float]]
```

### Memory Class

```python
class Memory:
    async def write(text: str, metadata: dict = None) -> str
    async def read(query: str, k: int = 5) -> List[str]
    async def link(a: str, b: str, reason: str) -> str
```

## How It Works

### 1. Query Encoding

The search process starts by encoding the query:

```python
async def search(self, query: str, k: int = 5, ...):
    # Encode query to embedding
    query_embedding = self.embedding_model.encode_single(query)
    current_meta = self.embedding_model.get_embedding_metadata()
```

**Example**:
```python
query = "How do I structure my code?"
query_embedding = [0.2, 0.8, -0.1, ...]  # 384 numbers
```

### 2. FAISS Search (Fast Path)

If FAISS is available and not filtering by node_ids:

```python
if self.use_faiss and self._faiss_store and not node_ids:
    # Ensure index is built
    if self._faiss_store.size() == 0:
        await self._build_faiss_index()
    
    # Search using FAISS
    faiss_results = self._faiss_store.search(query_embedding, k=k * 2)
    
    # Fetch chunks and validate
    results = []
    for chunk_id, similarity in faiss_results:
        chunk = await self._get_chunk_by_id(chunk_id)
        
        # Validate metadata compatibility
        if not metadata_matches(current_meta, chunk.metadata["embedding"]):
            continue  # Skip incompatible embeddings
        
        results.append((chunk, similarity))
    
    return results[:k]
```

**Benefits**:
- **Fast**: O(log n) search instead of O(n)
- **Scalable**: Handles millions of embeddings
- **Efficient**: Pre-computed index

### 3. Linear Search (Fallback)

If FAISS unavailable or filtering by node_ids:

```python
# Get all chunks (or filtered by node_ids)
if node_ids:
    all_chunks = []
    for node_id in node_ids:
        chunks = await self.db.get_chunks_for_node(node_id)
        all_chunks.extend(chunks)
else:
    # Get all chunks from all nodes
    all_chunks = await self._get_all_chunks()

# Compute similarity for each chunk
results = []
for chunk in all_chunks:
    # Validate metadata
    if not metadata_matches(current_meta, chunk.metadata["embedding"]):
        continue
    
    # Compute cosine similarity
    similarity = self._compute_similarity(query_embedding, chunk.embedding)
    
    if similarity >= threshold:
        results.append((chunk, similarity))

# Sort by similarity and return top k
results.sort(key=lambda x: x[1], reverse=True)
return results[:k]
```

### 4. Similarity Computation

Cosine similarity measures the angle between vectors:

```python
def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)
```

**Range**: -1.0 to 1.0
- **1.0**: Identical meaning
- **0.0**: Unrelated
- **-1.0**: Opposite meaning

## Usage Examples

### Basic Search

```python
from minime.memory.search import MemorySearch
from minime.memory.db import AsyncDatabase
from minime.memory.embeddings import EmbeddingModel

# Initialize
db = AsyncDatabase("./data/minime.db")
model = EmbeddingModel()
search = MemorySearch(db, model)

# Search
results = await search.search("How do I structure my code?", k=5)

# Process results
for chunk, similarity in results:
    print(f"Similarity: {similarity:.3f}")
    print(f"Content: {chunk.content[:100]}...")
```

### Search with Threshold

```python
# Only return results above 0.7 similarity
results = await search.search(
    "Python best practices",
    k=10,
    threshold=0.7
)
```

### Search Within Specific Nodes

```python
# Search only in specific notes
node_ids = ["node1", "node2", "node3"]
results = await search.search(
    "modularity principles",
    k=5,
    node_ids=node_ids
)
```

### Search Nodes (Not Chunks)

```python
# Search for entire nodes (uses primary chunk)
node_results = await search.search_nodes(
    "architecture patterns",
    k=5
)

for node, similarity in node_results:
    print(f"{node.title}: {similarity:.3f}")
```

## Memory Interface

The `Memory` class provides a simpler interface:

### Write to Memory

```python
from minime.memory.search import Memory

memory = Memory(db, model, indexer)

# Write text (creates ephemeral node)
node_id = await memory.write(
    "I learned that modularity is important for code organization",
    metadata={"source": "conversation"}
)
```

**Creates**:
- Ephemeral VaultNode
- MemoryChunk with embedding
- Ready for search

### Read from Memory

```python
# Search and get content
results = await memory.read("modularity", k=5)
# Returns: ["chunk content 1", "chunk content 2", ...]
```

### Link Nodes

```python
# Create manual link between nodes
edge_id = await memory.link(
    "node1",
    "node2",
    reason="Both discuss modularity"
)
```

## FAISS Integration

### Building Index

```python
# Build index from all chunks
await search.rebuild_index()
```

**When to rebuild**:
- After bulk inserts
- After model upgrade
- Periodically for optimization

### Incremental Updates

```python
# Add single chunk to index
await search.add_to_index(chunk)
```

**Use case**: Add new chunks without full rebuild

## Metadata Validation

### Why It Matters

Embeddings from different model versions are incompatible:

```python
# Old embedding (model v1)
old_embedding = [0.2, 0.8, ...]  # Created with v1

# New embedding (model v2)
new_embedding = [0.3, 0.7, ...]  # Created with v2

# Can't compare directly - different vector spaces!
```

### Validation Process

```python
from minime.memory.embedding_utils import metadata_matches

current_meta = embedding_model.get_embedding_metadata()
stored_meta = chunk.metadata.get("embedding")

if not metadata_matches(current_meta, stored_meta):
    # Skip incompatible embeddings
    continue
```

**Checks**:
- Provider (sentence-transformers, openai, etc.)
- Model name (all-MiniLM-L6-v2, etc.)
- Revision (v1, v2, etc.)
- Encoder SHA (code version)

## Performance Considerations

### FAISS vs Linear Search

**FAISS (IndexFlatIP)**:
- **Time**: O(log n) for search
- **Space**: O(n) for index
- **Best for**: Large datasets (>1000 chunks)

**Linear Search**:
- **Time**: O(n) for search
- **Space**: O(1) (no index)
- **Best for**: Small datasets (<1000 chunks)

### Optimization Tips

1. **Use FAISS**: Enable for better performance
2. **Batch Queries**: Process multiple queries together
3. **Filter Early**: Use node_ids filter when possible
4. **Rebuild Index**: Periodically for optimization
5. **Cache Results**: Cache frequent queries

## Best Practices

1. **Set Appropriate Threshold**: Filter low-quality results
   ```python
   results = await search.search(query, threshold=0.6)
   ```

2. **Use Node Filtering**: Narrow search when possible
   ```python
   results = await search.search(query, node_ids=[...])
   ```

3. **Validate Metadata**: Always check compatibility
4. **Rebuild Index**: After bulk operations
5. **Handle Empty Results**: Check if results are empty

## Common Issues

### Issue: No Results

**Problem**: Search returns empty results

**Solutions**:
- Lower threshold
- Check if embeddings exist
- Verify metadata compatibility
- Ensure index is built (FAISS)

### Issue: Slow Search

**Problem**: Search takes too long

**Solutions**:
- Enable FAISS
- Use node_ids filter
- Reduce k (number of results)
- Rebuild index

### Issue: Incompatible Embeddings

**Problem**: Results filtered out due to version mismatch

**Solutions**:
- Re-embed with current model
- Check embedding metadata
- Upgrade model version consistently

## Summary

The `MemorySearch` class provides:

- ✅ Semantic search by meaning
- ✅ FAISS integration for speed
- ✅ Metadata validation for compatibility
- ✅ Flexible filtering options
- ✅ Simple Memory interface

It's the core component that enables finding relevant content in the MiniMe memory system.

