# Utility Scripts Documentation

## Overview

This document describes utility scripts that help you explore, maintain, and manage your MiniMe memory system.

## Utility Scripts

### 1. memory_explorer.py

**Purpose**: Interactive CLI for exploring the memory system

**Features**:
- Semantic search
- Node details
- System statistics
- Interactive exploration mode

**Usage**:

#### Search Command

```bash
python scripts/memory_explorer.py search "your query"
```

**Options**:
```bash
python scripts/memory_explorer.py search "machine learning" \
  --db-path "memory.db" \
  --k 10  # Number of results
```

**Example**:
```bash
python scripts/memory_explorer.py search "Python async programming"
```

**Output**:
```
🔍 Searching for: 'Python async programming'

Found 3 results:

1. [Python Best Practices] (similarity: 0.856)
   Use async/await for I/O operations in Python. This improves performance and scalability...

2. [Concurrency Patterns] (similarity: 0.823)
   Asynchronous programming allows multiple operations to run concurrently...

3. [API Design] (similarity: 0.789)
   When designing APIs, consider using async patterns for better responsiveness...
```

#### Stats Command

```bash
python scripts/memory_explorer.py stats
```

**Output**:
```
📊 Memory System Statistics

Nodes (Notes): 15
Chunks: 42
Edges: 23
Pending Proposals: 5
```

#### Node Command

```bash
python scripts/memory_explorer.py node "Note Title"
```

**Example**:
```bash
python scripts/memory_explorer.py node "Machine Learning Basics"
```

**Output**:
```
📄 Node: Machine Learning Basics

Path: notes/ml-basics.md
Domain: coding
Scope: global
Tags: ai, ml, tutorial
Created: 2025-12-29 10:30:00
Updated: 2025-12-29 10:30:00

Chunks: 3
Connections: 5

Connections:
  - Deep Learning Overview (wikilink)
  - Neural Networks (similar)
  - AI Fundamentals (similar)
  ...
```

#### Explore Command (Interactive Mode)

```bash
python scripts/memory_explorer.py explore
```

**Interactive Commands**:
```
🔍 Memory Explorer - Interactive Mode

Commands:
  search <query> - Search semantically
  stats - Show statistics
  node <title> - Show node details
  quit - Exit

> search machine learning
1. [Machine Learning Basics] (0.856)
   Machine learning is a subset of AI...

> stats
Nodes: 15

> node Python Basics
Found: Python Basics

> quit
```

**When to use**:
- Quick searches
- Exploring your knowledge base
- Finding specific notes
- Understanding connections

---

### 2. rebuild_faiss_index.py

**Purpose**: Rebuild FAISS index from existing database

**Why needed**:
- After bulk updates
- After model upgrade
- If index becomes corrupted
- To optimize search performance

**Usage**:
```bash
python scripts/rebuild_faiss_index.py rebuild
```

**Options**:
```bash
python scripts/rebuild_faiss_index.py rebuild \
  --db-path "memory.db" \
  --index-path "memory.faiss"
```

**What it does**:
1. Loads all chunks from database
2. Validates embedding metadata
3. Rebuilds FAISS index
4. Saves index to disk
5. Tests search functionality

**Output**:
```
Initializing search engine with FAISS...
Building FAISS index from database...
Index built successfully!
  - Index size: 42 vectors
  - Index saved to: memory.faiss
  - ID map saved to: memory.faiss.idmap

Testing search...
  - Test search returned 3 results
  - Top result similarity: 0.856
```

**When to use**:
- After indexing many new notes
- After upgrading embedding model
- If search seems slow
- If index file is missing
- Periodic maintenance

**Performance**:
- **Time**: O(n) where n is number of chunks
- **Typical**: ~1-5 seconds for 1000 chunks
- **Memory**: Loads all embeddings temporarily

---

## Utility Workflows

### Daily Exploration

```bash
# Quick search
python scripts/memory_explorer.py search "today's topic"

# Check stats
python scripts/memory_explorer.py stats

# View specific note
python scripts/memory_explorer.py node "Note Title"
```

### After Indexing

```bash
# Index vault
python scripts/demo_memory.py run

# Rebuild FAISS index
python scripts/rebuild_faiss_index.py rebuild

# Verify with search
python scripts/memory_explorer.py search "test query"
```

### Maintenance

```bash
# Check system health
python scripts/memory_explorer.py stats

# Rebuild index (if needed)
python scripts/rebuild_faiss_index.py rebuild

# Visualize graph
python scripts/viz_graph.py generate
```

---

## Integration Examples

### Search and Explore

```bash
# Search for content
python scripts/memory_explorer.py search "Python"

# Get details on top result
python scripts/memory_explorer.py node "Python Basics"

# See connections
python scripts/viz_graph.py generate
```

### Index and Rebuild

```bash
# Index new notes
python scripts/demo_memory.py run --vault-path "./vault"

# Rebuild search index
python scripts/rebuild_faiss_index.py rebuild

# Verify
python scripts/memory_explorer.py stats
```

---

## Best Practices

### Regular Maintenance

1. **Rebuild Index**: After bulk updates
   ```bash
   python scripts/rebuild_faiss_index.py rebuild
   ```

2. **Check Statistics**: Monitor growth
   ```bash
   python scripts/memory_explorer.py stats
   ```

3. **Explore Graph**: Understand structure
   ```bash
   python scripts/viz_graph.py stats
   ```

### Search Optimization

1. **Use FAISS**: Faster for large databases
2. **Rebuild Regularly**: Keep index up-to-date
3. **Check Metadata**: Ensure compatibility

### Exploration Tips

1. **Start with Stats**: Understand your knowledge base
2. **Use Interactive Mode**: Better for exploration
3. **Follow Connections**: Use node command to explore
4. **Visualize**: See the big picture

---

## Troubleshooting

### Search Returns No Results

**Problem**: Search finds nothing

**Solutions**:
- Check database has data: `python scripts/memory_explorer.py stats`
- Verify FAISS index exists
- Rebuild index: `python scripts/rebuild_faiss_index.py rebuild`
- Try different query terms

### FAISS Rebuild Fails

**Problem**: Index rebuild fails

**Solutions**:
- Check FAISS installed: `pip install faiss-cpu`
- Verify database has chunks
- Check embedding metadata compatibility
- Delete old index files and rebuild

### Node Not Found

**Problem**: "Node not found" error

**Solutions**:
- Check exact title spelling
- Use partial search first
- Verify node exists: `python scripts/memory_explorer.py stats`
- Check database path

### Slow Search

**Problem**: Search takes too long

**Solutions**:
- Rebuild FAISS index
- Check FAISS is enabled
- Reduce number of results (--k)
- Verify index size matches database

---

## Advanced Usage

### Custom Database Paths

```bash
# Use different database
python scripts/memory_explorer.py search "query" --db-path "./data/custom.db"
```

### Batch Operations

```bash
# Rebuild multiple indices
for db in *.db; do
    python scripts/rebuild_faiss_index.py rebuild --db-path "$db"
done
```

### Integration with Other Tools

```bash
# Search and pipe to file
python scripts/memory_explorer.py search "query" > results.txt

# Stats in JSON (future feature)
python scripts/memory_explorer.py stats --format json
```

---

## Summary

The utility scripts provide:
- ✅ Interactive exploration (`memory_explorer.py`)
- ✅ Index maintenance (`rebuild_faiss_index.py`)
- ✅ Quick access to data
- ✅ System health monitoring
- ✅ Easy integration

Use them to explore, maintain, and understand your MiniMe memory system!

