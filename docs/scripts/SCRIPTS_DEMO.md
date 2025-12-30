# Demo Scripts Documentation

## Overview

This document describes the demo scripts that showcase the MiniMe memory system functionality.

## Demo Scripts

### 1. demo_memory.py

**Purpose**: Interactive demonstration of the complete memory system

**What it demonstrates**:
1. Vault indexing
2. Writing memory
3. Semantic search
4. Graph visualization
5. Memory API (write/read/link)

**Usage**:
```bash
python scripts/demo_memory.py run
```

**Options**:
```bash
python scripts/demo_memory.py run \
  --vault-path "./vault" \
  --db-path "demo_memory.db" \
  --reset  # Reset database before demo
```

**What it does**:

#### Step 1: Indexing Vault
- Scans vault directory for `.md` files
- Parses frontmatter, extracts tags and wikilinks
- Creates embeddings for all notes
- Stores in database

#### Step 2: Writing Memory
- Creates ephemeral nodes with sample memories
- Demonstrates the `Memory.write()` API
- Shows how to add memories programmatically

#### Step 3: Semantic Search
- Performs semantic search queries
- Shows similarity scores
- Demonstrates FAISS integration

#### Step 4: Graph Visualization
- Loads graph data
- Shows statistics
- Generates interactive HTML visualization

#### Step 5: Memory API
- Demonstrates `write()`, `read()`, and `link()` methods
- Shows how to create connections between concepts

**Output**:
```
MiniMe Memory System Demo

============================================================

Step 1: Indexing Vault
------------------------------------------------------------
[OK] Indexed 5 notes

Sample nodes:
  - Machine Learning Basics (3 tags)
  - Deep Learning Overview (2 tags)
  ...

Step 2: Writing Memory
------------------------------------------------------------
[OK] Wrote: Learned: Use async/await for I/O operations...
[OK] Wrote: Important: Always validate user input...
[OK] Wrote: Pattern: Dependency injection improves...

Step 3: Semantic Search
------------------------------------------------------------

Searching: 'Python async'
  1. [Note Title] (score: 0.856)
     Chunk content preview...

Step 4: Graph Visualization
------------------------------------------------------------
[OK] Loaded graph: 8 nodes, 12 edges

Graph Statistics:
  Nodes: 8
  Edges: 12
  Wikilinks: 5
  Similarity: 7
  Avg Connections: 1.5

[OK] Generated visualization: demo_graph.html
   Open in browser to explore the graph!

Step 5: Memory API (write/read/link)
------------------------------------------------------------
[OK] Created nodes: abc12345... and def67890...
[OK] Created link between concepts
[OK] Found 2 relevant memories

============================================================
[OK] Demo complete!

Next steps:
  1. Open demo_graph.html in your browser
  2. Try: python scripts/memory_explorer.py search 'your query'
  3. Try: python scripts/viz_graph.py generate
```

**Generated Files**:
- `demo_memory.db`: SQLite database with indexed data
- `demo_memory.faiss`: FAISS index file
- `demo_memory.faiss.idmap`: FAISS ID mapping
- `demo_graph.html`: Interactive graph visualization

**When to use**:
- First-time user demonstration
- System showcase
- Learning the API
- Testing full pipeline

---

## Demo Workflow

### 1. Prepare Vault (Optional)

Create a test vault with sample notes:

```bash
mkdir demo_vault
cat > demo_vault/note1.md << 'EOF'
---
title: Machine Learning Basics
tags: [ai, ml]
---

# Machine Learning Basics

Machine learning is fascinating. It uses algorithms to learn from data.

See also: [[Deep Learning]]
EOF
```

### 2. Run Demo

```bash
python scripts/demo_memory.py run --vault-path "./demo_vault"
```

### 3. Explore Results

- Open `demo_graph.html` in browser
- Try searching: `python scripts/memory_explorer.py search "machine learning"`
- View statistics: `python scripts/viz_graph.py stats`

---

## Demo Features

### Interactive Elements

The demo shows:
- **Real-time indexing**: See notes being processed
- **Search results**: See semantic similarity in action
- **Graph structure**: Visualize connections
- **Statistics**: Understand your knowledge base

### Sample Data

The demo creates sample memories:
- "Learned: Use async/await for I/O operations in Python"
- "Important: Always validate user input before processing"
- "Pattern: Dependency injection improves testability"

These demonstrate:
- Ephemeral node creation
- Automatic embedding
- Searchability

---

## Customization

### Custom Vault

Use your own Obsidian vault:
```bash
python scripts/demo_memory.py run --vault-path "~/Documents/my-vault"
```

### Custom Database

Use different database:
```bash
python scripts/demo_memory.py run --db-path "my_demo.db"
```

### Reset Database

Start fresh:
```bash
python scripts/demo_memory.py run --reset
```

---

## Demo Output Files

### demo_memory.db

SQLite database containing:
- All indexed nodes
- Chunks with embeddings
- Graph edges
- Similarity proposals

**Size**: Depends on vault size (~1-10MB for small vaults)

### demo_graph.html

Interactive HTML visualization:
- **Nodes**: Notes in your vault
- **Edges**: Connections (wikilinks and similarity)
- **Interactive**: Hover for details, zoom, pan
- **Color-coded**: By domain or connection type

**Open**: Double-click file or:
```bash
# macOS
open demo_graph.html

# Linux
xdg-open demo_graph.html

# Windows
start demo_graph.html
```

### FAISS Index Files

- `demo_memory.faiss`: Vector index
- `demo_memory.faiss.idmap`: ID mapping

**Purpose**: Fast similarity search

---

## Troubleshooting

### No Notes Found

**Problem**: "Indexed 0 notes"

**Solutions**:
- Check vault path is correct
- Ensure vault contains `.md` files
- Check file permissions

### Database Errors

**Problem**: Database locked or corrupted

**Solutions**:
- Use `--reset` flag to start fresh
- Close other connections to database
- Delete database file and recreate

### FAISS Not Working

**Problem**: FAISS search not available

**Solutions**:
```bash
pip install faiss-cpu
```

### Graph Empty

**Problem**: No edges in visualization

**Solutions**:
- Add wikilinks to notes: `[[Other Note]]`
- Wait for similarity proposals (may take time)
- Check similarity threshold (default: 0.7)

---

## Next Steps After Demo

1. **Explore Graph**: Open `demo_graph.html`
2. **Try Search**: Use `memory_explorer.py search`
3. **View Stats**: Use `viz_graph.py stats`
4. **Index Your Vault**: Use your own Obsidian vault
5. **Read Documentation**: See other explanation files

---

## Summary

The demo script provides:
- ✅ Complete system demonstration
- ✅ Interactive walkthrough
- ✅ Visual output (graph HTML)
- ✅ Sample data creation
- ✅ API examples

Use it to understand how all components work together!

