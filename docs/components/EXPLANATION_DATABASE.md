# AsyncDatabase - SQLite Database Interface

## Overview

`db.py` provides the `AsyncDatabase` class, which is the primary interface for all database operations in the MiniMe memory system. It manages vault nodes, memory chunks, graph edges, and proposals using SQLite with async/await support.

## Purpose

The AsyncDatabase class provides:
- **Unified Interface**: Single API for all database operations
- **Type Safety**: Uses Pydantic models for data validation
- **Transaction Management**: Proper commit/rollback handling
- **Efficient Storage**: Compressed embeddings and optimized queries
- **Graph Support**: Manages nodes, edges, and proposals

## Key Components

### Database Schema

The database consists of four main tables:

1. **vault_nodes**: Obsidian vault notes
2. **memory_chunks**: Chunked note content with embeddings
3. **graph_edges**: Explicit connections between nodes
4. **graph_proposals**: Proposed edges awaiting approval

### AsyncDatabase Class

```python
class AsyncDatabase:
    def __init__(self, db_path: str)
    async def insert_node(node: VaultNode) -> str
    async def get_node(node_id: str) -> Optional[VaultNode]
    async def insert_chunk(chunk: MemoryChunk) -> str
    async def get_chunks_for_node(node_id: str) -> List[MemoryChunk]
    async def insert_edge(edge: GraphEdge) -> str
    async def insert_proposal(proposal: GraphEdge) -> str
    async def get_all_node_embeddings_with_metadata() -> List[Tuple]
```

## Database Schema

### vault_nodes Table

```sql
CREATE TABLE vault_nodes (
    node_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    title TEXT,
    frontmatter JSON,
    tags JSON,
    domain TEXT,
    scope TEXT DEFAULT 'global',
    links JSON,
    backlinks JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    embedding_ref TEXT
)
```

**Purpose**: Stores metadata about each note in the vault.

**Key Fields**:
- `node_id`: Unique identifier (hash of path)
- `path`: Relative path in vault
- `frontmatter`: Parsed YAML frontmatter
- `links`: Outgoing wikilinks
- `embedding_ref`: Reference to primary chunk

### memory_chunks Table

```sql
CREATE TABLE memory_chunks (
    chunk_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding JSON,          -- Legacy (deprecated)
    embedding_blob BLOB,     -- Compressed float32 array
    metadata JSON,
    position INTEGER DEFAULT 0,
    FOREIGN KEY (node_id) REFERENCES vault_nodes(node_id)
)
```

**Purpose**: Stores chunked note content with embeddings.

**Key Features**:
- **Compressed Storage**: Embeddings stored as compressed float32 blobs
- **Metadata**: Versioning and content hashes
- **Position**: Order of chunks in note

### graph_edges Table

```sql
CREATE TABLE graph_edges (
    edge_id TEXT PRIMARY KEY,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    rationale TEXT,
    confidence REAL DEFAULT 1.0,
    is_approved BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP,
    approved_at TIMESTAMP,
    FOREIGN KEY (source_node_id) REFERENCES vault_nodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES vault_nodes(node_id)
)
```

**Purpose**: Stores explicit connections between nodes.

**Edge Types**:
- `wikilink`: Explicit [[link]] in note
- `similar`: Semantic similarity (proposed)
- `manual`: User-created link

### graph_proposals Table

```sql
CREATE TABLE graph_proposals (
    proposal_id TEXT PRIMARY KEY,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    edge_type TEXT DEFAULT 'similar',
    weight REAL,
    confidence REAL,
    rationale TEXT,
    requires_user_approval BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP,
    FOREIGN KEY (source_node_id) REFERENCES vault_nodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES vault_nodes(node_id)
)
```

**Purpose**: Stores proposed edges awaiting user approval.

## Key Operations

### 1. Node Management

#### Insert/Update Node

```python
async def insert_node(self, node: VaultNode) -> str:
    # Serialize JSON fields
    frontmatter_json = json.dumps(node.frontmatter)
    tags_json = json.dumps(node.tags)
    
    # Use INSERT OR REPLACE for upsert
    await self._session.execute("""
        INSERT OR REPLACE INTO vault_nodes
        (node_id, path, title, frontmatter, tags, ...)
        VALUES (?, ?, ?, ?, ?, ...)
    """, (node.node_id, node.path, ...))
    
    await self._session.commit()
    return node.node_id
```

**Features**:
- **Upsert**: Updates if exists, inserts if new
- **JSON Serialization**: Converts Python objects to JSON
- **Transaction**: Commits after operation

#### Get Node

```python
async def get_node(self, node_id: str) -> Optional[VaultNode]:
    row = await self._fetchone(
        "SELECT * FROM vault_nodes WHERE node_id = ?",
        (node_id,)
    )
    
    if row is None:
        return None
    
    # Deserialize JSON fields
    frontmatter = json.loads(row["frontmatter"]) if row["frontmatter"] else {}
    tags = json.loads(row["tags"]) if row["tags"] else []
    
    return VaultNode(
        node_id=row["node_id"],
        path=row["path"],
        frontmatter=frontmatter,
        tags=tags,
        ...
    )
```

### 2. Chunk Management

#### Insert Chunk with Compressed Embedding

```python
async def insert_chunk(self, chunk: MemoryChunk) -> str:
    import numpy as np
    import zlib
    
    # Convert to float32 array and compress
    embedding_array = np.array(chunk.embedding, dtype=np.float32)
    embedding_bytes = embedding_array.tobytes()
    embedding_blob = zlib.compress(embedding_bytes)
    
    await self._session.execute("""
        INSERT OR REPLACE INTO memory_chunks
        (chunk_id, node_id, content, embedding_blob, metadata, position)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        chunk.chunk_id,
        chunk.node_id,
        chunk.content,
        embedding_blob,  # Compressed blob
        json.dumps(chunk.metadata),
        chunk.position,
    ))
    
    await self._session.commit()
    return chunk.chunk_id
```

**Why Compression?**
- **Space Savings**: ~75% reduction in storage
- **Performance**: Faster I/O with smaller data
- **Scalability**: Can store millions of embeddings

#### Retrieve Chunk

```python
async def get_chunks_for_node(self, node_id: str) -> List[MemoryChunk]:
    rows = await self._fetch(
        "SELECT * FROM memory_chunks WHERE node_id = ? ORDER BY position",
        (node_id,)
    )
    
    chunks = []
    for row in rows:
        # Decompress embedding
        if row["embedding_blob"]:
            embedding_bytes = zlib.decompress(row["embedding_blob"])
            embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
            embedding = embedding_array.tolist()
        elif row["embedding"]:
            # Fallback to JSON (legacy)
            embedding = json.loads(row["embedding"])
        else:
            embedding = []
        
        chunks.append(MemoryChunk(
            chunk_id=row["chunk_id"],
            content=row["content"],
            embedding=embedding,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            ...
        ))
    
    return chunks
```

### 3. Graph Operations

#### Get All Node Embeddings

```python
async def get_all_node_embeddings_with_metadata(self) -> List[Tuple]:
    # Get primary chunk for each node
    cursor = await self._session.execute("""
        SELECT 
            vn.node_id,
            COALESCE(
                (SELECT mc.chunk_id FROM memory_chunks mc 
                 WHERE mc.node_id = vn.node_id AND mc.chunk_id = vn.embedding_ref),
                (SELECT mc.chunk_id FROM memory_chunks mc 
                 WHERE mc.node_id = vn.node_id 
                 ORDER BY mc.position ASC LIMIT 1)
            ) as chunk_id
        FROM vault_nodes vn
        WHERE EXISTS (
            SELECT 1 FROM memory_chunks mc WHERE mc.node_id = vn.node_id
        )
    """)
    
    # Fetch embeddings with metadata
    embeddings = []
    for node_row in node_rows:
        chunk_row = await self._fetchone(
            "SELECT embedding_blob, metadata FROM memory_chunks WHERE chunk_id = ?",
            (chunk_id,)
        )
        
        # Decompress and parse
        embedding = decompress_embedding(chunk_row["embedding_blob"])
        metadata = json.loads(chunk_row["metadata"])
        
        embeddings.append((node_id, embedding, metadata))
    
    return embeddings
```

**Purpose**: Used for similarity search across all nodes.

### 4. Proposal Management

#### Approve Proposal

```python
async def approve_proposal(self, proposal_id: str) -> bool:
    # Get proposal
    row = await self._fetchone(
        "SELECT * FROM graph_proposals WHERE proposal_id = ?",
        (proposal_id,)
    )
    
    if row is None:
        return False
    
    # Create edge from proposal
    edge = GraphEdge(
        edge_id=row["proposal_id"],
        source_node_id=row["source_node_id"],
        target_node_id=row["target_node_id"],
        is_approved=True,
        approved_at=datetime.now(),
        ...
    )
    
    # Insert edge and delete proposal in transaction
    conn = await self._session.connect()
    await conn.execute("INSERT OR REPLACE INTO graph_edges ...")
    await conn.execute("DELETE FROM graph_proposals WHERE proposal_id = ?", (proposal_id,))
    await self._session.commit()
    
    return True
```

## Usage Examples

### Basic Usage

```python
from minime.memory.db import AsyncDatabase, init_db

# Initialize database
init_db("./data/minime.db")

# Create database instance
db = AsyncDatabase("./data/minime.db")

# Insert node
node = VaultNode(
    node_id="abc123",
    path="notes/example.md",
    title="Example Note",
    ...
)
await db.insert_node(node)

# Get node
retrieved = await db.get_node("abc123")

# Close when done
await db.close()
```

### Batch Operations

```python
# Use session for batch operations
async with db.session() as conn:
    for node in nodes:
        await db.insert_node(node)
    # Single commit for all operations
    await db.session().commit()
```

### Query Examples

```python
# Get all nodes
all_nodes = await db.get_all_nodes()

# Get chunks for node
chunks = await db.get_chunks_for_node("abc123")

# Get pending proposals
proposals = await db.get_pending_proposals()

# Approve proposal
success = await db.approve_proposal("proposal_123")
```

## Performance Optimizations

### 1. Indexes

The database creates indexes for common queries:

```sql
CREATE INDEX idx_vault_nodes_path ON vault_nodes(path);
CREATE INDEX idx_vault_nodes_domain ON vault_nodes(domain);
CREATE INDEX idx_memory_chunks_node_id ON memory_chunks(node_id);
CREATE INDEX idx_graph_edges_source ON graph_edges(source_node_id);
```

### 2. WAL Mode

Write-Ahead Logging enables:
- Concurrent reads and writes
- Better performance
- Reduced lock contention

### 3. Compression

Embeddings are compressed using zlib:
- **Before**: 384 floats × 4 bytes = 1.5KB
- **After**: ~400 bytes (compressed)
- **Savings**: ~75% reduction

## Best Practices

1. **Always Close**: Call `close()` when done
2. **Use Transactions**: Group related operations
3. **Batch Operations**: Use session for multiple inserts
4. **Check Existence**: Verify before inserting
5. **Handle Errors**: Use try/except for database operations

## Common Issues

### Issue: Database Locked

**Problem**: Multiple processes accessing database

**Solution**: WAL mode handles this, but ensure proper connection management

### Issue: Large Database

**Problem**: Database grows too large

**Solutions**:
- Use compression (already implemented)
- Archive old data
- Consider database sharding

### Issue: Slow Queries

**Problem**: Queries taking too long

**Solutions**:
- Ensure indexes exist
- Use LIMIT for large result sets
- Consider query optimization

## Summary

The `AsyncDatabase` class provides:

- ✅ Complete database interface for memory system
- ✅ Efficient storage with compression
- ✅ Type-safe operations with Pydantic
- ✅ Graph support (nodes, edges, proposals)
- ✅ Optimized queries with indexes

It's the foundation for all data persistence in the MiniMe memory system.

