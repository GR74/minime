# Explanation: `minime/memory/db.py`

This file provides the database layer for storing vault nodes, chunks, embeddings, and graph edges in SQLite.

## Overview

The database layer uses SQLite (a local, file-based database) to store all memory-related data:
- **Vault nodes**: Metadata about each Obsidian note
- **Memory chunks**: Text chunks with embeddings
- **Graph edges**: Explicit connections (wikilinks) and approved edges
- **Graph proposals**: Similarity-based edge proposals awaiting approval

## File Structure

```python
# Functions
init_db(db_path)                    # Create database schema

# Classes
class AsyncDatabase:
    # Node operations
    - insert_node(node)             # Insert/update vault node
    - get_node(node_id)             # Retrieve node
    - count_nodes()                 # Count total nodes
    - node_exists(node_id)          # Check if node exists
    
    # Chunk operations
    - insert_chunk(chunk)           # Store chunk with embedding
    - get_chunks_for_node(node_id)  # Get all chunks for a node
    
    # Edge operations
    - insert_edge(edge)             # Store approved edge
    - insert_proposal(proposal)     # Store similarity proposal
    - get_pending_proposals()       # Get proposals needing approval
    - approve_proposal(proposal_id) # Move proposal to edges
    
    # Similarity search
    - get_all_node_embeddings()     # Get embeddings for similarity search
```

---

## Database Schema

### Table: `vault_nodes`

Stores metadata about each note in the Obsidian vault.

**Columns:**
- `node_id` (TEXT PRIMARY KEY) - Hash of file path
- `path` (TEXT) - Relative path in vault
- `title` (TEXT) - Note title
- `frontmatter` (JSON) - Parsed YAML frontmatter
- `tags` (JSON) - Array of tags
- `domain` (TEXT) - Domain tag (e.g., "biotech", "coding")
- `scope` (TEXT) - Scope tag (default: "global")
- `links` (JSON) - Outgoing wikilinks `[[link]]`
- `backlinks` (JSON) - Incoming links (computed later)
- `created_at` (TIMESTAMP) - Creation time
- `updated_at` (TIMESTAMP) - Last update time
- `embedding_ref` (TEXT) - Reference to primary chunk

**Indexes:**
- `idx_vault_nodes_path` - Fast path lookups
- `idx_vault_nodes_domain` - Filter by domain
- `idx_vault_nodes_scope` - Filter by scope

**Example:**
```python
{
    "node_id": "abc123...",
    "path": "notes/my-note.md",
    "title": "My Note",
    "tags": ["ai", "memory"],
    "domain": "coding",
    "scope": "global",
    "links": ["other-note", "reference"],
    "backlinks": [],
    "embedding_ref": "abc123_chunk_0"
}
```

---

### Table: `memory_chunks`

Stores chunked text content with embeddings.

**Columns:**
- `chunk_id` (TEXT PRIMARY KEY) - Unique chunk ID
- `node_id` (TEXT) - Foreign key to vault_nodes
- `content` (TEXT) - Chunk text content
- `embedding` (JSON) - Embedding vector (384 floats for all-MiniLM-L6-v2)
- `metadata` (JSON) - Additional metadata (chunk_index, total_chunks)
- `position` (INTEGER) - Position in note (0, 1, 2, ...)

**Why chunking?**
- Notes can be very long
- Embeddings work better on smaller text segments
- Enables fine-grained retrieval (find specific parts of notes)

**Example:**
```python
{
    "chunk_id": "abc123_chunk_0",
    "node_id": "abc123...",
    "content": "This is the first chunk of text...",
    "embedding": [0.2, 0.8, -0.1, ...],  # 384 numbers
    "metadata": {"chunk_index": 0, "total_chunks": 3},
    "position": 0
}
```

---

### Table: `graph_edges`

Stores approved graph connections between notes.

**Columns:**
- `edge_id` (TEXT PRIMARY KEY) - Unique edge ID
- `source_node_id` (TEXT) - Source note
- `target_node_id` (TEXT) - Target note
- `edge_type` (TEXT) - "wikilink" | "similar" | "related"
- `weight` (REAL) - Edge weight (0.0-1.0)
- `rationale` (TEXT) - Why edge exists
- `confidence` (REAL) - Confidence score (0.0-1.0)
- `is_approved` (BOOLEAN) - Always TRUE in this table
- `created_at` (TIMESTAMP)
- `approved_at` (TIMESTAMP)

**Types of edges:**
1. **wikilink**: Explicit `[[link]]` in note (confidence=1.0, weight=1.0)
2. **similar**: Semantic similarity (confidence=similarity_score, weight=similarity_score)

---

### Table: `graph_proposals`

Stores similarity-based edge proposals awaiting approval.

**Columns:**
- `proposal_id` (TEXT PRIMARY KEY) - Unique proposal ID
- `source_node_id` (TEXT) - Source note
- `target_node_id` (TEXT) - Target note
- `edge_type` (TEXT) - Usually "similar"
- `weight` (REAL) - Similarity score
- `confidence` (REAL) - Confidence score
- `rationale` (TEXT) - Proposal rationale
- `requires_user_approval` (BOOLEAN) - Approval flag
- `created_at` (TIMESTAMP)

**Workflow:**
1. Similarity search finds related notes
2. Creates proposal in `graph_proposals`
3. User reviews proposals (via CLI)
4. Approved proposals move to `graph_edges`

---

## Function: `init_db(db_path: str)`

**Purpose**: Creates all database tables and indexes.

**What it does:**
1. Creates database file if it doesn't exist
2. Creates all 4 tables with proper schema
3. Creates indexes for fast queries
4. Sets up foreign key constraints

**Example:**
```python
from minime.memory.db import init_db

# Initialize database
init_db("./data/minime.db")
# Now database is ready to use!
```

**Why synchronous?**
- SQLite table creation must be synchronous
- Called once during initialization
- Async operations happen after tables exist

---

## Class: `AsyncDatabase`

**Purpose**: Provides async database operations for all memory layer data.

### Constructor: `__init__(db_path: str)`

**What it does:**
- Stores database path
- Connection created lazily on first use

**Example:**
```python
db = AsyncDatabase("./data/minime.db")
# Connection not created yet
```

---

### Method: `async insert_node(node: VaultNode) -> str`

**Purpose**: Insert or update a vault node.

**What it does:**
1. Serializes JSON fields (frontmatter, tags, links, backlinks)
2. Uses `INSERT OR REPLACE` for upsert behavior
3. Returns node_id

**Example:**
```python
node = VaultNode(
    node_id="abc123",
    path="notes/my-note.md",
    title="My Note",
    frontmatter={"tags": ["ai"]},
    tags=["ai", "memory"],
    links=["other-note"],
    # ... other fields
)

node_id = await db.insert_node(node)
# Node is now stored in database
```

---

### Method: `async get_node(node_id: str) -> Optional[VaultNode]`

**Purpose**: Retrieve a node by ID.

**Returns**: `VaultNode` if found, `None` otherwise.

**Example:**
```python
node = await db.get_node("abc123")
if node:
    print(node.title)  # "My Note"
```

---

### Method: `async insert_chunk(chunk: MemoryChunk) -> str`

**Purpose**: Store a text chunk with its embedding.

**What it does:**
1. Serializes embedding (list of floats → JSON)
2. Stores chunk metadata
3. Links chunk to node via `node_id`

**Example:**
```python
chunk = MemoryChunk(
    chunk_id="abc123_chunk_0",
    node_id="abc123",
    content="This is chunk content...",
    embedding=[0.2, 0.8, ...],  # 384 numbers
    metadata={"chunk_index": 0},
    position=0
)

await db.insert_chunk(chunk)
```

---

### Method: `async get_chunks_for_node(node_id: str) -> List[MemoryChunk]`

**Purpose**: Get all chunks for a specific note.

**Returns**: List of chunks ordered by position.

**Use case**: Retrieving all chunks when reconstructing a note.

---

### Method: `async insert_edge(edge: GraphEdge) -> str`

**Purpose**: Store an approved graph edge.

**What it does:**
- Stores edge with `is_approved=True`
- Sets `approved_at` timestamp

**Example:**
```python
edge = GraphEdge(
    edge_id="abc123_def456_wikilink",
    source_node_id="abc123",
    target_node_id="def456",
    edge_type="wikilink",
    weight=1.0,
    rationale="Explicit wikilink: [[other-note]]",
    confidence=1.0,
    is_approved=True,
    created_at=datetime.now(),
    approved_at=datetime.now()
)

await db.insert_edge(edge)
```

---

### Method: `async insert_proposal(proposal: GraphEdge) -> str`

**Purpose**: Store a similarity proposal for review.

**What it does:**
1. Creates proposal with `is_approved=False`
2. Sets `requires_user_approval` based on confidence
   - If confidence < 0.9: requires approval
   - If confidence >= 0.9: auto-approve (future)

**Example:**
```python
proposal = GraphEdge(
    edge_id="abc123_def456_similar",
    source_node_id="abc123",
    target_node_id="def456",
    edge_type="similar",
    weight=0.85,  # Similarity score
    rationale="Semantic similarity: 0.850",
    confidence=0.85,
    is_approved=False,
    created_at=datetime.now()
)

await db.insert_proposal(proposal)
# Proposal now available for review
```

---

### Method: `async get_pending_proposals() -> List[GraphEdge]`

**Purpose**: Get all proposals requiring approval.

**Returns**: List of proposals ordered by creation date (newest first).

**Use case**: CLI command to review proposals.

---

### Method: `async approve_proposal(proposal_id: str) -> bool`

**Purpose**: Approve a proposal by moving it to `graph_edges`.

**What it does:**
1. Retrieves proposal from `graph_proposals`
2. Creates edge with `is_approved=True`
3. Inserts into `graph_edges`
4. Deletes from `graph_proposals`

**Returns**: `True` if successful, `False` if proposal not found.

**Example:**
```python
success = await db.approve_proposal("abc123_def456_similar")
if success:
    print("Proposal approved!")
```

---

### Method: `async get_all_node_embeddings() -> List[Tuple[str, List[float]]]`

**Purpose**: Get all primary chunk embeddings for similarity search.

**Returns**: List of `(node_id, embedding)` tuples.

**How it works:**
1. For each node, gets the primary chunk embedding
2. Primary chunk = `embedding_ref` if set, otherwise first chunk
3. Returns embeddings for similarity comparison

**Use case**: Computing similarity between new note and existing notes.

**Example:**
```python
embeddings = await db.get_all_node_embeddings()
# Returns: [("abc123", [0.2, 0.8, ...]), ("def456", [0.3, 0.7, ...]), ...]

for node_id, embedding in embeddings:
    similarity = compute_cosine_similarity(new_embedding, embedding)
    if similarity > 0.7:
        # These notes are similar!
```

---

## Key Concepts

### 1. JSON Storage

SQLite doesn't have native JSON columns, so we store JSON as TEXT and serialize/deserialize manually:

```python
# Storing
frontmatter_json = json.dumps(node.frontmatter)

# Retrieving
frontmatter = json.loads(row["frontmatter"])
```

### 2. Async Operations

All database operations are async using `aiosqlite`:
- Non-blocking I/O
- Better for concurrent operations
- Required for async/await patterns

### 3. Upsert Pattern

`INSERT OR REPLACE` allows updating existing records:
- First insert: Creates new record
- Subsequent inserts: Updates existing record
- Useful for re-indexing (update when note changes)

### 4. Foreign Key Constraints

Tables use foreign keys with `ON DELETE CASCADE`:
- Deleting a node automatically deletes its chunks and edges
- Maintains data consistency
- Prevents orphaned records

### 5. Primary Chunk Selection

Each node has one "primary chunk" used for similarity:
- Stored in `embedding_ref` field
- If not set, uses first chunk (lowest position)
- Enables fast similarity search (one embedding per note)

---

## Integration

### With VaultIndexer

```python
# VaultIndexer uses AsyncDatabase to store indexed notes
indexer = VaultIndexer(vault_path, db, embedding_model)
nodes = await indexer.index()  # Stores nodes, chunks, edges in db
```

### With Context Manager (Future)

```python
# Context Manager queries database for relevant chunks
chunks = await db.get_chunks_for_node(node_id)
# Retrieves chunks for context assembly
```

### With Graph System (Future)

```python
# Graph system queries edges for traversal
edges = await db.get_edges_for_node(node_id)
# Builds graph connections
```

---

## Error Handling

Current implementation has basic error handling:
- Database errors propagate (should be caught by caller)
- JSON serialization errors would raise `json.JSONDecodeError`
- Foreign key violations would raise SQLite errors

**Potential improvements:**
- Retry logic for transient errors
- Better error messages
- Transaction rollback on errors

---

## Performance Considerations

### Indexes

Database has indexes on:
- `vault_nodes.path` - Fast path lookups
- `vault_nodes.domain` - Fast domain filtering
- `memory_chunks.node_id` - Fast chunk retrieval
- `graph_edges.source_node_id` - Fast edge traversal

### Query Optimization

- `get_all_node_embeddings()` uses efficient SQL with `COALESCE` and subqueries
- Limits results to primary chunks only (one per node)
- Avoids loading all chunks for similarity search

### Scaling

For large vaults (1000+ notes):
- Consider batch operations
- May need vector database (Milvus, Qdrant) for similarity search
- Index maintenance for frequent updates

---

## Summary

The `AsyncDatabase` class is the **storage layer** for MiniMe's memory system:

1. **Stores vault metadata** - Notes, tags, links, frontmatter
2. **Stores embeddings** - Chunks with vector embeddings
3. **Manages graph** - Explicit edges and similarity proposals
4. **Enables similarity search** - Provides embeddings for comparison
5. **Async interface** - Non-blocking database operations

Without the database layer, MiniMe would have no persistent memory. All indexed notes, embeddings, and connections are stored here, making them available for retrieval and similarity search.

