# Explanation: `minime/memory/vault.py`

This file provides the vault indexer that scans Obsidian vaults, extracts metadata, computes embeddings, and creates graph connections.

## Overview

The `VaultIndexer` class is responsible for:
1. **Scanning** the Obsidian vault for markdown files
2. **Parsing** notes (frontmatter, content, wikilinks, tags)
3. **Chunking** long notes into smaller segments
4. **Computing embeddings** for each chunk
5. **Creating graph edges** (wikilinks + similarity proposals)
6. **Storing everything** in the database

Think of it as the "crawler" that processes your Obsidian vault and makes it searchable.

---

## File Structure

```python
class VaultIndexer:
    # Main method
    - index()                          # Index all .md files in vault
    
    # File processing
    - _process_file(file_path)         # Process single markdown file
    
    # Metadata extraction
    - _extract_wikilinks(text)         # Extract [[wikilinks]]
    - _extract_inline_tags(text)       # Extract #tags
    - _extract_tags(frontmatter, body) # Combine all tags
    
    # Graph creation
    - _create_wikilink_edges(node, links)  # Create explicit edges
    - _compute_similarity(emb1, emb2)      # Cosine similarity
    - _generate_similarity_proposals(node, embedding)  # Create proposals
```

---

## Class: `VaultIndexer`

### Constructor: `__init__(vault_path, db, embedding_model)`

**Parameters:**
- `vault_path`: Path to Obsidian vault directory
- `db`: `AsyncDatabase` instance
- `embedding_model`: `EmbeddingModel` instance

**What it does:**
- Stores paths and dependencies
- Expands user path (`~` → `/home/user/`)

**Example:**
```python
db = AsyncDatabase("./data/minime.db")
embedding_model = EmbeddingModel()

indexer = VaultIndexer(
    vault_path="~/Documents/my-vault",
    db=db,
    embedding_model=embedding_model
)
```

---

## Method: `async index() -> List[VaultNode]`

**Purpose**: Index all markdown files in the vault.

**What it does:**
1. Checks if vault path exists
2. Finds all `.md` files recursively
3. Processes each file (with error handling)
4. Returns list of indexed nodes

**Example:**
```python
nodes = await indexer.index()
print(f"Indexed {len(nodes)} notes")
```

**Error handling:**
- Skips corrupt files (logs warning, continues)
- Handles empty vault (returns empty list)
- Handles individual file errors (continues with next file)

---

## Method: `async _process_file(file_path: Path) -> Optional[VaultNode]`

**Purpose**: Process a single markdown file into a `VaultNode`.

**Steps:**

### 1. Read File
```python
content = file_path.read_text(encoding="utf-8")
```

### 2. Parse Frontmatter
Uses `python-frontmatter` library:
```python
post = frontmatter.loads(content)
frontmatter_data = post.metadata  # YAML frontmatter
body = post.content               # Note body (without frontmatter)
```

**Frontmatter example:**
```yaml
---
title: My Note
tags: [ai, memory]
domain: coding
scope: global
---

# Note Content
This is the body...
```

### 3. Extract Metadata
```python
title = frontmatter_data.get("title") or file_path.stem
tags = self._extract_tags(frontmatter_data, body)
domain = frontmatter_data.get("domain")
scope = frontmatter_data.get("scope", "global")
links = self._extract_wikilinks(body)
```

### 4. Generate Node ID
```python
relative_path = str(file_path.relative_to(self.vault_path))
node_id = hashlib.md5(relative_path.encode()).hexdigest()
```

**Why MD5 hash?**
- Stable: Same path = same ID
- Unique: Different paths = different IDs
- Works even if notes are renamed (path-based)

### 5. Create VaultNode
```python
node = VaultNode(
    node_id=node_id,
    path=relative_path,
    title=title,
    frontmatter=frontmatter_data,
    tags=tags,
    domain=domain,
    scope=scope,
    links=links,
    backlinks=[],  # Computed later
    created_at=datetime.now(),
    updated_at=datetime.now(),
    embedding_ref=None  # Set after chunking
)
```

### 6. Chunk Note
```python
chunks = chunk_note(body)  # Split into overlapping chunks
```

### 7. Compute Embeddings
```python
embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
```

**Batch encoding:**
- Processes all chunks at once
- More efficient than one-by-one
- Returns list of embeddings (one per chunk)

### 8. Store Chunks
```python
primary_chunk_id = None
for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
    chunk_id = f"{node_id}_chunk_{idx}"
    if primary_chunk_id is None:
        primary_chunk_id = chunk_id  # First chunk is primary
    
    chunk = MemoryChunk(
        chunk_id=chunk_id,
        node_id=node_id,
        content=chunk_text,
        embedding=embedding,
        metadata={"chunk_index": idx, "total_chunks": len(chunks)},
        position=idx
    )
    
    await self.db.insert_chunk(chunk)
```

### 9. Update Node
```python
node.embedding_ref = primary_chunk_id  # Reference to first chunk
await self.db.insert_node(node)
```

### 10. Create Graph Edges
```python
await self._create_wikilink_edges(node, links)  # Explicit edges
await self._generate_similarity_proposals(node, embeddings[0])  # Similarity
```

---

## Method: `_extract_wikilinks(text: str) -> List[str]`

**Purpose**: Extract all `[[wikilink]]` patterns from text.

**Pattern**: `r"\[\[([^\]]+)\]\]"`

**What it matches:**
- `[[note-name]]` → `"note-name"`
- `[[note-name|display text]]` → `"note-name"` (extracts target only)

**Example:**
```python
text = "See [[other-note]] and [[reference|this reference]]"
links = indexer._extract_wikilinks(text)
# Returns: ["other-note", "reference"]
```

---

## Method: `_extract_inline_tags(text: str) -> List[str]`

**Purpose**: Extract inline `#tags` from text.

**Pattern**: `r"#([a-zA-Z0-9_/-]+)"`

**What it matches:**
- `#tag` → `"tag"`
- `#tag/subtag` → `"tag/subtag"`
- `#tag-name` → `"tag-name"`

**Example:**
```python
text = "This is about #ai and #machine-learning"
tags = indexer._extract_inline_tags(text)
# Returns: ["ai", "machine-learning"]
```

---

## Method: `_extract_tags(frontmatter_data, body) -> List[str]`

**Purpose**: Combine tags from frontmatter and inline tags.

**What it does:**
1. Extracts tags from frontmatter (handles both list and string)
2. Extracts inline tags from body
3. Combines and deduplicates
4. Normalizes (lowercase, strip)

**Example:**
```yaml
# frontmatter
tags: [ai, memory]

# body
This note discusses #machine-learning and #ai concepts.
```

**Result**: `["ai", "memory", "machine-learning"]` (deduplicated)

---

## Method: `async _create_wikilink_edges(node, links) -> None`

**Purpose**: Create explicit graph edges for wikilinks.

**What it does:**
1. For each wikilink target:
   - Generates target node_id (hash of link name)
   - Creates edge with `edge_type="wikilink"`
   - Sets `weight=1.0`, `confidence=1.0`
   - Sets `is_approved=True` (wikilinks are explicit)

**Note**: Target node might not exist yet (created edge anyway, resolved later).

**Example:**
```python
node.links = ["other-note", "reference"]

await indexer._create_wikilink_edges(node, node.links)
# Creates 2 edges in graph_edges table
```

---

## Method: `_compute_similarity(embedding1, embedding2) -> float`

**Purpose**: Compute cosine similarity between two embeddings.

**Formula**: `cosine_similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))`

**Returns**: Float between -1.0 and 1.0
- `1.0` = Identical meaning
- `0.7` = Similar (threshold for proposals)
- `0.0` = Unrelated
- `-1.0` = Opposite meaning

**Implementation:**
```python
vec1 = np.array(embedding1)
vec2 = np.array(embedding2)

dot_product = np.dot(vec1, vec2)
norm1 = np.linalg.norm(vec1)
norm2 = np.linalg.norm(vec2)

similarity = dot_product / (norm1 * norm2)
return float(similarity)
```

**Example:**
```python
emb1 = [0.2, 0.8, -0.1, ...]
emb2 = [0.3, 0.7, -0.2, ...]

similarity = indexer._compute_similarity(emb1, emb2)
# Returns: 0.85 (very similar!)
```

---

## Method: `async _generate_similarity_proposals(node, embedding, threshold=0.7) -> None`

**Purpose**: Generate similarity-based edge proposals for related notes.

**What it does:**
1. Gets all existing node embeddings from database
2. Computes similarity with current note's embedding
3. If similarity > threshold:
   - Creates proposal with `edge_type="similar"`
   - Sets `weight` and `confidence` to similarity score
   - Stores in `graph_proposals` table

**Parameters:**
- `node`: Current node being indexed
- `embedding`: Primary chunk embedding
- `threshold`: Minimum similarity (default: 0.7)

**Limits:**
- Maximum 10 proposals per note (to avoid spam)

**Example:**
```python
# New note has embedding
new_embedding = [0.2, 0.8, ...]

# Compare with all existing notes
existing_embeddings = await db.get_all_node_embeddings()
# Returns: [("note1", [0.3, 0.7, ...]), ("note2", [0.1, 0.9, ...]), ...]

for other_node_id, other_embedding in existing_embeddings:
    similarity = compute_similarity(new_embedding, other_embedding)
    if similarity > 0.7:
        # Create proposal!
```

**Proposal creation:**
```python
proposal = GraphEdge(
    edge_id=f"{node.node_id}_{other_node_id}_similar",
    source_node_id=node.node_id,
    target_node_id=other_node_id,
    edge_type="similar",
    weight=similarity,
    rationale=f"Semantic similarity: {similarity:.3f}",
    confidence=similarity,
    is_approved=False,  # Needs approval
    created_at=datetime.now()
)

await db.insert_proposal(proposal)
```

---

## Key Concepts

### 1. Recursive File Scanning

Uses `Path.rglob("*.md")` to find all markdown files:
- Searches all subdirectories
- Handles nested vault structure
- Efficient for large vaults

### 2. Frontmatter Parsing

Uses `python-frontmatter` library:
- Handles YAML frontmatter
- Separates metadata from content
- Gracefully handles missing frontmatter

### 3. Chunking Strategy

Notes are split into overlapping chunks:
- Default: 512 tokens per chunk, 128 overlap
- Enables fine-grained retrieval
- Overlap preserves context at boundaries

### 4. Primary Chunk

First chunk is the "primary chunk":
- Used for similarity comparisons
- Stored in `node.embedding_ref`
- Represents overall note meaning

### 5. Two Types of Edges

1. **Explicit (wikilinks)**: User-created, auto-approved
2. **Implicit (similarity)**: AI-proposed, requires approval

### 6. Similarity Threshold

Default threshold: 0.7
- Below 0.7: Not similar enough
- 0.7-0.9: Similar, needs review
- Above 0.9: Very similar, could auto-approve (future)

---

## Error Handling

### File Reading Errors
```python
try:
    content = file_path.read_text(encoding="utf-8")
except Exception as e:
    print(f"Warning: Could not read {file_path}: {e}")
    return None  # Skip this file
```

### Frontmatter Parsing Errors
```python
try:
    post = frontmatter.loads(content)
except Exception as e:
    frontmatter_data = {}  # No frontmatter
    body = content  # Use entire content as body
```

### Processing Errors
```python
try:
    node = await self._process_file(md_file)
except Exception as e:
    print(f"Warning: Failed to process {md_file}: {e}")
    continue  # Continue with next file
```

---

## Integration

### With Database
```python
# Stores nodes, chunks, edges
await db.insert_node(node)
await db.insert_chunk(chunk)
await db.insert_edge(edge)
await db.insert_proposal(proposal)
```

### With EmbeddingModel
```python
# Computes embeddings for chunks
embeddings = embedding_model.encode(chunks)
```

### With Chunking
```python
# Splits notes into chunks
chunks = chunk_note(body, max_tokens=512, overlap=128)
```

---

## Performance Considerations

### Batch Embedding

Uses `embedding_model.encode(chunks)` for batch processing:
- More efficient than encoding one-by-one
- Processes all chunks at once
- Faster for large notes

### Similarity Search

`get_all_node_embeddings()` loads all embeddings:
- Works well for <1000 notes
- For larger vaults, consider vector database
- In-memory comparison is fast for small datasets

### Incremental Indexing (Future)

Current implementation re-indexes all files:
- For MVP, this is fine
- Future: Compare modification times
- Only process new/changed files

---

## Example Workflow

```python
# 1. Initialize
db = AsyncDatabase("./data/minime.db")
embedding_model = EmbeddingModel()
indexer = VaultIndexer("~/my-vault", db, embedding_model)

# 2. Index vault
nodes = await indexer.index()
print(f"Indexed {len(nodes)} notes")

# 3. Check proposals
proposals = await db.get_pending_proposals()
print(f"Found {len(proposals)} similarity proposals")

# 4. Approve proposals (via CLI or code)
for proposal in proposals:
    if proposal.confidence > 0.8:
        await db.approve_proposal(proposal.edge_id)
```

---

## Summary

The `VaultIndexer` is the **ingestion layer** for MiniMe's memory system:

1. **Scans vault** - Finds all markdown files
2. **Parses notes** - Extracts metadata, content, links
3. **Chunks content** - Splits long notes into segments
4. **Computes embeddings** - Creates vector representations
5. **Creates graph** - Builds explicit and implicit connections
6. **Stores everything** - Saves to database for retrieval

Without the indexer, MiniMe would have no memory. It transforms your Obsidian vault into a searchable, graph-connected knowledge base that powers semantic search and context retrieval.

