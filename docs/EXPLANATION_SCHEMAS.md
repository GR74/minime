# Explanation: `minime/schemas.py`

This file defines all the data structures (schemas) used throughout the MiniMe system using Pydantic models.

## Overview

`schemas.py` is the **single source of truth** for all data structures in MiniMe. It uses Pydantic v2 to define:
- Data models (what data looks like)
- Validation rules (what data is valid)
- Default values (what happens if data is missing)

Think of it as the "database schema" but for Python objects instead of SQL tables.

---

## What is Pydantic?

**Pydantic** is a Python library that:
- Validates data automatically
- Provides type hints
- Converts data between formats (dict, JSON, etc.)
- Gives helpful error messages

### Simple Example

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# Valid - works fine
person = Person(name="Alice", age=30)

# Invalid - raises error
person = Person(name="Bob", age="thirty")  # Error: age must be int
```

---

## File Structure

The file is organized into sections:

1. **Identity Layer** - Your personal principles
2. **Memory Layer** - Obsidian vault notes and graph
3. **Retrieval** - Search queries and results
4. **Masking System** - How to weight different information
5. **Agent System** - LLM agents and tools
6. **Actions & Approval** - Proposed actions (write file, run command)
7. **Tracing & Feedback** - Logging and learning
8. **Configuration** - System settings

---

## Section 1: Identity Layer

### `IdentityPrinciple`

**Purpose**: Represents one of your personal principles (e.g., "Modularity", "Clarity")

**Fields:**
```python
class IdentityPrinciple(BaseModel):
    id: str                    # Unique identifier: "modularity"
    name: str                  # Display name: "Modularity"
    description: str            # What it means: "Break systems into modules"
    vector: List[float]         # Embedding (384 numbers) - the ML part!
    magnitude: float = 1.0     # How important (0.0 to 1.0)
    decay_rate: float = 0.05   # How fast it adapts (0.0 = frozen, 1.0 = plastic)
    scope: str = "global"      # Where it applies: "global" | "domain:biotech"
    tags: List[str]            # Categories: ["architecture", "code"]
    created_at: datetime       # When created
    updated_at: datetime       # Last modified
```

**Key Concepts:**

1. **`vector`**: The embedding (384 numbers) that represents the meaning
   - Created by `EmbeddingModel.encode_single()`
   - Used for similarity search

2. **`magnitude`**: Importance weight
   - `1.0` = Very important
   - `0.5` = Somewhat important
   - Used to weight principles when assembling context

3. **`decay_rate`**: How much the principle can change
   - `0.0` = Never changes (frozen)
   - `1.0` = Changes quickly (plastic)
   - `0.05` = Changes slowly (default)

4. **`scope`**: Where the principle applies
   - `"global"` = Applies everywhere
   - `"domain:biotech"` = Only for biotech tasks
   - Allows domain-specific principles

**Example:**
```python
principle = IdentityPrinciple(
    id="modularity",
    name="Modularity",
    description="Break systems into independent, testable modules",
    vector=[0.2, 0.8, -0.1, ...],  # 384 numbers
    magnitude=1.0,
    decay_rate=0.05,
    scope="global",
    tags=["architecture", "code"]
)
```

---

### `GlobalIdentityMatrix`

**Purpose**: Container for all your identity principles (the "P_global" matrix)

**Fields:**
```python
class GlobalIdentityMatrix(BaseModel):
    principles: List[IdentityPrinciple]  # All your principles
    version: str = "1.0"                  # Schema version
```

**Methods:**

1. **`to_dict() -> Dict[str, Any]`**
   - Returns: `{"principle_id": [vector], ...}`
   - Used to get all embeddings as a dictionary

2. **`get_principle(principle_id: str) -> Optional[IdentityPrinciple]`**
   - Finds a principle by ID
   - Returns `None` if not found

3. **`add_principle(principle: IdentityPrinciple)`**
   - Adds a new principle to the matrix

4. **`update_principle(principle_id: str, updates: Dict) -> bool`**
   - Updates a principle's fields
   - Returns `True` if successful, `False` if not found

**Example:**
```python
matrix = GlobalIdentityMatrix(
    principles=[
        IdentityPrinciple(id="modularity", ...),
        IdentityPrinciple(id="clarity", ...),
    ],
    version="1.0"
)

# Get all embeddings
embeddings_dict = matrix.to_dict()
# {"modularity": [0.2, 0.8, ...], "clarity": [0.3, 0.7, ...]}

# Find a principle
principle = matrix.get_principle("modularity")
```

---

## Section 2: Memory Layer

### `VaultNode`

**Purpose**: Represents one note in your Obsidian vault

**Fields:**
```python
class VaultNode(BaseModel):
    node_id: str              # Unique ID (hash of path)
    path: str                 # File path: "notes/my-note.md"
    title: str                # Note title
    frontmatter: Dict         # YAML frontmatter from note
    tags: List[str]           # Tags: ["coding", "python"]
    domain: Optional[str]     # Domain: "biotech" | None
    scope: str = "global"     # Scope: "global" | "project:myproj"
    links: List[str]          # Wikilinks: ["[[other-note]]"]
    backlinks: List[str]      # Notes that link to this one
    created_at: datetime      # File creation time
    updated_at: datetime      # File modification time
    embedding_ref: Optional[str]  # Reference to embedding in DB
```

**Key Concepts:**

1. **`frontmatter`**: YAML metadata at top of note
   ```markdown
   ---
   title: My Note
   domain: biotech
   tags: [research, protein]
   ---
   ```

2. **`links`**: Wikilinks in the note
   - `[[other-note]]` → stored as `"other-note"`

3. **`backlinks`**: Reverse links
   - If Note A links to Note B, then Note B has Note A in backlinks

4. **`embedding_ref`**: Pointer to embedding in database
   - Embeddings stored separately (they're large)
   - Reference used to retrieve when needed

---

### `GraphEdge`

**Purpose**: Represents a connection between two notes

**Fields:**
```python
class GraphEdge(BaseModel):
    edge_id: str              # Unique ID
    source_node_id: str      # From this note
    target_node_id: str      # To this note
    edge_type: str            # "wikilink" | "similar" | "related"
    weight: float = 1.0       # Strength (0.0 to 1.0)
    rationale: str            # Why this edge exists
    confidence: float = 1.0    # How confident (0.0 to 1.0)
    is_approved: bool = True  # User approved?
    created_at: datetime
    approved_at: Optional[datetime]
```

**Edge Types:**
- `"wikilink"`: Explicit link in note (`[[note]]`)
- `"similar"`: Similar embeddings (ML-detected)
- `"related"`: Related by metadata (same tags, domain)

---

### `MemoryChunk`

**Purpose**: A piece of a note (for retrieval)

**Fields:**
```python
class MemoryChunk(BaseModel):
    chunk_id: str             # Unique ID
    node_id: str              # Which note it's from
    content: str              # The actual text
    embedding: List[float]     # Embedding of this chunk
    metadata: Dict            # Domain, tags, scope, etc.
    position: int = 0         # Byte offset in note
```

**Why chunks?**
- Notes can be long (1000+ words)
- LLMs have token limits
- We split notes into smaller chunks
- Each chunk gets its own embedding

**Example:**
```python
# Original note (500 words)
note = "Long note about protein folding..."

# Split into chunks
chunks = [
    MemoryChunk(chunk_id="1", node_id="note-123", content="First 200 words...", ...),
    MemoryChunk(chunk_id="2", node_id="note-123", content="Next 200 words...", ...),
    MemoryChunk(chunk_id="3", node_id="note-123", content="Last 100 words...", ...),
]
```

---

## Section 3: Retrieval

### `RetrievalQuery`

**Purpose**: A request to search memory

**Fields:**
```python
class RetrievalQuery(BaseModel):
    query: str                        # Search text
    query_embedding: Optional[List[float]]  # Pre-computed embedding
    filters: Optional[Dict]           # Domain, tags, scope filters
    k: int = 5                        # How many results
    include_graph_neighbors: bool = True  # Include linked notes?
    max_context_tokens: int = 2000    # Token limit
```

**Example:**
```python
query = RetrievalQuery(
    query="How should I structure my code?",
    filters={"domain": "coding", "tags": ["architecture"]},
    k=5,
    include_graph_neighbors=True
)
```

---

### `RetrievalResult`

**Purpose**: Results from a memory search

**Fields:**
```python
class RetrievalResult(BaseModel):
    chunks: List[MemoryChunk]        # Found chunks
    scores: List[float]               # Similarity scores
    graph_traversal_depth: int = 0    # How deep we searched
    blocked_count: int = 0            # Chunks filtered out
    context_tokens_used: int = 0      # Tokens in result
    is_cold_start: bool = False      # Vault empty?
```

**Key Concepts:**

1. **`scores`**: Similarity scores (0.0 to 1.0)
   - Higher = more relevant
   - Used to rank results

2. **`is_cold_start`**: True if vault is empty
   - System works even with no notes
   - Falls back to general knowledge

---

## Section 4: Masking System

### `MaskWeights`

**Purpose**: Controls how information is weighted and used

**Fields:**
```python
class MaskWeights(BaseModel):
    # Retrieval settings
    retrieval_k: int = 5              # How many results
    retrieval_min_similarity: float = 0.5  # Minimum similarity
    block_weights: Dict[str, float]   # Weight per context block
    
    # Generation settings
    temperature: float = 0.7          # LLM creativity (0.0-1.0)
    verbosity: int = 3                # How detailed (1-5)
    rigor: int = 3                    # How strict (1-5)
    
    # Agent routing
    agent_routing_bias: Dict[str, float]  # Which agents to prefer
    
    # Graph settings
    graph_proximity_weight: float = 0.2  # Boost for linked notes
    graph_max_hops: int = 2           # How far to traverse
    
    # Mask strengths
    global_mask_strength: float = 1.0
    domain_mask_strength: float = 0.0
    task_mask_strength: float = 0.0
    agent_mask_strength: float = 0.0
```

**What is "masking"?**
- Think of it as a "filter" or "lens"
- Controls what information is emphasized
- Different tasks need different information

**Example:**
```python
# For a coding task
weights = MaskWeights(
    retrieval_k=10,              # Get more results
    temperature=0.3,             # More focused (less creative)
    rigor=5,                     # Very strict
    agent_routing_bias={"builder": 0.8}  # Prefer builder agent
)
```

---

## Section 5: Agent System

### `AgentSpec`

**Purpose**: Definition of an LLM agent

**Fields:**
```python
class AgentSpec(BaseModel):
    name: str                    # "architect" | "builder" | "critic"
    purpose: str                 # What this agent does
    io_schema: Dict              # Input/output schemas
    system_prompt: str           # Instructions for the agent
    model: str = "gpt-4"         # Which LLM to use
    tools_allowed: List[str]     # What tools it can use
    routing_hints: Dict          # When to use this agent
    constraints: List[str]       # What it must NOT do
    temperature: float = 0.7     # Creativity level
    max_tokens: int = 2000       # Response length limit
    created_at: datetime
    approved_at: Optional[datetime]
```

**Example:**
```python
architect = AgentSpec(
    name="architect",
    purpose="Design system architecture",
    system_prompt="You are a software architect...",
    model="gpt-4",
    tools_allowed=["propose_plan"],
    constraints=["No implementation details", "Focus on structure"],
    temperature=0.3  # More focused
)
```

---

## Section 6: Configuration

### `MiniMeConfig`

**Purpose**: System-wide configuration

**Fields:**
```python
class MiniMeConfig(BaseModel):
    vault_path: str              # Where your Obsidian vault is
    db_path: str                 # SQLite database location
    embedding_model: str         # Which embedding model
    default_provider: str        # Which LLM provider
    trace_dir: str               # Where to store logs
    config_dir: str              # Where config files are
    max_context_tokens: int      # Token limit
    safe_paths: List[str]        # Safe to write here
    system_paths: List[str]      # Never write here
    # ... and more
```

**Methods:**

1. **`load_from_file(path: str) -> MiniMeConfig`**
   - Loads config from YAML file
   - Merges with defaults

2. **`save_to_file(path: str)`**
   - Saves config to YAML file

---

## Key Pydantic Features Used

### 1. Default Values
```python
magnitude: float = 1.0  # Default if not provided
```

### 2. Optional Fields
```python
domain: Optional[str] = None  # Can be None
```

### 3. Field Factories
```python
tags: List[str] = Field(default_factory=list)  # New list each time
```

### 4. Validation
Pydantic automatically validates:
- Types (str, int, float, etc.)
- Required vs optional
- Custom validators (can be added)

---

## How Schemas Are Used

### 1. Data Validation
```python
# This will raise an error
principle = IdentityPrinciple(
    id="test",
    name="Test",
    description="Test",
    vector="not a list"  # Error: must be List[float]
)
```

### 2. Serialization
```python
# Convert to dict
data = principle.model_dump()

# Convert to JSON
json_str = principle.model_dump_json()

# Convert from dict
principle = IdentityPrinciple(**data)
```

### 3. Type Hints
```python
def process_principle(p: IdentityPrinciple) -> str:
    # IDE knows p has .name, .vector, etc.
    return p.name
```

---

## Summary

`schemas.py` is the **foundation** of MiniMe's data structure:

1. **Defines all data models** (what data looks like)
2. **Validates automatically** (catches errors early)
3. **Provides type hints** (better IDE support)
4. **Enables serialization** (save/load from files, databases)

Without schemas, you'd have:
- No validation (bugs catch later)
- No type hints (harder to code)
- Inconsistent data structures
- More errors

With schemas, you get:
- ✅ Automatic validation
- ✅ Type safety
- ✅ Clear data contracts
- ✅ Easy serialization

