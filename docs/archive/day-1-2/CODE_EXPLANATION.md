# MiniMe Code Structure & ML Concepts Explained

## Overview

MiniMe uses **embeddings** to represent text as vectors so it can compare and retrieve similar content. The code is organized into layers: schemas (data models), identity (principles), memory (embeddings), and config (settings).

---

## 1. `minime/schemas.py` - Data Models

### Purpose
Defines all data structures using Pydantic for validation and type safety.

### Structure
- **Identity models**: `IdentityPrinciple`, `GlobalIdentityMatrix`
- **Memory models**: `VaultNode`, `GraphEdge`, `MemoryChunk`
- **Agent models**: `AgentSpec`, `ToolDefinition`
- **Config model**: `MiniMeConfig`

### ML Concept: Vector Embeddings
```python
vector: List[float] = Field(default_factory=list)  # embedding
```
- Text is converted into a fixed-size list of numbers (e.g., 384 floats).
- Similar texts have similar vectors (close in a high-dimensional space).
- Example: "modularity" and "modular design" are close; "modularity" and "banana" are far.

### Example
```python
# An IdentityPrinciple stores:
- id: "modularity"
- name: "Modularity" 
- description: "Break systems into independent modules"
- vector: [0.23, -0.45, 0.67, ...]  # 384 numbers representing the meaning
- magnitude: 1.0  # How important this principle is
```

---

## 2. `minime/memory/embeddings.py` - The ML Engine

### Purpose
Converts text into embedding vectors using a pre-trained model.

### Structure
```python
class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2")
    def encode(texts: List[str]) -> List[List[float]]  # Batch processing
    def encode_single(text: str) -> List[float]  # Single text
```

### ML Concept: Sentence Transformers
- Uses a pre-trained neural network (`all-MiniLM-L6-v2`) that maps text to vectors.
- **Lazy loading**: model loads on first use.
- **Deterministic**: same text → same vector.

### How It Works
```python
# Input: "Modularity: Break systems into independent modules"
# Output: [0.23, -0.45, 0.67, 0.12, ...]  # 384 numbers

embedding_model = EmbeddingModel()
vector = embedding_model.encode_single("I love modular code")
# This vector can now be compared with other vectors!
```

### Why 384 Dimensions?
- More dimensions capture more nuance.
- 384 is a balance between quality and speed.
- Similar texts cluster in this space.

---

## 3. `minime/identity/loader.py` - Loading Your Principles

### Purpose
Reads identity principles from YAML and computes their embeddings.

### Flow
1. Read YAML config
2. Extract principles
3. Compute embeddings for each principle
4. Return `GlobalIdentityMatrix`

### Code Flow
```python
# Step 1: Load YAML
data = yaml.safe_load(f)  # {"principles": [...]}

# Step 2: For each principle
for p_data in principles_data:
    # Step 3: Create embedding text
    embedding_text = f"{name}: {description}"
    # "Modularity: Break systems into independent modules"
    
    # Step 4: Convert to vector
    vector = embedding_model.encode_single(embedding_text)
    # [0.23, -0.45, 0.67, ...]
    
    # Step 5: Create IdentityPrinciple object
    principle = IdentityPrinciple(
        id="modularity",
        name="Modularity",
        vector=vector,  # The ML magic happens here!
        ...
    )
```

### ML Concept: Semantic Representation
- The model encodes meaning, not just keywords.
- "Modularity" and "breaking into modules" map to similar vectors.
- Enables similarity search later.

---

## 4. `minime/identity/principles.py` - Managing Principles

### Purpose
Manages the collection of identity principles.

### Structure
```python
class IdentityManager:
    def get_principle(id: str) -> IdentityPrinciple
    def add_principle(...) -> IdentityPrinciple
    def update_principle(id: str, updates: Dict) -> bool
    def to_dict() -> Dict[str, List[float]]  # Returns all vectors
```

### How It Works
- Wraps `GlobalIdentityMatrix` with helper methods.
- `to_dict()` returns a mapping of IDs to vectors for similarity operations.

### Example
```python
manager = IdentityManager()
manager.add_principle(
    name="Modularity",
    description="Break systems into modules",
    vector=[0.23, -0.45, ...]  # Pre-computed embedding
)

# Later, you can get all vectors for similarity search
all_vectors = manager.to_dict()
# {"modularity": [0.23, -0.45, ...], "clarity": [0.12, 0.34, ...]}
```

---

## 5. `minime/config.py` - Configuration Management

### Purpose
Loads and manages system configuration from YAML.

### Functions
- `load_config()`: Loads config from YAML
- `create_default_config_file()`: Creates default YAML template

### Structure
```python
def load_config(config_path: Optional[str] = None) -> MiniMeConfig:
    # 1. Check if file exists
    # 2. Load YAML
    # 3. Parse into MiniMeConfig object
    # 4. Return validated config
```

### What `MiniMeConfig` Contains
- System paths (vault, database, logs)
- ML settings (embedding model name, cache size)
- Behavior settings (risk levels, auto-approval)

---

## How It All Works Together

### Complete Flow Example

```python
# 1. Load configuration
config = load_config("./config/identity.yaml")

# 2. Load identity principles (this uses embeddings!)
identity_matrix = load_identity_from_yaml(
    config_path="./config/identity.yaml",
    embedding_model=EmbeddingModel()  # Creates the ML model
)

# What happens inside:
# - Reads YAML: "Modularity: Break systems into modules"
# - Converts to vector: [0.23, -0.45, 0.67, ...]
# - Stores in IdentityPrinciple object

# 3. Use the identity
manager = IdentityManager(identity_matrix)
principle = manager.get_principle("modularity")
print(principle.vector)  # [0.23, -0.45, 0.67, ...]
```

---

## ML Concepts Explained Simply

### 1. Embeddings (Vector Representations)
- **What**: Text → fixed-size list of numbers.
- **Why**: Similar meaning → similar vectors.
- **How**: Enables mathematical similarity.

### 2. Vector Space
- Each text is a point in a high-dimensional space.
- Distance ≈ semantic similarity.
- Clustering groups related concepts.

### 3. Pre-trained Models
- `all-MiniLM-L6-v2` is already trained on large text.
- No training needed; use it directly.
- Fast and deterministic.

### 4. Semantic Similarity
- Compare vectors with cosine similarity or Euclidean distance.
- Example: "modularity" and "modular design" are close.

### 5. Why This Matters for MiniMe
- Identity principles are stored as vectors.
- Later, compare user queries to principles to find relevant ones.
- Retrieve similar notes from the vault using vector search.

---

## Key Design Patterns

### 1. Lazy Loading
```python
@property
def model(self) -> SentenceTransformer:
    if self._model is None:
        self._model = SentenceTransformer(...)  # Only loads when needed
    return self._model
```
- Model loads on first use to avoid startup delay.

### 2. Graceful Degradation
```python
if not config_file.exists():
    return GlobalIdentityMatrix(principles=[], version="1.0")  # Empty but valid
```
- Handles missing files without crashing.

### 3. Type Safety with Pydantic
- Validates data at runtime.
- Catches errors early.

---

## Summary

- **`schemas.py`**: Data models with vector fields.
- **`embeddings.py`**: Converts text to vectors.
- **`identity/loader.py`**: Loads principles and computes embeddings.
- **`identity/principles.py`**: Manages the collection of principles.
- **`config.py`**: Loads system settings.

The ML core is embeddings: text becomes vectors, enabling similarity search and retrieval. The rest of the system uses these vectors to personalize behavior.

---

## File Structure Reference

```
minime/
├── schemas.py              # All data models (Pydantic)
├── config.py               # Configuration loading
├── identity/
│   ├── principles.py       # IdentityManager class
│   └── loader.py           # Load from YAML + compute embeddings
└── memory/
    └── embeddings.py       # EmbeddingModel (ML engine)
```

---

## Next Steps

To understand how these pieces connect:
1. See how embeddings are used in retrieval (coming in Day 3-4)
2. Learn about vector similarity search
3. Understand how the mask system uses these vectors

