# Explanation: Identity Layer

This document explains the identity layer files:
- `minime/identity/principles.py` - Identity manager
- `minime/identity/loader.py` - Loading from YAML

## Overview

The identity layer stores and manages your **personal principles** - the values, preferences, and guidelines that make MiniMe "you". These principles are converted to embeddings (vectors) so MiniMe can understand their meaning and apply them when answering questions.

---

## File 1: `minime/identity/principles.py`

### Purpose

Provides the `IdentityManager` class to manage identity principles programmatically.

### Class: `IdentityManager`

**Purpose**: High-level interface for managing identity principles.

#### Constructor: `__init__(matrix: Optional[GlobalIdentityMatrix] = None)`

**What it does:**
- Creates an identity manager
- Optionally takes an existing `GlobalIdentityMatrix`
- If None, creates an empty matrix

**Example:**
```python
# Create empty manager
manager = IdentityManager()

# Or use existing matrix
matrix = GlobalIdentityMatrix(principles=[...])
manager = IdentityManager(matrix=matrix)
```

---

#### Method: `get_principle(principle_id: str) -> Optional[IdentityPrinciple]`

**Purpose**: Find a principle by its ID.

**Parameters:**
- `principle_id`: The unique ID (e.g., `"modularity"`)

**Returns:**
- `IdentityPrinciple` if found
- `None` if not found

**Example:**
```python
manager = IdentityManager()
principle = manager.get_principle("modularity")
if principle:
    print(principle.name)  # "Modularity"
```

---

#### Method: `add_principle(...) -> IdentityPrinciple`

**Purpose**: Add a new principle to the identity matrix.

**Parameters:**
```python
def add_principle(
    self,
    name: str,                    # "Modularity"
    description: str,              # "Break systems into modules"
    vector: list[float],           # Embedding vector (384 numbers)
    magnitude: float = 1.0,        # Importance (default: 1.0)
    decay_rate: float = 0.05,      # Adaptation rate (default: 0.05)
    scope: str = "global",         # Where it applies (default: "global")
    tags: Optional[list[str]] = None,  # Categories (default: empty)
    principle_id: Optional[str] = None,  # Auto-generated if None
) -> IdentityPrinciple:
```

**What it does:**
1. Generates ID from name if not provided
2. Creates `IdentityPrinciple` object
3. Adds to the matrix
4. Returns the created principle

**Example:**
```python
manager = IdentityManager()

# Add a principle
principle = manager.add_principle(
    name="Modularity",
    description="Break systems into independent, testable modules",
    vector=[0.2, 0.8, -0.1, ...],  # 384 numbers from embedding
    magnitude=1.0,
    scope="global",
    tags=["architecture", "code"]
)
# ID auto-generated as "modularity"
```

**Note**: You need to provide the `vector` (embedding). Usually you'd get this from `EmbeddingModel.encode_single()`.

---

#### Method: `update_principle(principle_id: str, updates: Dict[str, Any]) -> bool`

**Purpose**: Update fields of an existing principle.

**Parameters:**
- `principle_id`: ID of principle to update
- `updates`: Dictionary of fields to update

**Returns:**
- `True` if successful
- `False` if principle not found

**Example:**
```python
manager = IdentityManager()

# Update magnitude
success = manager.update_principle(
    "modularity",
    {"magnitude": 0.8}  # Make it less important
)

if success:
    print("Updated!")
```

**What can be updated?**
- `magnitude`: Change importance
- `description`: Update description (would need new embedding)
- `tags`: Change categories
- `scope`: Change where it applies
- Any field except `id` and timestamps

---

#### Method: `to_dict() -> Dict[str, Any]`

**Purpose**: Get all principles as a dictionary mapping IDs to embedding vectors.

**Returns:**
```python
{
    "modularity": [0.2, 0.8, -0.1, ...],  # 384 numbers
    "clarity": [0.3, 0.7, -0.2, ...],     # 384 numbers
    ...
}
```

**Use case**: When you need all embeddings for similarity search.

**Example:**
```python
manager = IdentityManager()
# ... add principles ...

embeddings_dict = manager.to_dict()
# Use for similarity search
```

---

#### Method: `get_all_principles() -> list[IdentityPrinciple]`

**Purpose**: Get list of all principles.

**Returns:**
- List of `IdentityPrinciple` objects

**Example:**
```python
manager = IdentityManager()
principles = manager.get_all_principles()

for p in principles:
    print(f"{p.name}: {p.description}")
```

---

## File 2: `minime/identity/loader.py`

### Purpose

Loads identity principles from YAML configuration files and computes their embeddings.

### Function: `load_identity_from_yaml(...) -> GlobalIdentityMatrix`

**Purpose**: Load principles from YAML and compute embeddings automatically.

**Signature:**
```python
def load_identity_from_yaml(
    config_path: str,
    embedding_model: Optional[EmbeddingModel] = None,
) -> GlobalIdentityMatrix:
```

**Parameters:**
- `config_path`: Path to YAML config file
- `embedding_model`: Optional embedding model (creates new one if None)

**Returns:**
- `GlobalIdentityMatrix` with loaded principles and computed embeddings

**How it works:**

#### Step 1: Check if file exists
```python
config_file = Path(config_path)
if not config_file.exists():
    # Return empty matrix (graceful handling)
    return GlobalIdentityMatrix(principles=[], version="1.0")
```

**Why graceful?** System works even if config file doesn't exist yet.

---

#### Step 2: Load YAML
```python
with open(config_file, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
```

**What's in the YAML?**
```yaml
principles:
  - id: modularity
    name: Modularity
    description: "Break systems into independent, testable modules"
    magnitude: 1.0
    decay_rate: 0.05
    scope: global
    tags: [architecture, code]
```

---

#### Step 3: Extract principles
```python
principles_data = data.get("principles", [])
if not principles_data:
    # Return empty matrix if no principles
    return GlobalIdentityMatrix(principles=[], version="1.0")
```

---

#### Step 4: Initialize embedding model
```python
if embedding_model is None:
    embedding_model = EmbeddingModel()  # Uses default: all-MiniLM-L6-v2
```

**Why optional?** Allows reusing the same model for multiple loads (more efficient).

---

#### Step 5: Create principles with embeddings
```python
principles = []
for idx, p_data in enumerate(principles_data):
    # Generate ID if not provided
    principle_id = p_data.get("id")
    if not principle_id:
        name = p_data.get("name", f"principle_{idx}")
        principle_id = name.lower().replace(" ", "_").replace("-", "_")
    
    # Get name and description
    description = p_data.get("description", "")
    name = p_data.get("name", principle_id)
    
    # Compute embedding from name + description
    embedding_text = f"{name}: {description}"
    vector = embedding_model.encode_single(embedding_text)
    # vector = [0.2, 0.8, -0.1, ...] (384 numbers)
    
    # Create principle
    principle = IdentityPrinciple(
        id=principle_id,
        name=name,
        description=description,
        vector=vector,  # The ML embedding!
        magnitude=p_data.get("magnitude", 1.0),
        decay_rate=p_data.get("decay_rate", 0.05),
        scope=p_data.get("scope", "global"),
        tags=p_data.get("tags", []),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    
    principles.append(principle)
```

**Key Points:**

1. **ID Generation**: If ID not provided, generates from name
   - `"Modularity"` → `"modularity"`
   - `"Clarity Over Cleverness"` → `"clarity_over_cleverness"`

2. **Embedding Creation**: Combines name and description
   - `"Modularity: Break systems into independent, testable modules"`
   - This text is converted to 384 numbers

3. **Default Values**: Uses defaults if fields missing
   - `magnitude`: defaults to `1.0`
   - `decay_rate`: defaults to `0.05`
   - `scope`: defaults to `"global"`

---

#### Step 6: Return matrix
```python
return GlobalIdentityMatrix(
    principles=principles,
    version=data.get("version", "1.0")
)
```

---

## Complete Example: Loading Identity

Here's how the identity layer works end-to-end:

### 1. YAML Config File

```yaml
# config/identity.yaml
principles:
  - id: modularity
    name: Modularity
    description: "Break systems into independent, testable modules"
    magnitude: 1.0
    decay_rate: 0.05
    scope: global
    tags: [architecture, code]
  
  - id: clarity
    name: Clarity Over Cleverness
    description: "Prefer clear, obvious code to clever optimizations"
    magnitude: 0.9
    scope: global
    tags: [code, style]
```

### 2. Load from YAML

```python
from minime.identity.loader import load_identity_from_yaml

# Load principles and compute embeddings
matrix = load_identity_from_yaml("./config/identity.yaml")

# Now matrix has:
# - 2 principles
# - Each with computed embeddings (384 numbers each)
```

### 3. Use Identity Manager

```python
from minime.identity.principles import IdentityManager

# Create manager with loaded matrix
manager = IdentityManager(matrix=matrix)

# Get a principle
principle = manager.get_principle("modularity")
print(principle.name)  # "Modularity"
print(len(principle.vector))  # 384 (the embedding)

# Get all embeddings for similarity search
embeddings_dict = manager.to_dict()
# {"modularity": [0.2, 0.8, ...], "clarity": [0.3, 0.7, ...]}
```

### 4. Use for Similarity Search

```python
from minime.memory.embeddings import EmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# User asks a question
query = "How should I structure my code?"

# Convert query to embedding
embedding_model = EmbeddingModel()
query_vector = embedding_model.encode_single(query)

# Compare with all principles
embeddings_dict = manager.to_dict()
for principle_id, principle_vector in embeddings_dict.items():
    similarity = cosine_similarity(
        np.array([query_vector]),
        np.array([principle_vector])
    )[0][0]
    
    if similarity > 0.7:  # Threshold
        principle = manager.get_principle(principle_id)
        print(f"Relevant principle: {principle.name} (similarity: {similarity:.2f})")
```

---

## Key Concepts

### 1. Embeddings are the Key

- Principles are stored as **embeddings** (384 numbers)
- Embeddings capture **meaning**, not just words
- Similar meanings = similar embeddings

### 2. Graceful Degradation

- If config file missing → returns empty matrix
- If no principles → returns empty matrix
- System works even with no identity (uses general knowledge)

### 3. Automatic Embedding

- You don't need to compute embeddings manually
- `load_identity_from_yaml()` does it automatically
- Uses `EmbeddingModel` under the hood

### 4. Two Interfaces

- **`IdentityManager`**: Programmatic interface (add/update/get)
- **`load_identity_from_yaml()`**: File-based interface (load from YAML)

---

## Integration with Other Modules

### 1. Config Module

```python
# In minime/config.py
from minime.identity.loader import load_identity_from_yaml

# Load config
config = load_config()

# Load identity from same config file
matrix = load_identity_from_yaml(config.config_dir + "/identity.yaml")
```

### 2. Embeddings Module

```python
# In minime/identity/loader.py
from minime.memory.embeddings import EmbeddingModel

# Uses EmbeddingModel to compute embeddings
embedding_model = EmbeddingModel()
vector = embedding_model.encode_single(text)
```

### 3. Future: Context Manager

```python
# In minime/context/manager.py (future)
from minime.identity.principles import IdentityManager

# Use identity for similarity search
manager = IdentityManager(matrix=loaded_matrix)
embeddings_dict = manager.to_dict()
# Compare query embeddings with principle embeddings
```

---

## Summary

The identity layer is the **personalization engine** of MiniMe:

1. **Stores your principles** (values, preferences, guidelines)
2. **Converts to embeddings** (so ML can understand meaning)
3. **Enables similarity search** (find relevant principles for queries)
4. **Personalizes responses** (apply your principles to answers)

**Key Files:**
- `principles.py`: Programmatic management of principles
- `loader.py`: Load from YAML and compute embeddings

**Without identity layer:**
- MiniMe would be generic (same for everyone)
- No personalization
- No way to apply your values

**With identity layer:**
- ✅ Personalized responses
- ✅ Applies your principles
- ✅ Understands your values
- ✅ Learns your preferences

