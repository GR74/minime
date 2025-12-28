# Explanation: `minime/memory/embeddings.py`

This file provides the embedding functionality that converts text into numerical vectors for semantic search and similarity matching.

## Overview

The `EmbeddingModel` class wraps the `sentence-transformers` library to provide a simple interface for converting text into embeddings (vectors of numbers that represent meaning).

## File Structure

```python
class EmbeddingModel:
    - __init__(model_name)      # Initialize the model
    - model (property)          # Lazy-load the actual model
    - encode(texts)             # Batch encode multiple texts
    - encode_single(text)       # Encode a single text
```

---

## Detailed Explanation

### Class: `EmbeddingModel`

#### Purpose
Wraps the sentence-transformers library to provide embeddings for text. This is the foundation for all semantic search in MiniMe.

#### Constructor: `__init__(model_name: str = "all-MiniLM-L6-v2")`

**What it does:**
- Stores the model name (doesn't load it yet - lazy loading)
- Initializes `_model` to `None`

**Why lazy loading?**
- Loading the model takes time and memory
- We only load it when actually needed
- Saves resources if the model is never used

**Parameters:**
- `model_name`: Name of the sentence transformer model
  - Default: `"all-MiniLM-L6-v2"` (fast, small, good quality)
  - Could be: `"all-mpnet-base-v2"` (better quality, slower)
  - Or any model from [HuggingFace](https://huggingface.co/models)

**Example:**
```python
model = EmbeddingModel()  # Uses default: all-MiniLM-L6-v2
# Model not loaded yet!

model2 = EmbeddingModel("all-mpnet-base-v2")  # Different model
```

---

#### Property: `model` (Lazy Loader)

**What it does:**
- First access: Loads the SentenceTransformer model
- Subsequent accesses: Returns the already-loaded model

**Why this pattern?**
```python
@property
def model(self) -> SentenceTransformer:
    if self._model is None:
        self._model = SentenceTransformer(self.model_name)
    return self._model
```

**Benefits:**
1. **Performance**: Model only loaded when needed
2. **Memory**: Doesn't consume memory until first use
3. **Flexibility**: Can change model name without loading

**What happens when loaded?**
- Downloads model from HuggingFace (first time only)
- Loads ~80MB of weights into memory
- Takes 1-2 seconds (one-time cost)

**Example:**
```python
model = EmbeddingModel()
# No model loaded yet

embedding = model.encode_single("test")  # NOW the model loads
# Model is now in memory and ready
```

---

#### Method: `encode(texts: List[str], show_progress_bar: bool = False) -> List[List[float]]`

**What it does:**
- Takes a list of text strings
- Converts each to an embedding vector
- Returns a list of embedding vectors

**Parameters:**
- `texts`: List of strings to encode
  - Example: `["Hello world", "How are you?", "Goodbye"]`
- `show_progress_bar`: Whether to show progress (useful for large batches)

**Returns:**
- List of lists of floats
- Each inner list is one embedding (384 numbers for all-MiniLM-L6-v2)

**Example:**
```python
model = EmbeddingModel()
texts = [
    "I love coding",
    "Programming is fun",
    "What's the weather?"
]

embeddings = model.encode(texts)
# Returns:
# [
#   [0.2, 0.8, -0.1, 0.5, ...],  # 384 numbers for "I love coding"
#   [0.3, 0.7, -0.2, 0.4, ...],  # 384 numbers for "Programming is fun"
#   [-0.5, 0.1, 0.9, -0.3, ...]  # 384 numbers for "What's the weather?"
# ]
```

**Why batch encoding?**
- More efficient than encoding one-by-one
- The model processes multiple texts at once
- Faster for large datasets

**Edge case handling:**
```python
if not texts:
    return []  # Empty list returns empty list
```

---

#### Method: `encode_single(text: str) -> List[float]`

**What it does:**
- Takes a single text string
- Converts it to one embedding vector
- Returns a list of floats

**Parameters:**
- `text`: Single string to encode

**Returns:**
- List of floats (384 numbers for all-MiniLM-L6-v2)
- Or `[0.0] * 384` if text is empty

**Example:**
```python
model = EmbeddingModel()
embedding = model.encode_single("Modularity: Break systems into modules")
# Returns: [0.2, 0.8, -0.1, 0.5, ...] (384 numbers)
```

**Edge case handling:**
```python
if not text:
    return [0.0] * 384  # Zero vector for empty text
```

**Why zero vector?**
- Empty text has no meaning
- Zero vector = "no similarity" to anything
- Prevents errors in similarity calculations

---

## How It's Used in MiniMe

### 1. Identity Principles Loading

When loading identity principles from YAML:

```python
# In minime/identity/loader.py
embedding_model = EmbeddingModel()
text = f"{name}: {description}"  # "Modularity: Break systems into modules"
vector = embedding_model.encode_single(text)
# Stores vector in IdentityPrinciple.vector
```

### 2. Vault Note Indexing (Future)

When indexing Obsidian notes:

```python
# In minime/memory/vault.py (future implementation)
embedding_model = EmbeddingModel()
note_content = "My thoughts on architecture..."
chunk_embeddings = embedding_model.encode(chunks)  # Batch encode all chunks
# Stores embeddings in database
```

### 3. Semantic Search (Future)

When searching for relevant notes:

```python
# In minime/context/manager.py (future implementation)
query = "How should I structure my code?"
query_embedding = embedding_model.encode_single(query)

# Compare with all stored embeddings
for note_embedding in stored_embeddings:
    similarity = cosine_similarity(query_embedding, note_embedding)
    if similarity > threshold:
        # This note is relevant!
```

---

## Technical Details

### Model: all-MiniLM-L6-v2

**Specifications:**
- **Dimensions**: 384 (output vector length)
- **Size**: ~80MB
- **Speed**: ~300 texts/second on CPU
- **Quality**: Good for most semantic tasks
- **Language**: English (primarily)

**Architecture:**
- Based on BERT (Bidirectional Encoder Representations from Transformers)
- 6 transformer layers
- Trained on diverse text corpus
- Fine-tuned for sentence similarity

### Why This Model?

1. **Fast**: Processes text quickly without GPU
2. **Small**: Doesn't require much memory
3. **Good Quality**: Works well for semantic similarity
4. **Local**: Runs on your machine (no API calls)

### Alternative Models

You could use different models:

```python
# Better quality, slower
model = EmbeddingModel("all-mpnet-base-v2")  # 768 dimensions

# Multilingual
model = EmbeddingModel("paraphrase-multilingual-MiniLM-L12-v2")

# OpenAI embeddings (requires API key)
# Would need different implementation
```

---

## Code Flow Example

Here's a complete example of how embeddings work:

```python
from minime.memory.embeddings import EmbeddingModel

# 1. Initialize (model not loaded yet)
model = EmbeddingModel("all-MiniLM-L6-v2")

# 2. First use - model loads now
principle_text = "Modularity: Break systems into independent, testable modules"
principle_embedding = model.encode_single(principle_text)
# principle_embedding = [0.2, 0.8, -0.1, 0.5, ...] (384 numbers)

# 3. Encode user query
query_text = "How do I split my code into modules?"
query_embedding = model.encode_single(query_text)
# query_embedding = [0.3, 0.7, -0.2, 0.4, ...] (384 numbers)

# 4. Calculate similarity (using cosine similarity)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

similarity = cosine_similarity(
    np.array([principle_embedding]),
    np.array([query_embedding])
)[0][0]
# similarity ≈ 0.95 (very similar!)

# 5. Batch encoding (more efficient)
multiple_texts = [
    "I love coding",
    "Programming is fun",
    "What's the weather?"
]
embeddings = model.encode(multiple_texts)
# Returns list of 3 embeddings
```

---

## Key Concepts

### 1. Lazy Loading
- Model only loads when first used
- Saves memory and startup time
- Common pattern in Python

### 2. Batch Processing
- `encode()` handles multiple texts efficiently
- Faster than calling `encode_single()` in a loop
- Use for large datasets

### 3. Vector Representation
- Text → List of 384 numbers
- Similar meanings → Similar numbers
- Enables mathematical operations on meaning

### 4. Model Abstraction
- Wraps sentence-transformers library
- Provides simple interface
- Could swap implementation (e.g., OpenAI embeddings)

---

## Dependencies

- `sentence-transformers`: The ML library that does the actual embedding
- `torch` (PyTorch): Underlying framework (installed with sentence-transformers)
- `transformers`: HuggingFace transformers library (dependency)

---

## Error Handling

The current implementation has minimal error handling. Potential improvements:

```python
def encode_single(self, text: str) -> List[float]:
    if not text:
        return [0.0] * 384
    
    try:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        # Could log error, return zero vector, or raise
        raise ValueError(f"Failed to encode text: {e}") from e
```

---

## Summary

The `EmbeddingModel` class is the **foundation** of MiniMe's semantic understanding:

1. **Converts text to numbers** (embeddings)
2. **Enables similarity search** (find related concepts)
3. **Powers personalization** (match queries to identity principles)
4. **Simple interface** (hides ML complexity)

Without embeddings, MiniMe would only do exact word matching. With embeddings, it understands **meaning** and can find relevant information even when different words are used.

