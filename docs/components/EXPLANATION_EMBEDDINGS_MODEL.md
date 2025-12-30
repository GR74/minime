# EmbeddingModel - Text Embedding Wrapper

## Overview

`embeddings.py` provides the `EmbeddingModel` class, which wraps sentence-transformers models to convert text into numerical vectors (embeddings). These embeddings capture semantic meaning and enable similarity search.

## Purpose

The EmbeddingModel class serves as a unified interface for:
- **Text Encoding**: Converting text to embedding vectors
- **Model Management**: Loading and managing embedding models
- **Versioning**: Tracking embedding model versions for compatibility
- **Batch Processing**: Efficient encoding of multiple texts

## Key Components

### EmbeddingModel Class

```python
class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        provider: str = "sentence-transformers",
        revision: str = "v1",
        encoder_sha: Optional[str] = None,
    )
    
    def encode(texts: List[str]) -> List[List[float]]
    def encode_single(text: str) -> List[float]
    def get_embedding_metadata() -> dict
```

## How It Works

### 1. Model Initialization

The model uses lazy loading - it only loads when first used:

```python
@property
def model(self) -> SentenceTransformer:
    """Lazy load the model."""
    if self._model is None:
        self._model = SentenceTransformer(self.model_name)
    return self._model
```

**Benefits:**
- **Fast Startup**: No model loading until needed
- **Memory Efficient**: Model only loaded when required
- **Flexible**: Can switch models without restarting

### 2. Versioning System

Each embedding includes metadata for version tracking:

```python
def get_embedding_metadata(self) -> dict:
    return {
        "provider": self.provider,      # "sentence-transformers"
        "model": self.model_name,      # "all-MiniLM-L6-v2"
        "revision": self.revision,      # "v1"
        "encoder_sha": self.encoder_sha, # SHA hash of encoder code
    }
```

**Why Versioning?**
- **Compatibility**: Ensures embeddings from different model versions aren't mixed
- **Migration**: Allows upgrading models while maintaining old embeddings
- **Debugging**: Tracks which model version created each embedding

### 3. Single Text Encoding

```python
def encode_single(self, text: str) -> List[float]:
    if not text:
        return [0.0] * 384  # Return zero vector for empty text
    
    embedding = self.model.encode(text, convert_to_numpy=True)
    return embedding.tolist()
```

**Features:**
- **Empty Text Handling**: Returns zero vector for empty input
- **Dimension**: 384 for all-MiniLM-L6-v2 (model-specific)
- **Format**: Returns Python list (not numpy array)

### 4. Batch Encoding

```python
def encode(self, texts: List[str], show_progress_bar: bool = False) -> List[List[float]]:
    if not texts:
        return []
    
    embeddings = self.model.encode(
        texts,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    
    return [emb.tolist() for emb in embeddings]
```

**Benefits:**
- **Efficiency**: Batch processing is faster than individual calls
- **Progress Tracking**: Optional progress bar for large batches
- **Consistent Format**: Returns list of lists (same as single encoding)

## The Model: all-MiniLM-L6-v2

### Why This Model?

**all-MiniLM-L6-v2** is the default embedding model because:

- ✅ **Fast**: Processes ~300 texts/second
- ✅ **Small**: 22M parameters, ~80MB download
- ✅ **Quality**: Good semantic understanding for most tasks
- ✅ **CPU-Friendly**: Runs efficiently on CPU (no GPU needed)
- ✅ **Proven**: Widely used and well-tested

### Model Specifications

- **Dimensions**: 384 (output vector size)
- **Max Sequence Length**: 256 tokens (handled by chunking)
- **Architecture**: 6-layer transformer
- **Training**: Trained on diverse text corpus

## Usage Examples

### Basic Usage

```python
from minime.memory.embeddings import EmbeddingModel

# Create model (lazy loading)
model = EmbeddingModel()

# Encode single text
text = "I love coding"
embedding = model.encode_single(text)
# Returns: [0.2, 0.8, -0.1, 0.5, ...] (384 numbers)

# Encode batch
texts = ["I love coding", "I enjoy programming", "I hate cooking"]
embeddings = model.encode(texts)
# Returns: [[...], [...], [...]] (list of 384-element lists)
```

### With Custom Model

```python
# Use different model
model = EmbeddingModel(
    model_name="all-mpnet-base-v2",  # Larger, higher quality
    provider="sentence-transformers",
    revision="v1",
)

embedding = model.encode_single("Custom model example")
```

### Getting Metadata

```python
model = EmbeddingModel()
metadata = model.get_embedding_metadata()

# Returns:
# {
#     "provider": "sentence-transformers",
#     "model": "all-MiniLM-L6-v2",
#     "revision": "v1",
#     "encoder_sha": "a1b2c3d4"
# }
```

### Integration with Memory System

```python
# In VaultIndexer
embedding_model = EmbeddingModel()

# Encode note chunks
chunks = chunk_note(note_body)
embeddings = embedding_model.encode(chunks)

# Store with metadata
for chunk, embedding in zip(chunks, embeddings):
    chunk_metadata = {
        "embedding": embedding_model.get_embedding_metadata()
    }
    # Store chunk with embedding and metadata
```

## Versioning and Compatibility

### Why Versioning Matters

When you upgrade the embedding model, old embeddings become incompatible:

```python
# Old embeddings (model v1)
old_embedding = [0.2, 0.8, ...]  # Created with v1

# New model (model v2)
new_model = EmbeddingModel(revision="v2")
new_embedding = new_model.encode_single("same text")
# Different vector! Can't compare directly
```

### Version Checking

The system checks embedding metadata before comparing:

```python
from minime.memory.embedding_utils import metadata_matches

current_meta = model.get_embedding_metadata()
stored_meta = chunk.metadata.get("embedding")

if not metadata_matches(current_meta, stored_meta):
    # Incompatible versions - skip or re-embed
    pass
```

## Technical Details

### Embedding Dimensions

- **all-MiniLM-L6-v2**: 384 dimensions
- **all-mpnet-base-v2**: 768 dimensions
- **OpenAI text-embedding-ada-002**: 1536 dimensions

**Why 384?**
- Balance between quality and efficiency
- Good enough for most semantic tasks
- Fast to compute and compare

### Vector Normalization

Embeddings are typically normalized (unit vectors) for cosine similarity:

```python
# Normalize vector
vec = np.array(embedding)
norm = np.linalg.norm(vec)
normalized = vec / norm if norm > 0 else vec
```

**Note**: The model doesn't normalize automatically - normalization is done during similarity computation.

### Memory Usage

- **Model Size**: ~80MB (all-MiniLM-L6-v2)
- **Per Embedding**: 384 floats × 4 bytes = 1.5KB
- **10,000 Embeddings**: ~15MB

## Best Practices

1. **Reuse Model Instance**: Create once, use many times
   ```python
   model = EmbeddingModel()  # Create once
   # Use for many texts
   ```

2. **Batch When Possible**: Use `encode()` for multiple texts
   ```python
   # Good: Batch processing
   embeddings = model.encode(texts)
   
   # Less efficient: Individual calls
   embeddings = [model.encode_single(t) for t in texts]
   ```

3. **Track Versions**: Always store embedding metadata
   ```python
   metadata = model.get_embedding_metadata()
   chunk.metadata["embedding"] = metadata
   ```

4. **Handle Empty Text**: Check for empty strings
   ```python
   if text.strip():
       embedding = model.encode_single(text)
   else:
       embedding = [0.0] * 384  # Zero vector
   ```

## Common Issues

### Issue: Slow Encoding

**Problem**: Encoding takes too long

**Solutions**:
- Use batch encoding instead of individual calls
- Consider GPU acceleration (if available)
- Use smaller model for faster encoding

### Issue: Memory Errors

**Problem**: Out of memory with large batches

**Solutions**:
- Process in smaller batches
- Use generator for streaming
- Consider model with smaller dimensions

### Issue: Version Mismatches

**Problem**: Embeddings from different versions can't be compared

**Solutions**:
- Always check metadata before comparing
- Re-embed old data when upgrading models
- Use versioning system to track compatibility

## Future Enhancements

1. **Multiple Providers**: Support OpenAI, Cohere, etc.
2. **GPU Acceleration**: Automatic GPU detection and usage
3. **Caching**: Cache embeddings for repeated texts
4. **Streaming**: Support for streaming large text corpora
5. **Custom Models**: Support for fine-tuned models

## Summary

The `EmbeddingModel` class provides:

- ✅ Simple, unified interface for text embeddings
- ✅ Lazy loading for fast startup
- ✅ Version tracking for compatibility
- ✅ Batch processing for efficiency
- ✅ Flexible model selection

It's the core component that enables semantic search and similarity matching in the MiniMe memory system.

