# Embedding Utils - Metadata Validation

## Overview

`embedding_utils.py` provides utility functions for validating and managing embedding metadata. It ensures embedding compatibility across model versions and prevents errors from comparing incompatible embeddings.

## Purpose

The embedding utilities address critical issues:
- **Version Compatibility**: Ensures embeddings from same model version are compared
- **Schema Validation**: Enforces consistent metadata structure
- **Error Prevention**: Fails fast on invalid metadata
- **Migration Support**: Enables safe model upgrades

## Key Components

### Metadata Schema

```python
EMBEDDING_METADATA_SCHEMA = {
    "provider": str,        # "sentence-transformers", "openai", etc.
    "model": str,           # "all-MiniLM-L6-v2", etc.
    "revision": str,        # "v1", "v2", etc.
    "encoder_sha": str,     # SHA hash of encoder code
    "dim": int,             # Vector dimension (e.g., 384)
    "ts": float,            # Timestamp
}
```

### Validation Functions

```python
def validate_embedding_metadata(metadata: Dict) -> bool
def get_embedding_metadata(metadata: Dict) -> Optional[Dict]
def metadata_matches(current_meta: Dict, stored_meta: Dict) -> bool
```

## How It Works

### 1. Metadata Validation

Validates that metadata conforms to required schema:

```python
def validate_embedding_metadata(metadata: Dict) -> bool:
    # Check structure
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a dict, got {type(metadata)}")
    
    if "embedding" not in metadata:
        raise ValueError("Metadata missing required 'embedding' key")
    
    embed_meta = metadata["embedding"]
    if not isinstance(embed_meta, dict):
        raise ValueError(f"Metadata.embedding must be a dict, got {type(embed_meta)}")
    
    # Check all required fields
    for field, field_type in EMBEDDING_METADATA_SCHEMA.items():
        if field not in embed_meta:
            raise ValueError(f"Missing required embedding metadata field: {field}")
        if not isinstance(embed_meta[field], field_type):
            raise ValueError(
                f"Invalid type for embedding.{field}: expected {field_type.__name__}, "
                f"got {type(embed_meta[field]).__name__}"
            )
    
    return True
```

**Validation Checks**:
- ✅ Metadata is a dictionary
- ✅ Has "embedding" key
- ✅ Embedding metadata is a dictionary
- ✅ All required fields present
- ✅ All fields have correct types

**Fail Fast**: Raises `ValueError` immediately on invalid metadata

### 2. Metadata Extraction

Safely extracts embedding metadata from chunk metadata:

```python
def get_embedding_metadata(metadata: Dict) -> Optional[Dict]:
    if not metadata or not isinstance(metadata, dict):
        return None
    
    if "embedding" not in metadata:
        return None
    
    try:
        validate_embedding_metadata(metadata)
        return metadata["embedding"]
    except ValueError:
        raise  # Re-raise with context
```

**Features**:
- **Safe Extraction**: Returns None if missing/invalid
- **Validation**: Validates before returning
- **Error Propagation**: Re-raises validation errors

### 3. Version Matching

Checks if two embedding metadata dicts are compatible:

```python
def metadata_matches(current_meta: Dict, stored_meta: Dict) -> bool:
    if not stored_meta:
        return False
    
    # Validate stored metadata first
    try:
        validate_embedding_metadata({"embedding": stored_meta})
    except ValueError:
        return False  # Invalid metadata = no match
    
    # Check version fields
    version_fields = ["provider", "model", "revision", "encoder_sha"]
    for field in version_fields:
        if stored_meta.get(field) != current_meta.get(field):
            return False
    
    return True
```

**Version Fields Checked**:
- `provider`: Must match (sentence-transformers, openai, etc.)
- `model`: Must match (all-MiniLM-L6-v2, etc.)
- `revision`: Must match (v1, v2, etc.)
- `encoder_sha`: Must match (code version)

**Why These Fields?**
- **Provider/Model/Revision**: Different models produce different embeddings
- **Encoder SHA**: Code changes can affect embedding computation

## Usage Examples

### Validating Metadata

```python
from minime.memory.embedding_utils import validate_embedding_metadata

# Valid metadata
metadata = {
    "embedding": {
        "provider": "sentence-transformers",
        "model": "all-MiniLM-L6-v2",
        "revision": "v1",
        "encoder_sha": "a1b2c3d4",
        "dim": 384,
        "ts": 1234567890.0
    }
}

try:
    validate_embedding_metadata(metadata)
    print("Metadata is valid!")
except ValueError as e:
    print(f"Invalid metadata: {e}")
```

### Extracting Metadata

```python
from minime.memory.embedding_utils import get_embedding_metadata

chunk_metadata = {
    "chunk_index": 0,
    "embedding": {
        "provider": "sentence-transformers",
        "model": "all-MiniLM-L6-v2",
        ...
    }
}

embedding_meta = get_embedding_metadata(chunk_metadata)
if embedding_meta:
    print(f"Model: {embedding_meta['model']}")
    print(f"Dimension: {embedding_meta['dim']}")
```

### Checking Compatibility

```python
from minime.memory.embedding_utils import metadata_matches

# Current model metadata
current_meta = {
    "provider": "sentence-transformers",
    "model": "all-MiniLM-L6-v2",
    "revision": "v1",
    "encoder_sha": "a1b2c3d4"
}

# Stored chunk metadata
stored_meta = chunk.metadata.get("embedding")

if metadata_matches(current_meta, stored_meta):
    # Compatible - can compare embeddings
    similarity = cosine_similarity(query_embedding, chunk.embedding)
else:
    # Incompatible - skip or re-embed
    print("Embedding version mismatch - skipping")
```

### Integration with Search

```python
# In MemorySearch.search()
current_meta = self.embedding_model.get_embedding_metadata()

for chunk in all_chunks:
    # Validate metadata compatibility
    if not chunk.metadata or "embedding" not in chunk.metadata:
        continue  # Skip if metadata missing
    
    try:
        if not metadata_matches(current_meta, chunk.metadata["embedding"]):
            continue  # Skip if version mismatch
    except (ValueError, TypeError):
        continue  # Skip if validation fails
    
    # Safe to compare embeddings
    similarity = self._compute_similarity(query_embedding, chunk.embedding)
```

## Why This Matters

### Problem: Incompatible Embeddings

```python
# Old embedding (model v1)
old_embedding = [0.2, 0.8, -0.1, ...]  # Created with v1

# New embedding (model v2, same text)
new_embedding = [0.3, 0.7, -0.2, ...]  # Created with v2

# Comparing them gives meaningless results!
similarity = cosine_similarity(old_embedding, new_embedding)
# Result: 0.45 (seems similar, but it's wrong!)
```

### Solution: Version Checking

```python
# Check compatibility first
if metadata_matches(current_meta, stored_meta):
    # Safe to compare
    similarity = cosine_similarity(query, stored)
else:
    # Skip or re-embed
    pass
```

## Design Decisions

### 1. Fail Fast

**Decision**: Raise `ValueError` immediately on invalid metadata

**Rationale**:
- **Early Detection**: Catch errors immediately
- **No Silent Failures**: Invalid data doesn't propagate
- **Clear Errors**: Easy to debug

### 2. Strict Validation

**Decision**: Require all fields, check types

**Rationale**:
- **Consistency**: All embeddings have same metadata
- **Reliability**: No missing fields cause errors
- **Migration**: Easy to identify incompatible embeddings

### 3. Version Fields Only

**Decision**: Only check provider, model, revision, encoder_sha

**Rationale**:
- **Focus**: These fields determine compatibility
- **Flexibility**: Other fields (dim, ts) can vary
- **Performance**: Faster comparison

## Best Practices

1. **Always Validate**: Validate metadata before storing
2. **Check Compatibility**: Check before comparing embeddings
3. **Handle Errors**: Catch ValueError and handle gracefully
4. **Re-embed on Mismatch**: Re-embed old data when upgrading models
5. **Log Mismatches**: Track version mismatches for debugging

## Common Issues

### Issue: Validation Errors

**Problem**: `ValueError` on metadata validation

**Solutions**:
- Check metadata structure
- Ensure all required fields present
- Verify field types match schema

### Issue: False Mismatches

**Problem**: Compatible embeddings marked as incompatible

**Solutions**:
- Check encoder_sha matches
- Verify revision matches
- Ensure provider/model match

### Issue: Missing Metadata

**Problem**: Old chunks don't have metadata

**Solutions**:
- Re-embed old chunks
- Add migration script
- Handle gracefully (skip or re-embed)

## Future Enhancements

1. **Migration Tools**: Automatically re-embed incompatible chunks
2. **Version History**: Track metadata version changes
3. **Compatibility Matrix**: Define which versions are compatible
4. **Auto-Upgrade**: Automatically upgrade old embeddings
5. **Validation Rules**: Custom validation rules per provider

## Summary

The `embedding_utils` module provides:

- ✅ Metadata schema validation
- ✅ Version compatibility checking
- ✅ Safe metadata extraction
- ✅ Fail-fast error handling
- ✅ Migration support

It's the critical component that ensures embedding compatibility and prevents errors from comparing incompatible embeddings in MiniMe.

