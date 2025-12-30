"""Utilities for embedding metadata validation and schema enforcement."""

from typing import Dict, Optional


EMBEDDING_METADATA_SCHEMA = {
    "provider": str,
    "model": str,
    "revision": str,
    "encoder_sha": str,  # Changed from "sha" to "encoder_sha" to match EmbeddingModel
    "dim": int,
    "ts": float,
}


def validate_embedding_metadata(metadata: Dict) -> bool:
    """
    Validate embedding metadata against required schema.
    
    Args:
        metadata: Metadata dictionary (should have "embedding" key with nested metadata)
    
    Returns:
        True if valid, False otherwise
    
    Raises:
        ValueError: If metadata structure is invalid (fail fast, no silent skips)
    """
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


def get_embedding_metadata(metadata: Dict) -> Optional[Dict]:
    """
    Extract and validate embedding metadata from chunk metadata.
    
    Args:
        metadata: Chunk metadata dictionary
    
    Returns:
        Embedding metadata dict if valid, None if missing/invalid
    
    Raises:
        ValueError: If metadata structure is invalid (fail fast)
    """
    if not metadata or not isinstance(metadata, dict):
        return None
    
    if "embedding" not in metadata:
        return None
    
    try:
        validate_embedding_metadata(metadata)
        return metadata["embedding"]
    except ValueError:
        # Re-raise with context
        raise


def metadata_matches(current_meta: Dict, stored_meta: Dict) -> bool:
    """
    Check if two embedding metadata dicts match (for version validation).
    
    Args:
        current_meta: Current embedding metadata (from EmbeddingModel)
        stored_meta: Stored embedding metadata (from chunk)
    
    Returns:
        True if all version fields match, False otherwise
    """
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

