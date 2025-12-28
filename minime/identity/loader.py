"""Load identity principles from YAML configuration."""

from pathlib import Path
from typing import Optional
import yaml
from datetime import datetime

from minime.schemas import IdentityPrinciple, GlobalIdentityMatrix
from minime.memory.embeddings import EmbeddingModel


def load_identity_from_yaml(
    config_path: str,
    embedding_model: Optional[EmbeddingModel] = None,
) -> GlobalIdentityMatrix:
    """
    Load identity principles from YAML config file and compute embeddings.

    Args:
        config_path: Path to YAML config file
        embedding_model: Optional EmbeddingModel. If None, creates a new one.

    Returns:
        GlobalIdentityMatrix with loaded principles and computed embeddings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid or missing required fields
    """
    config_file = Path(config_path)
    if not config_file.exists():
        # Return empty matrix if file doesn't exist (graceful handling)
        return GlobalIdentityMatrix(principles=[], version="1.0")

    # Load YAML
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Failed to parse YAML config: {e}") from e

    # Extract principles
    principles_data = data.get("principles", [])
    if not principles_data:
        # Return empty matrix if no principles defined
        return GlobalIdentityMatrix(principles=[], version="1.0")

    # Initialize embedding model if not provided
    if embedding_model is None:
        embedding_model = EmbeddingModel()

    # Create principles
    principles = []
    for idx, p_data in enumerate(principles_data):
        # Generate ID if not provided
        principle_id = p_data.get("id")
        if not principle_id:
            # Generate ID from name or index
            name = p_data.get("name", f"principle_{idx}")
            principle_id = name.lower().replace(" ", "_").replace("-", "_")

        # Get description for embedding
        description = p_data.get("description", "")
        name = p_data.get("name", principle_id)

        # Compute embedding from name + description
        embedding_text = f"{name}: {description}"
        vector = embedding_model.encode_single(embedding_text)

        # Create principle
        principle = IdentityPrinciple(
            id=principle_id,
            name=name,
            description=description,
            vector=vector,
            magnitude=p_data.get("magnitude", 1.0),
            decay_rate=p_data.get("decay_rate", 0.05),
            scope=p_data.get("scope", "global"),
            tags=p_data.get("tags", []),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        principles.append(principle)

    return GlobalIdentityMatrix(principles=principles, version=data.get("version", "1.0"))

