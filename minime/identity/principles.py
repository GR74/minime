"""Identity principles and global identity matrix."""

from typing import Dict, Any, Optional
from datetime import datetime

from minime.schemas import IdentityPrinciple, GlobalIdentityMatrix


class IdentityManager:
    """Manager for identity principles."""

    def __init__(self, matrix: Optional[GlobalIdentityMatrix] = None):
        """
        Initialize identity manager.

        Args:
            matrix: Optional GlobalIdentityMatrix to use. If None, creates empty matrix.
        """
        self.matrix = matrix or GlobalIdentityMatrix(principles=[], version="1.0")

    def get_principle(self, principle_id: str) -> Optional[IdentityPrinciple]:
        """Get a principle by ID."""
        return self.matrix.get_principle(principle_id)

    def add_principle(
        self,
        name: str,
        description: str,
        vector: list[float],
        magnitude: float = 1.0,
        decay_rate: float = 0.05,
        scope: str = "global",
        tags: Optional[list[str]] = None,
        principle_id: Optional[str] = None,
    ) -> IdentityPrinciple:
        """
        Add a new principle to the identity matrix.

        Args:
            name: Principle name
            description: Principle description
            vector: Embedding vector for the principle
            magnitude: Importance weight (default: 1.0)
            decay_rate: Adaptation rate (default: 0.05)
            scope: Scope of the principle (default: "global")
            tags: List of tags (default: empty list)
            principle_id: Optional ID. If None, generates from name.

        Returns:
            Created IdentityPrinciple
        """
        if principle_id is None:
            principle_id = name.lower().replace(" ", "_")

        principle = IdentityPrinciple(
            id=principle_id,
            name=name,
            description=description,
            vector=vector,
            magnitude=magnitude,
            decay_rate=decay_rate,
            scope=scope,
            tags=tags or [],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self.matrix.add_principle(principle)
        return principle

    def update_principle(self, principle_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a principle with new values.

        Args:
            principle_id: ID of the principle to update
            updates: Dictionary of fields to update

        Returns:
            True if update was successful, False if principle not found
        """
        success = self.matrix.update_principle(principle_id, updates)
        if success:
            # Update the updated_at timestamp
            principle = self.get_principle(principle_id)
            if principle:
                principle.updated_at = datetime.now()
        return success

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary mapping principle IDs to their embedding vectors."""
        return self.matrix.to_dict()

    def get_all_principles(self) -> list[IdentityPrinciple]:
        """Get all principles in the matrix."""
        return self.matrix.principles

