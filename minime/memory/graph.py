"""Graph service for managing edges and similarity proposals."""

import random
from datetime import datetime
from typing import List, Optional

import numpy as np

from minime.memory.db import AsyncDatabase
from minime.memory.embeddings import EmbeddingModel
from minime.schemas import GraphEdge, VaultNode


class GraphService:
    """
    Service for managing graph edges and similarity proposals.
    
    Separated from VaultIndexer to maintain single responsibility:
    - VaultIndexer: file I/O, parsing, chunking, embedding
    - GraphService: graph operations (edges, proposals)
    """

    def __init__(
        self,
        db: AsyncDatabase,
        embedding_model: EmbeddingModel,
    ):
        """
        Initialize GraphService.

        Args:
            db: AsyncDatabase instance
            embedding_model: EmbeddingModel instance for computing similarity
        """
        self.db = db
        self.embedding_model = embedding_model

    async def create_wikilink_edges(self, node: VaultNode) -> List[GraphEdge]:
        """
        Create edges for wikilinks found in a node.

        Args:
            node: VaultNode with links to create edges for

        Returns:
            List of created GraphEdge objects
        """
        edges = []

        for link_target in node.links:
            # Resolve target node by title
            target_node_id = await self.db.get_node_by_title(link_target)
            if not target_node_id:
                # Target doesn't exist yet - skip (no ghost nodes)
                continue

            # Check if edge already exists
            existing = await self.db.get_edge(
                node.node_id,
                target_node_id,
                edge_type="wikilink",
            )
            if existing:
                continue  # Edge already exists

            # Create edge
            edge_id = f"{node.node_id}_{target_node_id}_wikilink"
            edge = GraphEdge(
                edge_id=edge_id,
                source_node_id=node.node_id,
                target_node_id=target_node_id,
                edge_type="wikilink",
                weight=1.0,
                rationale=f"Explicit wikilink: [[{link_target}]]",
                confidence=1.0,
                is_approved=True,
                created_at=datetime.now(),
                approved_at=datetime.now(),
            )

            await self.db.insert_edge(edge)
            edges.append(edge)

        return edges

    async def generate_similarity_proposals(
        self,
        node: VaultNode,
        embedding: Optional[List[float]],
        threshold: float = 0.7,
        max_comparisons: int = 200,
        max_proposals: int = 10,
    ) -> List[GraphEdge]:
        """
        Generate similarity-based edge proposals.

        Args:
            node: Current node being indexed
            embedding: Primary embedding for the node
            threshold: Similarity threshold for proposals (default: 0.7)
            max_comparisons: Maximum number of nodes to compare against (default: 200)
            max_proposals: Maximum proposals to create (default: 10)

        Returns:
            List of created GraphEdge proposals
        """
        if not embedding:
            return []

        # FIX #5: Compute embedding SHA for edge revisioning
        import hashlib
        import numpy as np
        embedding_array = np.array(embedding, dtype=np.float32)
        embedding_bytes = embedding_array.tobytes()
        embedding_sha = hashlib.sha256(embedding_bytes).hexdigest()[:16]  # Use first 16 chars

        # Get all existing node embeddings with metadata
        existing_embeddings = await self.db.get_all_node_embeddings_with_metadata()
        current_meta = self.embedding_model.get_embedding_metadata()

        # Limit comparisons to prevent O(N²) explosion
        if len(existing_embeddings) > max_comparisons:
            existing_embeddings = random.sample(existing_embeddings, max_comparisons)

        proposals = []
        proposals_created = 0

        for other_node_id, other_embedding, other_metadata in existing_embeddings:
            # Skip self
            if other_node_id == node.node_id:
                continue

            # FIX #2: Validate embedding metadata - fail if invalid structure
            from minime.memory.embedding_utils import metadata_matches
            if not other_metadata or "embedding" not in other_metadata:
                continue  # Skip if metadata missing (invalid)
            
            try:
                if not metadata_matches(current_meta, other_metadata["embedding"]):
                    continue  # Skip if version mismatch
            except (ValueError, TypeError):
                continue  # Skip if metadata validation fails

            # Compute similarity
            similarity = self._compute_similarity(embedding, other_embedding)

            if similarity > threshold:
                # FIX #6: Include embedding SHA in edge ID for revisioning
                other_embedding_array = np.array(other_embedding, dtype=np.float32)
                other_embedding_bytes = other_embedding_array.tobytes()
                other_embedding_sha = hashlib.sha256(other_embedding_bytes).hexdigest()[:16]
                
                edge_id = f"{node.node_id}_{other_node_id}_similar_{embedding_sha}_{other_embedding_sha}"
                
                # Check if proposal already exists
                existing = await self.db.get_proposal(edge_id)
                if existing:
                    continue  # Proposal already exists

                # Create proposal
                proposal = GraphEdge(
                    edge_id=edge_id,
                    source_node_id=node.node_id,
                    target_node_id=other_node_id,
                    edge_type="similar",
                    weight=similarity,
                    rationale=f"Semantic similarity: {similarity:.3f}",
                    confidence=similarity,
                    is_approved=False,  # Proposals need approval
                    created_at=datetime.now(),
                    approved_at=None,
                )

                await self.db.insert_proposal(proposal)
                proposals.append(proposal)
                proposals_created += 1

                if proposals_created >= max_proposals:
                    break

        return proposals

    def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not embedding1 or not embedding2:
            return 0.0

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

