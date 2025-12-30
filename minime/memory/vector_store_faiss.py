"""FAISS vector store for efficient similarity search."""

import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


class FaissVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    
    Uses IndexFlatIP (inner product) for cosine similarity with normalized vectors.
    """

    def __init__(self, dim: int, index_path: Optional[str] = None):
        """
        Initialize FAISS vector store.

        Args:
            dim: Dimension of embedding vectors (e.g., 384 for all-MiniLM-L6-v2)
            index_path: Optional path to save/load index from disk
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu is required for vector search. "
                "Install with: pip install faiss-cpu"
            )

        self.dim = dim
        self.index_path = Path(index_path) if index_path else None
        self.id_map_path = (
            Path(str(index_path) + ".idmap") if index_path else None
        )

        # Initialize empty index (inner product for cosine similarity)
        # Note: IndexFlatIP requires normalized vectors
        self.index = faiss.IndexFlatIP(dim)
        self.id_map: List[str] = []  # Maps FAISS index position to chunk_id

    def add(self, embedding: List[float], ref_id: str) -> None:
        """
        Add an embedding to the index.

        Args:
            embedding: Embedding vector (list of floats)
            ref_id: Reference ID (chunk_id) to map back to chunk
        """
        if len(embedding) != self.dim:
            raise ValueError(
                f"Embedding dimension {len(embedding)} doesn't match index dimension {self.dim}"
            )

        # Normalize vector for cosine similarity (IndexFlatIP needs normalized)
        vec = np.array([embedding], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            # Zero vector - use small epsilon to avoid division by zero
            vec = vec + 1e-8

        self.index.add(vec)
        self.id_map.append(ref_id)

    def add_batch(self, embeddings: List[List[float]], ref_ids: List[str]) -> None:
        """
        Add multiple embeddings in batch (more efficient).

        Args:
            embeddings: List of embedding vectors
            ref_ids: List of reference IDs (chunk_ids)
        """
        if len(embeddings) != len(ref_ids):
            raise ValueError("embeddings and ref_ids must have same length")

        if not embeddings:
            return

        # Normalize all vectors
        vecs = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-8)  # Avoid division by zero
        vecs = vecs / norms

        self.index.add(vecs)
        self.id_map.extend(ref_ids)

    def search(
        self, query_embedding: List[float], k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of (ref_id, similarity_score) tuples, sorted by similarity (highest first)
        """
        if len(query_embedding) != self.dim:
            raise ValueError(
                f"Query dimension {len(query_embedding)} doesn't match index dimension {self.dim}"
            )

        if self.index.ntotal == 0:
            return []

        # Normalize query vector
        query_vec = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        else:
            query_vec = query_vec + 1e-8

        # Search (returns distances and indices)
        distances, indices = self.index.search(query_vec, min(k, self.index.ntotal))

        results = []
        for i, score in zip(indices[0], distances[0]):
            if i < 0 or i >= len(self.id_map):
                continue  # Invalid index
            ref_id = self.id_map[i]
            # FAISS returns inner product (cosine similarity for normalized vectors)
            # Clamp to [0, 1] range (though it should already be in that range)
            similarity = float(max(0.0, min(1.0, score)))
            results.append((ref_id, similarity))

        return results

    def rebuild(self, embeddings: List[List[float]], ref_ids: List[str]) -> None:
        """
        Rebuild the entire index from scratch.

        Args:
            embeddings: List of all embedding vectors
            ref_ids: List of all reference IDs
        """
        # Clear existing index
        self.index.reset()
        self.id_map.clear()

        # Add all embeddings
        if embeddings:
            self.add_batch(embeddings, ref_ids)

    def size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal

    def save(self) -> None:
        """Save index and id_map to disk."""
        if not self.index_path:
            return

        # Save FAISS index
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        # Save id_map
        if self.id_map_path:
            with open(self.id_map_path, "wb") as f:
                pickle.dump(self.id_map, f)

    def load(self) -> bool:
        """
        Load index and id_map from disk.

        Returns:
            True if loaded successfully, False if files don't exist
        """
        if not self.index_path or not self.index_path.exists():
            return False

        if not self.id_map_path or not self.id_map_path.exists():
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load id_map
            with open(self.id_map_path, "rb") as f:
                self.id_map = pickle.load(f)

            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all vectors from the index."""
        self.index.reset()
        self.id_map.clear()

