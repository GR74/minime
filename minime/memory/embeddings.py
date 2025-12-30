"""Embedding model for text embeddings."""

import hashlib
from typing import List, Optional
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper for embedding models (sentence-transformers or OpenAI)."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        provider: str = "sentence-transformers",
        revision: str = "v1",
        encoder_sha: Optional[str] = None,
    ):
        """
        Initialize embedding model.

        Args:
            model_name: Name of the model to use. Default: "all-MiniLM-L6-v2"
            provider: Provider name (e.g., "sentence-transformers", "openai", "cohere")
            revision: Model revision/version (e.g., "v1", "v2")
            encoder_sha: SHA hash of encoder code (for versioning). If None, computed from code.
        """
        self.model_name = model_name
        self.provider = provider
        self.revision = revision
        self._encoder_sha = encoder_sha or self._compute_encoder_sha()
        self._model: SentenceTransformer | None = None

    def _compute_encoder_sha(self) -> str:
        """Compute SHA of encoder code for versioning."""
        # For now, use a simple hash of the class code
        # In production, this would be a git commit hash or build ID
        code = f"{self.__class__.__name__}:{self.model_name}:{self.provider}"
        return hashlib.sha256(code.encode()).hexdigest()[:8]

    @property
    def encoder_sha(self) -> str:
        """Get encoder SHA for versioning."""
        return self._encoder_sha

    def get_embedding_metadata(self) -> dict:
        """
        Get metadata dict for embedding versioning.
        
        Returns:
            Dictionary with provider, model, revision, encoder_sha
        """
        return {
            "provider": self.provider,
            "model": self.model_name,
            "revision": self.revision,
            "encoder_sha": self.encoder_sha,
        }

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> List[List[float]]:
        """
        Encode a batch of texts to embeddings.

        Args:
            texts: List of text strings to encode
            show_progress_bar: Whether to show progress bar

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

        # Convert numpy arrays to lists
        return [emb.tolist() for emb in embeddings]

    def encode_single(self, text: str) -> List[float]:
        """
        Encode a single text to an embedding vector.

        Args:
            text: Text string to encode

        Returns:
            Embedding vector as a list of floats
        """
        if not text:
            # Return zero vector if text is empty
            # Dimension depends on model, but all-MiniLM-L6-v2 is 384
            return [0.0] * 384

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

