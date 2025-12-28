"""Embedding model for text embeddings."""

from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper for embedding models (sentence-transformers or OpenAI)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.

        Args:
            model_name: Name of the model to use. Default: "all-MiniLM-L6-v2"
        """
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

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

