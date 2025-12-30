"""Memory layer: vault indexing, embeddings, and storage."""

from minime.memory.chunk import chunk_note
from minime.memory.db import AsyncDatabase, init_db
from minime.memory.embeddings import EmbeddingModel
from minime.memory.embedding_utils import (
    get_embedding_metadata,
    metadata_matches,
    validate_embedding_metadata,
)
from minime.memory.graph import GraphService
from minime.memory.search import Memory, MemorySearch
from minime.memory.session import DBSession
from minime.memory.summarizer import NoteSummarizer
from minime.memory.vault import VaultIndexer
from minime.memory.vector_store_faiss import FaissVectorStore
from minime.memory.visualizer import GraphData, GraphVisualizer

__all__ = [
    "chunk_note",
    "AsyncDatabase",
    "init_db",
    "DBSession",
    "EmbeddingModel",
    "FaissVectorStore",
    "get_embedding_metadata",
    "GraphData",
    "GraphService",
    "GraphVisualizer",
    "metadata_matches",
    "Memory",
    "MemorySearch",
    "NoteSummarizer",
    "validate_embedding_metadata",
    "VaultIndexer",
]

