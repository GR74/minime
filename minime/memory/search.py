"""Semantic search and retrieval for memory chunks."""

from typing import List, Optional, Tuple

import numpy as np

from minime.memory.db import AsyncDatabase
from minime.memory.embeddings import EmbeddingModel
from minime.schemas import MemoryChunk, VaultNode


async def promote_ephemeral_node(db: AsyncDatabase, node_id: str, target_path: str, title: str = None):
    """
    Promote an ephemeral node to a full vault node.
    
    Args:
        db: AsyncDatabase instance
        node_id: ID of ephemeral node to promote
        target_path: Target path for the promoted node (e.g., "notes/promoted.md")
        title: Optional title for the promoted node
    
    Returns:
        Updated VaultNode
    """
    node = await db.get_node(node_id)
    if not node:
        raise ValueError(f"Node {node_id} not found")
    
    if node.scope != "ephemeral":
        raise ValueError(f"Node {node_id} is not ephemeral (scope: {node.scope})")
    
    # Update node to full vault node
    node.path = target_path
    node.scope = "global"
    if title:
        node.title = title
    
    await db.insert_node(node)
    return node


class MemorySearch:
    """Semantic search over indexed memory chunks."""

    def __init__(
        self,
        db: AsyncDatabase,
        embedding_model: EmbeddingModel,
        use_faiss: bool = True,
        faiss_index_path: Optional[str] = None,
    ):
        """
        Initialize MemorySearch.

        Args:
            db: AsyncDatabase instance
            embedding_model: EmbeddingModel instance for encoding queries
            use_faiss: Whether to use FAISS for vector search (default: True)
            faiss_index_path: Optional path to FAISS index file
        """
        self.db = db
        self.embedding_model = embedding_model
        self.use_faiss = use_faiss
        self._faiss_store = None

        if use_faiss:
            try:
                from minime.memory.vector_store_faiss import FaissVectorStore

                # Get embedding dimension from model
                # all-MiniLM-L6-v2 is 384, but we'll detect it
                test_embedding = embedding_model.encode_single("test")
                dim = len(test_embedding)

                self._faiss_store = FaissVectorStore(dim=dim, index_path=faiss_index_path)

                # Try to load existing index
                if not self._faiss_store.load():
                    # Index doesn't exist, will be built on first search or explicit build
                    pass
            except ImportError:
                # FAISS not available, fall back to linear search
                self.use_faiss = False
                self._faiss_store = None

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

    async def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0,
        node_ids: List[str] = None,
    ) -> List[Tuple[MemoryChunk, float]]:
        """
        Search for relevant chunks by semantic similarity.

        Args:
            query: Search query text
            k: Number of results to return (default: 5)
            threshold: Minimum similarity threshold (default: 0.0, return all)
            node_ids: Optional list of node_ids to search within (default: None, search all)

        Returns:
            List of (MemoryChunk, similarity_score) tuples, sorted by similarity (highest first)
        """
        # Encode query
        query_embedding = self.embedding_model.encode_single(query)
        current_meta = self.embedding_model.get_embedding_metadata()

        # Use FAISS if available and not filtering by node_ids
        if self.use_faiss and self._faiss_store and not node_ids:
            # Ensure index is built
            if self._faiss_store.size() == 0:
                await self._build_faiss_index()

            # Search using FAISS
            faiss_results = self._faiss_store.search(query_embedding, k=k * 2)  # Get more for filtering

            # Fetch chunks and validate
            results = []
            from minime.memory.embedding_utils import metadata_matches

            for chunk_id, similarity in faiss_results:
                if similarity < threshold:
                    continue

                # Get chunk from database
                chunk = await self._get_chunk_by_id(chunk_id)
                if not chunk:
                    continue

                # Validate metadata
                if not chunk.metadata or "embedding" not in chunk.metadata:
                    continue

                try:
                    if not metadata_matches(current_meta, chunk.metadata["embedding"]):
                        continue  # Skip if version mismatch
                except (ValueError, TypeError):
                    continue

                if not chunk.embedding:
                    continue

                results.append((chunk, similarity))

                if len(results) >= k:
                    break

            return results

        # Fallback to linear search (for node_ids filter or FAISS unavailable)
        # Get all chunks with embeddings
        if node_ids:
            # Filter by node_ids
            all_chunks = []
            for node_id in node_ids:
                chunks = await self.db.get_chunks_for_node(node_id)
                all_chunks.extend(chunks)
        else:
            # Get all chunks (we'll need to fetch all and compute similarity)
            # For now, get all nodes and their chunks
            # This is O(N) - for scale, would need vector DB (FAISS/Annoy)
            # Get all node embeddings first (for faster lookup)
            node_embeddings = await self.db.get_all_node_embeddings_with_metadata()
            node_id_set = {node_id for node_id, _, _ in node_embeddings}
            
            all_chunks = []
            for node_id in node_id_set:
                chunks = await self.db.get_chunks_for_node(node_id)
                all_chunks.extend(chunks)

        # Compute similarity for each chunk
        results = []
        from minime.memory.embedding_utils import metadata_matches
        for chunk in all_chunks:
            # FIX #2: Validate embedding metadata - fail if invalid structure
            if not chunk.metadata or "embedding" not in chunk.metadata:
                continue  # Skip if metadata missing (invalid)
            
            try:
                if not metadata_matches(current_meta, chunk.metadata["embedding"]):
                    continue  # Skip if version mismatch
            except (ValueError, TypeError):
                continue  # Skip if metadata validation fails

            if not chunk.embedding:
                continue

            similarity = self._compute_similarity(query_embedding, chunk.embedding)
            
            if similarity >= threshold:
                results.append((chunk, similarity))

        # Sort by similarity (highest first) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    async def _get_chunk_by_id(self, chunk_id: str) -> Optional[MemoryChunk]:
        """
        Get a chunk by its chunk_id.

        Args:
            chunk_id: Chunk ID to retrieve

        Returns:
            MemoryChunk if found, None otherwise
        """
        rows = await self.db._fetch(
            "SELECT * FROM memory_chunks WHERE chunk_id = ?",
            (chunk_id,)
        )

        if not rows:
            return None

        row = rows[0]

        # FIX #3: Read from compressed float32 blob, fallback to JSON
        import zlib
        import numpy as np
        if row["embedding_blob"]:
            embedding_bytes = zlib.decompress(row["embedding_blob"])
            embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
            embedding = embedding_array.tolist()
        elif row["embedding"]:
            import json
            embedding = json.loads(row["embedding"])
        else:
            embedding = []

        import json
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return MemoryChunk(
            chunk_id=row["chunk_id"],
            node_id=row["node_id"],
            content=row["content"],
            embedding=embedding,
            metadata=metadata,
            position=row["position"],
        )

    async def _build_faiss_index(self) -> None:
        """
        Build FAISS index from all chunks in database.
        """
        if not self.use_faiss or not self._faiss_store:
            return

        # Get all chunks with embeddings
        node_embeddings = await self.db.get_all_node_embeddings_with_metadata()
        node_id_set = {node_id for node_id, _, _ in node_embeddings}

        # Collect all chunks with valid embeddings
        embeddings = []
        ref_ids = []
        current_meta = self.embedding_model.get_embedding_metadata()
        from minime.memory.embedding_utils import metadata_matches

        for node_id in node_id_set:
            chunks = await self.db.get_chunks_for_node(node_id)
            for chunk in chunks:
                # Validate metadata
                if not chunk.metadata or "embedding" not in chunk.metadata:
                    continue

                try:
                    if not metadata_matches(current_meta, chunk.metadata["embedding"]):
                        continue  # Skip if version mismatch
                except (ValueError, TypeError):
                    continue

                if not chunk.embedding:
                    continue

                embeddings.append(chunk.embedding)
                ref_ids.append(chunk.chunk_id)

        # Rebuild index
        if embeddings:
            self._faiss_store.rebuild(embeddings, ref_ids)
            self._faiss_store.save()

    async def rebuild_index(self) -> None:
        """
        Public method to rebuild FAISS index (useful after bulk updates).
        """
        await self._build_faiss_index()

    async def add_to_index(self, chunk: MemoryChunk) -> None:
        """
        Add a single chunk to the FAISS index (for incremental updates).

        Args:
            chunk: MemoryChunk to add
        """
        if not self.use_faiss or not self._faiss_store:
            return

        # Validate metadata
        if not chunk.metadata or "embedding" not in chunk.metadata:
            return

        current_meta = self.embedding_model.get_embedding_metadata()
        from minime.memory.embedding_utils import metadata_matches

        try:
            if not metadata_matches(current_meta, chunk.metadata["embedding"]):
                return  # Skip if version mismatch
        except (ValueError, TypeError):
            return

        if not chunk.embedding:
            return

        # Add to index
        self._faiss_store.add(chunk.embedding, chunk.chunk_id)
        self._faiss_store.save()

    async def search_nodes(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[VaultNode, float]]:
        """
        Search for relevant nodes by semantic similarity (uses primary chunk).

        Args:
            query: Search query text
            k: Number of results to return (default: 5)
            threshold: Minimum similarity threshold (default: 0.0)

        Returns:
            List of (VaultNode, similarity_score) tuples, sorted by similarity
        """
        # Encode query
        query_embedding = self.embedding_model.encode_single(query)
        current_meta = self.embedding_model.get_embedding_metadata()

        # Get all node embeddings
        node_embeddings = await self.db.get_all_node_embeddings_with_metadata()

        results = []
        from minime.memory.embedding_utils import metadata_matches
        for node_id, embedding, metadata in node_embeddings:
            # FIX #2: Validate embedding metadata - fail if invalid structure
            if not metadata or "embedding" not in metadata:
                continue  # Skip if metadata missing (invalid)
            
            try:
                if not metadata_matches(current_meta, metadata["embedding"]):
                    continue  # Skip if version mismatch
            except (ValueError, TypeError):
                continue  # Skip if metadata validation fails

            if not embedding:
                continue

            similarity = self._compute_similarity(query_embedding, embedding)
            
            if similarity >= threshold:
                node = await self.db.get_node(node_id)
                if node:
                    results.append((node, similarity))

        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


class Memory:
    """
    Clean API for memory operations: write, read, link.
    
    This provides a simple interface on top of the underlying storage layer.
    """

    def __init__(
        self,
        db: AsyncDatabase,
        embedding_model: EmbeddingModel,
        indexer=None,  # VaultIndexer
    ):
        """
        Initialize Memory interface.

        Args:
            db: AsyncDatabase instance
            embedding_model: EmbeddingModel instance
            indexer: VaultIndexer instance (optional, for write operations)
        """
        self.db = db
        self.embedding_model = embedding_model
        self.indexer = indexer
        self.search_engine = MemorySearch(db, embedding_model)

    async def write(self, text: str, metadata: dict = None) -> str:
        """
        Write text to memory (creates ephemeral node and chunk).

        Args:
            text: Text content to store
            metadata: Optional metadata dictionary

        Returns:
            node_id of created ephemeral node

        Note: Creates an ephemeral node (path: "ephemeral/{hash}") to prevent orphan chunks.
        Ephemeral nodes can later be promoted to full nodes or merged.
        """
        # FIX #3: Create proper ephemeral node instead of orphan chunks
        from minime.schemas import MemoryChunk, VaultNode
        import time
        import hashlib
        from datetime import datetime

        # Generate node_id from content hash
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        node_id = hashlib.md5(f"ephemeral_{content_hash}".encode()).hexdigest()
        ephemeral_path = f"ephemeral/{content_hash[:16]}.md"

        # Check if ephemeral node already exists
        existing_node = await self.db.get_node(node_id)
        if existing_node:
            return node_id  # Already exists, return it

        # Create ephemeral VaultNode
        title = metadata.get("title") if metadata else f"Ephemeral: {text[:50]}..."
        if not title:
            title = f"Ephemeral: {text[:50]}..."
        node = VaultNode(
            node_id=node_id,
            path=ephemeral_path,
            title=title,
            frontmatter=metadata or {},
            tags=[],
            domain=None,
            scope="ephemeral",
            links=[],
            backlinks=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            embedding_ref=None,  # Will be set after chunking
        )

        # Insert node first
        await self.db.insert_node(node)

        # Chunk the text (simple single chunk for ephemeral)
        chunk_id = f"{node_id}_chunk_0"

        # Encode embedding
        embedding = self.embedding_model.encode_single(text)

        # FIX #2: Create chunk metadata with enforced schema
        # IMPORTANT: Build embedding metadata with all required fields BEFORE validation
        embedding_meta = self.embedding_model.get_embedding_metadata()
        chunk_metadata = {
            "chunk_hash": content_hash,
            "ephemeral": True,  # Mark as ephemeral for later promotion/merge
            "embedding": {
                **embedding_meta,  # provider, model, revision, encoder_sha
                "dim": len(embedding),  # Must be added before validation
                "ts": time.time(),  # Must be added before validation
            }
        }
        if metadata:
            # Merge user metadata but preserve embedding schema
            user_meta = {k: v for k, v in metadata.items() if k != "embedding" and k != "title"}
            chunk_metadata.update(user_meta)
        
        # Validate metadata schema (after all fields are set)
        from minime.memory.embedding_utils import validate_embedding_metadata
        validate_embedding_metadata(chunk_metadata)

        chunk = MemoryChunk(
            chunk_id=chunk_id,
            node_id=node_id,
            content=text,
            embedding=embedding,
            metadata=chunk_metadata,
            position=0,
        )

        await self.db.insert_chunk(chunk)

        # Update node with embedding reference
        node.embedding_ref = chunk_id
        await self.db.insert_node(node)

        return node_id

    async def read(self, query: str, k: int = 5) -> List[str]:
        """
        Read relevant chunks from memory.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of chunk content strings, sorted by relevance
        """
        results = await self.search_engine.search(query, k=k)
        return [chunk.content for chunk, _ in results]

    async def link(self, a: str, b: str, reason: str) -> str:
        """
        Create a link between two nodes.

        Args:
            a: Source node identifier (node_id or title)
            b: Target node identifier (node_id or title)
            reason: Reason for the link

        Returns:
            edge_id of created link

        Note: This is a simplified version - full implementation would handle node resolution
        """
        from minime.schemas import GraphEdge
        from datetime import datetime

        # Resolve node IDs (simplified - would need proper lookup)
        source_node_id = a if len(a) == 32 else await self.db.get_node_by_title(a)
        target_node_id = b if len(b) == 32 else await self.db.get_node_by_title(b)

        if not source_node_id or not target_node_id:
            raise ValueError(f"Could not resolve nodes: {a} -> {b}")

        edge_id = f"{source_node_id}_{target_node_id}_manual"
        edge = GraphEdge(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type="manual",
            weight=1.0,
            rationale=reason,
            confidence=1.0,
            is_approved=True,
            created_at=datetime.now(),
            approved_at=datetime.now(),
        )

        await self.db.insert_edge(edge)
        return edge_id

