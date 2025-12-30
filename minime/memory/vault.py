"""Vault indexer for parsing Obsidian notes and creating knowledge graph."""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import frontmatter

from minime.memory.chunk import chunk_note
from minime.memory.db import AsyncDatabase
from minime.memory.embeddings import EmbeddingModel
from minime.memory.graph import GraphService
from minime.schemas import MemoryChunk, VaultNode


class VaultIndexer:
    """
    Indexes Obsidian vault: file I/O, parsing, chunking, embedding.
    
    Graph operations are delegated to GraphService to maintain single responsibility.
    """

    def __init__(
        self,
        vault_path: str,
        db: AsyncDatabase,
        embedding_model: EmbeddingModel,
        graph_service: GraphService = None,
    ):
        """
        Initialize VaultIndexer.

        Args:
            vault_path: Path to Obsidian vault directory
            db: AsyncDatabase instance
            embedding_model: EmbeddingModel instance for computing embeddings
            graph_service: Optional GraphService instance (created if not provided)
        """
        self.vault_path = Path(vault_path).expanduser()
        self.db = db
        self.embedding_model = embedding_model
        self.graph_service = graph_service or GraphService(db, embedding_model)

    async def index(self) -> List[VaultNode]:
        """
        Index all markdown files in the vault.

        Returns:
            List of VaultNode objects created/updated
        """
        if not self.vault_path.exists():
            return []

        # Find all .md files recursively
        md_files = list(self.vault_path.rglob("*.md"))

        if not md_files:
            return []

        nodes = []
        for md_file in md_files:
            try:
                node = await self._process_file(md_file)
                if node:
                    nodes.append(node)
            except Exception as e:
                # Log error but continue processing other files
                print(f"Warning: Failed to process {md_file}: {e}")
                continue

        return nodes

    async def _process_file(self, file_path: Path) -> Optional[VaultNode]:
        """
        Process a single markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            VaultNode if successful, None otherwise
        """
        try:
            # Read file
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return None

        # Parse frontmatter
        try:
            post = frontmatter.loads(content)
            frontmatter_data = post.metadata
            body = post.content
        except Exception as e:
            # If frontmatter parsing fails, treat as no frontmatter
            print(f"Warning: Could not parse frontmatter for {file_path}: {e}")
            frontmatter_data = {}
            body = content

        # Extract metadata
        title = frontmatter_data.get("title") or file_path.stem
        tags = self._extract_tags(frontmatter_data, body)
        domain = frontmatter_data.get("domain")
        scope = frontmatter_data.get("scope", "global")
        links = self._extract_wikilinks(body)

        # Generate node_id from path
        relative_path = str(file_path.relative_to(self.vault_path))
        node_id = hashlib.md5(relative_path.encode()).hexdigest()

        # FIX #4: Compute content hash for deduplication
        content_hash = hashlib.sha256(body.encode()).hexdigest()
        
        # Check if node exists and if content has changed
        existing_node = await self.db.get_node(node_id)
        if existing_node:
            # Check if we need to re-embed (content changed)
            existing_chunks = await self.db.get_chunks_for_node(node_id)
            if existing_chunks and existing_chunks[0].metadata:
                # Get stored hash from metadata if available
                stored_hash = existing_chunks[0].metadata.get("content_hash")
                if stored_hash == content_hash:
                    # Content unchanged, skip re-embedding
                    # Update metadata in case it changed (title, tags, etc.)
                    existing_node.updated_at = datetime.now()
                    existing_node.title = title
                    existing_node.tags = tags
                    existing_node.domain = domain
                    existing_node.links = links
                    existing_node.frontmatter = frontmatter_data
                    await self.db.insert_node(existing_node)
                    return existing_node  # Return existing node without re-processing

        # Create VaultNode
        node = VaultNode(
            node_id=node_id,
            path=relative_path,
            title=title,
            frontmatter=frontmatter_data,
            tags=tags,
            domain=domain,
            scope=scope,
            links=links,
            backlinks=[],  # Will be computed later
            created_at=existing_node.created_at if existing_node else datetime.now(),
            updated_at=datetime.now(),
            embedding_ref=None,  # Will be set after chunking
        )

        # Chunk note
        chunks = chunk_note(body)

        if not chunks:
            # Empty note, still store the node
            await self.db.insert_node(node)
            return node

        # FIX #4: Check chunk-level hashes to skip re-embedding unchanged chunks
        import time
        existing_chunks = await self.db.get_chunks_for_node(node_id)
        existing_chunk_map = {chunk.chunk_id: chunk for chunk in existing_chunks}
        
        chunks_to_embed = []
        chunks_to_store = []
        embeddings = []  # Initialize to avoid NameError when all chunks unchanged
        primary_chunk_id = None
        embedding_meta = self.embedding_model.get_embedding_metadata()
        
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{node_id}_chunk_{idx}"
            
            # Compute chunk-level content hash
            chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
            
            # Check if chunk exists and is unchanged
            existing_chunk = existing_chunk_map.get(chunk_id)
            if existing_chunk and existing_chunk.metadata:
                stored_chunk_hash = existing_chunk.metadata.get("chunk_hash")
                stored_note_hash = existing_chunk.metadata.get("content_hash")
                
                # Skip if both chunk and note content unchanged
                if stored_chunk_hash == chunk_hash and stored_note_hash == content_hash:
                    if primary_chunk_id is None:
                        primary_chunk_id = chunk_id
                    chunks_to_store.append(existing_chunk)  # Reuse existing chunk
                    continue
            
            # Need to re-embed this chunk
            if primary_chunk_id is None:
                primary_chunk_id = chunk_id
            chunks_to_embed.append((idx, chunk_text, chunk_id, chunk_hash))

        # Compute embeddings only for changed chunks
        if chunks_to_embed:
            chunk_texts = [chunk_text for _, chunk_text, _, _ in chunks_to_embed]
            embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=False)
            
            for (idx, chunk_text, chunk_id, chunk_hash), embedding in zip(chunks_to_embed, embeddings):
                # FIX #2: Enforce consistent embedding metadata schema
                # IMPORTANT: Build embedding metadata with all required fields BEFORE validation
                # #region agent log
                import json
                try:
                    with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "B", "location": "vault.py:_process_file", "message": "Building chunk metadata", "data": {"chunk_id": chunk_id, "embedding_meta_keys": list(embedding_meta.keys()) if embedding_meta else None, "embedding_dim": len(embedding)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
                except: pass
                # #endregion
                
                chunk_metadata = {
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_hash": chunk_hash,  # Chunk-level hash
                    "content_hash": content_hash,  # Note-level hash
                    "embedding": {
                        **embedding_meta,  # provider, model, revision, encoder_sha
                        "dim": len(embedding),  # Must be added before validation
                        "ts": time.time(),  # Must be added before validation
                    }
                }
                # #region agent log
                try:
                    with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        embed_meta_keys = list(chunk_metadata["embedding"].keys()) if "embedding" in chunk_metadata else []
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "B", "location": "vault.py:_process_file", "message": "Metadata built, validating", "data": {"embedding_keys": embed_meta_keys, "has_dim": "dim" in chunk_metadata.get("embedding", {}), "has_ts": "ts" in chunk_metadata.get("embedding", {})}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
                except: pass
                # #endregion
                # Validate metadata schema (after all fields are set)
                from minime.memory.embedding_utils import validate_embedding_metadata
                try:
                    validate_embedding_metadata(chunk_metadata)
                    # #region agent log
                    try:
                        with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "B", "location": "vault.py:_process_file", "message": "Metadata validation passed", "data": {}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
                    except: pass
                    # #endregion
                except Exception as e:
                    # #region agent log
                    try:
                        with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "B", "location": "vault.py:_process_file", "message": "Metadata validation failed", "data": {"exception_type": type(e).__name__, "exception_msg": str(e)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
                    except: pass
                    # #endregion
                    raise

                chunk = MemoryChunk(
                    chunk_id=chunk_id,
                    node_id=node_id,
                    content=chunk_text,
                    embedding=embedding,
                    metadata=chunk_metadata,
                    position=idx,
                )

                await self.db.insert_chunk(chunk)
                chunks_to_store.append(chunk)
        else:
            # All chunks unchanged, update metadata if needed (e.g., timestamp)
            for chunk in chunks_to_store:
                if chunk.metadata.get("content_hash") != content_hash:
                    # Update note-level hash if changed
                    chunk.metadata["content_hash"] = content_hash
                    await self.db.insert_chunk(chunk)

        # Update node with primary chunk reference
        node.embedding_ref = primary_chunk_id
        await self.db.insert_node(node)

        # Delegate graph operations to GraphService
        # Create explicit edges (wikilinks)
        await self.graph_service.create_wikilink_edges(node)

        # Generate similarity proposals
        # Use primary chunk embedding (either newly computed or from existing chunk)
        primary_embedding = None
        # #region agent log
        import json
        try:
            with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "C", "location": "vault.py:_process_file", "message": "Getting primary embedding", "data": {"has_embeddings": len(embeddings) > 0, "has_chunks_to_store": len(chunks_to_store) > 0, "primary_chunk_id": primary_chunk_id}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
        except: pass
        # #endregion
        
        if embeddings:
            primary_embedding = embeddings[0]
            # #region agent log
            try:
                with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "C", "location": "vault.py:_process_file", "message": "Using new embedding", "data": {"embedding_dim": len(primary_embedding) if primary_embedding else 0}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
            except: pass
            # #endregion
        elif chunks_to_store and primary_chunk_id:
            # All chunks unchanged, get embedding from existing primary chunk
            primary_chunk = next((c for c in chunks_to_store if c.chunk_id == primary_chunk_id), None)
            # #region agent log
            try:
                with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "C", "location": "vault.py:_process_file", "message": "Looking for existing primary chunk", "data": {"found_chunk": primary_chunk is not None, "chunk_has_embedding": primary_chunk.embedding is not None if primary_chunk else False}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
            except: pass
            # #endregion
            if primary_chunk:
                primary_embedding = primary_chunk.embedding
                # #region agent log
                try:
                    with open(r'c:\Users\gowri\minime\.cursor\debug.log', 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "check1", "hypothesisId": "C", "location": "vault.py:_process_file", "message": "Using existing chunk embedding", "data": {"embedding_dim": len(primary_embedding) if primary_embedding else 0}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
                except: pass
                # #endregion
        
        await self.graph_service.generate_similarity_proposals(
            node,
            primary_embedding,
        )

        return node

    def _extract_wikilinks(self, text: str) -> List[str]:
        """
        Extract all wikilinks from text.

        Args:
            text: Text to search for wikilinks

        Returns:
            List of wikilink targets (without [[ ]])
        """
        # Pattern: [[link]] or [[link|display]]
        pattern = r"\[\[([^\]]+)\]\]"
        matches = re.findall(pattern, text)

        # Extract link target (before | if present)
        links = []
        for match in matches:
            link_target = match.split("|")[0].strip()
            if link_target:
                links.append(link_target)

        return links

    def _extract_inline_tags(self, text: str) -> List[str]:
        """
        Extract inline tags from text.

        Args:
            text: Text to search for tags

        Returns:
            List of tags (without #)
        """
        # Pattern: #tag or #tag/subtag
        pattern = r"#([a-zA-Z0-9_/-]+)"
        matches = re.findall(pattern, text)
        return list(set(matches))  # Remove duplicates

    def _extract_tags(self, frontmatter_data: dict, body: str) -> List[str]:
        """
        Extract tags from frontmatter and inline tags.

        Args:
            frontmatter_data: Parsed frontmatter
            body: Note body text

        Returns:
            Combined list of tags
        """
        tags = []

        # Tags from frontmatter
        if "tags" in frontmatter_data:
            frontmatter_tags = frontmatter_data["tags"]
            if isinstance(frontmatter_tags, list):
                tags.extend(frontmatter_tags)
            elif isinstance(frontmatter_tags, str):
                tags.append(frontmatter_tags)

        # Inline tags from body
        inline_tags = self._extract_inline_tags(body)
        tags.extend(inline_tags)

        # Remove duplicates and normalize
        return list(set(tag.lower().strip() for tag in tags if tag))


