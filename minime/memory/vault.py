"""Vault indexer for parsing Obsidian notes and creating knowledge graph."""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import frontmatter
import numpy as np

from minime.memory.chunk import chunk_note
from minime.memory.db import AsyncDatabase
from minime.memory.embeddings import EmbeddingModel
from minime.schemas import GraphEdge, MemoryChunk, VaultNode


class VaultIndexer:
    """Indexes Obsidian vault, extracts metadata, computes embeddings, creates graph edges."""

    def __init__(
        self,
        vault_path: str,
        db: AsyncDatabase,
        embedding_model: EmbeddingModel,
    ):
        """
        Initialize VaultIndexer.

        Args:
            vault_path: Path to Obsidian vault directory
            db: AsyncDatabase instance
            embedding_model: EmbeddingModel instance for computing embeddings
        """
        self.vault_path = Path(vault_path).expanduser()
        self.db = db
        self.embedding_model = embedding_model

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
            created_at=datetime.now(),
            updated_at=datetime.now(),
            embedding_ref=None,  # Will be set after chunking
        )

        # Chunk note
        chunks = chunk_note(body)

        if not chunks:
            # Empty note, still store the node
            await self.db.insert_node(node)
            return node

        # Compute embeddings for chunks
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)

        # Store chunks
        primary_chunk_id = None
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{node_id}_chunk_{idx}"
            if primary_chunk_id is None:
                primary_chunk_id = chunk_id

            chunk = MemoryChunk(
                chunk_id=chunk_id,
                node_id=node_id,
                content=chunk_text,
                embedding=embedding,
                metadata={"chunk_index": idx, "total_chunks": len(chunks)},
                position=idx,
            )

            await self.db.insert_chunk(chunk)

        # Update node with primary chunk reference
        node.embedding_ref = primary_chunk_id
        await self.db.insert_node(node)

        # Create explicit edges (wikilinks)
        await self._create_wikilink_edges(node, links)

        # Generate similarity proposals
        await self._generate_similarity_proposals(node, embeddings[0] if embeddings else None)

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

    async def _create_wikilink_edges(self, node: VaultNode, links: List[str]) -> None:
        """
        Create explicit edges for wikilinks.

        Args:
            node: Source node
            links: List of wikilink targets
        """
        for link_target in links:
            # Try to find target node by path or title
            # For MVP, we'll create edge if target exists
            # In future, could do fuzzy matching

            # Generate potential node_id from link (simple hash for now)
            # This is a heuristic - in production, might need better matching
            target_node_id = hashlib.md5(link_target.lower().encode()).hexdigest()

            # Check if target node exists (by checking if any node has this as title/path)
            # For MVP, we'll create the edge anyway and let it be resolved later
            edge_id = f"{node.node_id}_{target_node_id}_wikilink"

            edge = GraphEdge(
                edge_id=edge_id,
                source_node_id=node.node_id,
                target_node_id=target_node_id,  # May not exist yet
                edge_type="wikilink",
                weight=1.0,
                rationale=f"Explicit wikilink: [[{link_target}]]",
                confidence=1.0,
                is_approved=True,
                created_at=datetime.now(),
                approved_at=datetime.now(),
            )

            await self.db.insert_edge(edge)

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

        # Cosine similarity: dot product / (norm1 * norm2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    async def _generate_similarity_proposals(
        self, node: VaultNode, embedding: Optional[List[float]], threshold: float = 0.7
    ) -> None:
        """
        Generate similarity-based edge proposals.

        Args:
            node: Current node being indexed
            embedding: Primary embedding for the node
            threshold: Similarity threshold for proposals (default: 0.7)
        """
        if not embedding:
            return

        # Get all existing node embeddings
        existing_embeddings = await self.db.get_all_node_embeddings()

        proposals_created = 0
        max_proposals = 10  # Limit proposals per node

        for other_node_id, other_embedding in existing_embeddings:
            # Skip self
            if other_node_id == node.node_id:
                continue

            # Compute similarity
            similarity = self._compute_similarity(embedding, other_embedding)

            if similarity > threshold:
                # Create proposal
                edge_id = f"{node.node_id}_{other_node_id}_similar"
                confidence = similarity

                proposal = GraphEdge(
                    edge_id=edge_id,
                    source_node_id=node.node_id,
                    target_node_id=other_node_id,
                    edge_type="similar",
                    weight=similarity,
                    rationale=f"Semantic similarity: {similarity:.3f}",
                    confidence=confidence,
                    is_approved=False,  # Proposals need approval
                    created_at=datetime.now(),
                    approved_at=None,
                )

                await self.db.insert_proposal(proposal)
                proposals_created += 1

                if proposals_created >= max_proposals:
                    break

