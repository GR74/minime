"""SQLite database schema and AsyncDatabase class for MiniMe memory layer."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import aiosqlite

from minime.schemas import GraphEdge, MemoryChunk, VaultNode


def init_db(db_path: str) -> None:
    """
    Initialize SQLite database with all required tables.

    Args:
        db_path: Path to SQLite database file
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Create tables synchronously (aiosqlite doesn't support CREATE TABLE in async context easily)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # vault_nodes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vault_nodes (
            node_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            title TEXT,
            frontmatter JSON,
            tags JSON,
            domain TEXT,
            scope TEXT DEFAULT 'global',
            links JSON,
            backlinks JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            embedding_ref TEXT
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vault_nodes_path ON vault_nodes(path)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vault_nodes_domain ON vault_nodes(domain)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vault_nodes_scope ON vault_nodes(scope)")

    # memory_chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_chunks (
            chunk_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding JSON,
            metadata JSON,
            position INTEGER DEFAULT 0,
            FOREIGN KEY (node_id) REFERENCES vault_nodes(node_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_chunks_node_id ON memory_chunks(node_id)")

    # graph_edges table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_edges (
            edge_id TEXT PRIMARY KEY,
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            rationale TEXT,
            confidence REAL DEFAULT 1.0,
            is_approved BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            approved_at TIMESTAMP,
            FOREIGN KEY (source_node_id) REFERENCES vault_nodes(node_id) ON DELETE CASCADE,
            FOREIGN KEY (target_node_id) REFERENCES vault_nodes(node_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_node_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_node_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type)")

    # graph_proposals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_proposals (
            proposal_id TEXT PRIMARY KEY,
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            edge_type TEXT DEFAULT 'similar',
            weight REAL,
            confidence REAL,
            rationale TEXT,
            requires_user_approval BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_node_id) REFERENCES vault_nodes(node_id) ON DELETE CASCADE,
            FOREIGN KEY (target_node_id) REFERENCES vault_nodes(node_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_proposals_source ON graph_proposals(source_node_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_proposals_created ON graph_proposals(created_at)")

    conn.commit()
    conn.close()


class AsyncDatabase:
    """Async database wrapper for SQLite operations."""

    def __init__(self, db_path: str):
        """
        Initialize AsyncDatabase.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create async SQLite connection."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row
        return self._connection

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def insert_node(self, node: VaultNode) -> str:
        """
        Insert or update a vault node.

        Args:
            node: VaultNode object to insert

        Returns:
            node_id of the inserted/updated node
        """
        conn = await self._get_connection()
        cursor = await conn.cursor()

        # Serialize JSON fields
        frontmatter_json = json.dumps(node.frontmatter)
        tags_json = json.dumps(node.tags)
        links_json = json.dumps(node.links)
        backlinks_json = json.dumps(node.backlinks)

        # Use INSERT OR REPLACE for upsert behavior
        await cursor.execute("""
            INSERT OR REPLACE INTO vault_nodes
            (node_id, path, title, frontmatter, tags, domain, scope, links, backlinks,
             created_at, updated_at, embedding_ref)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node.node_id,
            node.path,
            node.title,
            frontmatter_json,
            tags_json,
            node.domain,
            node.scope,
            links_json,
            backlinks_json,
            node.created_at.isoformat(),
            node.updated_at.isoformat(),
            node.embedding_ref,
        ))

        await conn.commit()
        return node.node_id

    async def get_node(self, node_id: str) -> Optional[VaultNode]:
        """
        Get a vault node by ID.

        Args:
            node_id: Node ID to retrieve

        Returns:
            VaultNode if found, None otherwise
        """
        conn = await self._get_connection()
        cursor = await conn.cursor()

        await cursor.execute("SELECT * FROM vault_nodes WHERE node_id = ?", (node_id,))
        row = await cursor.fetchone()

        if row is None:
            return None

        # Deserialize JSON fields
        frontmatter = json.loads(row["frontmatter"]) if row["frontmatter"] else {}
        tags = json.loads(row["tags"]) if row["tags"] else []
        links = json.loads(row["links"]) if row["links"] else []
        backlinks = json.loads(row["backlinks"]) if row["backlinks"] else []

        return VaultNode(
            node_id=row["node_id"],
            path=row["path"],
            title=row["title"],
            frontmatter=frontmatter,
            tags=tags,
            domain=row["domain"],
            scope=row["scope"],
            links=links,
            backlinks=backlinks,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            embedding_ref=row["embedding_ref"],
        )

    async def count_nodes(self) -> int:
        """
        Count total number of nodes in the database.

        Returns:
            Number of nodes
        """
        conn = await self._get_connection()
        cursor = await conn.cursor()

        await cursor.execute("SELECT COUNT(*) as count FROM vault_nodes")
        row = await cursor.fetchone()
        return row["count"] if row else 0

    async def insert_chunk(self, chunk: MemoryChunk) -> str:
        """
        Insert a memory chunk.

        Args:
            chunk: MemoryChunk object to insert

        Returns:
            chunk_id of the inserted chunk
        """
        conn = await self._get_connection()
        cursor = await conn.cursor()

        # Serialize JSON fields
        embedding_json = json.dumps(chunk.embedding)
        metadata_json = json.dumps(chunk.metadata)

        await cursor.execute("""
            INSERT OR REPLACE INTO memory_chunks
            (chunk_id, node_id, content, embedding, metadata, position)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            chunk.chunk_id,
            chunk.node_id,
            chunk.content,
            embedding_json,
            metadata_json,
            chunk.position,
        ))

        await conn.commit()
        return chunk.chunk_id

    async def get_chunks_for_node(self, node_id: str) -> List[MemoryChunk]:
        """
        Get all chunks for a specific node.

        Args:
            node_id: Node ID to get chunks for

        Returns:
            List of MemoryChunk objects
        """
        conn = await self._get_connection()
        cursor = await conn.cursor()

        await cursor.execute(
            "SELECT * FROM memory_chunks WHERE node_id = ? ORDER BY position",
            (node_id,)
        )
        rows = await cursor.fetchall()

        chunks = []
        for row in rows:
            embedding = json.loads(row["embedding"]) if row["embedding"] else []
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            chunks.append(MemoryChunk(
                chunk_id=row["chunk_id"],
                node_id=row["node_id"],
                content=row["content"],
                embedding=embedding,
                metadata=metadata,
                position=row["position"],
            ))

        return chunks

    async def insert_edge(self, edge: GraphEdge) -> str:
        """
        Insert a graph edge.

        Args:
            edge: GraphEdge object to insert

        Returns:
            edge_id of the inserted edge
        """
        conn = await self._get_connection()
        cursor = await conn.cursor()

        approved_at = edge.approved_at.isoformat() if edge.approved_at else None

        await cursor.execute("""
            INSERT OR REPLACE INTO graph_edges
            (edge_id, source_node_id, target_node_id, edge_type, weight, rationale,
             confidence, is_approved, created_at, approved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            edge.edge_id,
            edge.source_node_id,
            edge.target_node_id,
            edge.edge_type,
            edge.weight,
            edge.rationale,
            edge.confidence,
            edge.is_approved,
            edge.created_at.isoformat(),
            approved_at,
        ))

        await conn.commit()
        return edge.edge_id

    async def insert_proposal(self, proposal: GraphEdge) -> str:
        """
        Insert a graph proposal.

        Args:
            proposal: GraphEdge object to insert as proposal

        Returns:
            proposal_id (using edge_id as proposal_id)
        """
        conn = await self._get_connection()
        cursor = await conn.cursor()

        # Determine if user approval is required (default: True unless confidence > 0.9)
        requires_approval = proposal.confidence < 0.9

        await cursor.execute("""
            INSERT OR REPLACE INTO graph_proposals
            (proposal_id, source_node_id, target_node_id, edge_type, weight,
             confidence, rationale, requires_user_approval, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            proposal.edge_id,  # Use edge_id as proposal_id
            proposal.source_node_id,
            proposal.target_node_id,
            proposal.edge_type,
            proposal.weight,
            proposal.confidence,
            proposal.rationale,
            requires_approval,
            proposal.created_at.isoformat(),
        ))

        await conn.commit()
        return proposal.edge_id

    async def get_pending_proposals(self) -> List[GraphEdge]:
        """
        Get all pending graph proposals.

        Returns:
            List of GraphEdge objects representing proposals
        """
        conn = await self._get_connection()
        cursor = await conn.cursor()

        await cursor.execute("""
            SELECT * FROM graph_proposals
            WHERE requires_user_approval = TRUE
            ORDER BY created_at DESC
        """)
        rows = await cursor.fetchall()

        proposals = []
        for row in rows:
            proposals.append(GraphEdge(
                edge_id=row["proposal_id"],
                source_node_id=row["source_node_id"],
                target_node_id=row["target_node_id"],
                edge_type=row["edge_type"],
                weight=row["weight"],
                rationale=row["rationale"],
                confidence=row["confidence"],
                is_approved=False,  # Proposals are not approved yet
                created_at=datetime.fromisoformat(row["created_at"]),
                approved_at=None,
            ))

        return proposals

    async def approve_proposal(self, proposal_id: str) -> bool:
        """
        Approve a proposal by moving it to graph_edges and deleting the proposal.

        Args:
            proposal_id: ID of the proposal to approve

        Returns:
            True if successful, False if proposal not found
        """
        conn = await self._get_connection()
        cursor = await conn.cursor()

        # Get the proposal
        await cursor.execute("SELECT * FROM graph_proposals WHERE proposal_id = ?", (proposal_id,))
        row = await cursor.fetchone()

        if row is None:
            return False

        # Create edge from proposal
        approved_at = datetime.now()
        edge = GraphEdge(
            edge_id=row["proposal_id"],
            source_node_id=row["source_node_id"],
            target_node_id=row["target_node_id"],
            edge_type=row["edge_type"],
            weight=row["weight"],
            rationale=row["rationale"],
            confidence=row["confidence"],
            is_approved=True,
            created_at=datetime.fromisoformat(row["created_at"]),
            approved_at=approved_at,
        )

        # Insert into graph_edges
        await self.insert_edge(edge)

        # Delete from proposals
        await cursor.execute("DELETE FROM graph_proposals WHERE proposal_id = ?", (proposal_id,))
        await conn.commit()

        return True

    async def get_all_node_embeddings(self) -> List[Tuple[str, List[float]]]:
        """
        Get all primary chunk embeddings for similarity search.

        Returns:
            List of (node_id, embedding) tuples
        """
        conn = await self._get_connection()
        cursor = await conn.cursor()

        # Get nodes with their primary chunk embeddings
        # Primary chunk is either the one referenced by embedding_ref or the first chunk (lowest position)
        await cursor.execute("""
            SELECT 
                vn.node_id,
                COALESCE(
                    (SELECT mc.embedding FROM memory_chunks mc 
                     WHERE mc.node_id = vn.node_id AND mc.chunk_id = vn.embedding_ref
                     LIMIT 1),
                    (SELECT mc.embedding FROM memory_chunks mc 
                     WHERE mc.node_id = vn.node_id 
                     ORDER BY mc.position ASC LIMIT 1)
                ) as embedding
            FROM vault_nodes vn
            WHERE EXISTS (
                SELECT 1 FROM memory_chunks mc WHERE mc.node_id = vn.node_id
            )
        """)
        rows = await cursor.fetchall()

        embeddings = []
        seen_nodes = set()
        for row in rows:
            node_id = row["node_id"]
            # Only take first embedding per node (primary chunk)
            if node_id not in seen_nodes:
                embedding = json.loads(row["embedding"]) if row["embedding"] else []
                if embedding:
                    embeddings.append((node_id, embedding))
                    seen_nodes.add(node_id)

        return embeddings

    async def node_exists(self, node_id: str) -> bool:
        """
        Check if a node exists in the database.

        Args:
            node_id: Node ID to check

        Returns:
            True if node exists, False otherwise
        """
        conn = await self._get_connection()
        cursor = await conn.cursor()

        await cursor.execute("SELECT 1 FROM vault_nodes WHERE node_id = ?", (node_id,))
        row = await cursor.fetchone()
        return row is not None

