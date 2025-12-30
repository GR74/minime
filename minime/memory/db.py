"""SQLite database schema and AsyncDatabase class for MiniMe memory layer."""

import json
import sqlite3
import zlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import aiosqlite

from minime.memory.session import DBSession
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
            embedding_blob BLOB,
            metadata JSON,
            position INTEGER DEFAULT 0,
            FOREIGN KEY (node_id) REFERENCES vault_nodes(node_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_chunks_node_id ON memory_chunks(node_id)")
    
    # Add embedding_blob column if it doesn't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE memory_chunks ADD COLUMN embedding_blob BLOB")
    except sqlite3.OperationalError:
        pass  # Column already exists

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

    # Enable WAL mode for better concurrency
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    
    conn.commit()
    conn.close()


class AsyncDatabase:
    """
    Async database wrapper for SQLite operations with session management.
    
    Uses DBSession internally for connection reuse and transaction management.
    For batch operations, use the session context manager directly.
    """

    def __init__(self, db_path: str):
        """
        Initialize AsyncDatabase.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._session = DBSession(db_path)

    def session(self) -> DBSession:
        """
        Get the database session for batch operations.
        
        Usage:
            async with db.session() as conn:
                await db.insert_node(node1)
                await db.insert_node(node2)
                # Both use same connection, single commit at end
        """
        return self._session

    async def _execute(self, query: str, params: tuple = ()) -> None:
        """Execute a query with commit."""
        conn = await self._session.connect()
        await conn.execute(query, params)
        await self._session.commit()

    async def _fetch(self, query: str, params: tuple = ()) -> list:
        """Fetch rows from a query."""
        cursor = await self._session.execute(query, params)
        return await cursor.fetchall()

    async def _fetchone(self, query: str, params: tuple = ()) -> Optional[aiosqlite.Row]:
        """Fetch a single row from a query."""
        cursor = await self._session.execute(query, params)
        row = await cursor.fetchone()
        return row

    async def close(self) -> None:
        """Close database session."""
        await self._session.close()

    async def insert_node(self, node: VaultNode) -> str:
        """
        Insert or update a vault node.

        Args:
            node: VaultNode object to insert

        Returns:
            node_id of the inserted/updated node
        """
        # Serialize JSON fields
        frontmatter_json = json.dumps(node.frontmatter)
        tags_json = json.dumps(node.tags)
        links_json = json.dumps(node.links)
        backlinks_json = json.dumps(node.backlinks)

        # Use INSERT OR REPLACE for upsert behavior
        await self._session.execute("""
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
        await self._session.commit()

        return node.node_id

    async def get_node(self, node_id: str) -> Optional[VaultNode]:
        """
        Get a vault node by ID.

        Args:
            node_id: Node ID to retrieve

        Returns:
            VaultNode if found, None otherwise
        """
        row = await self._fetchone("SELECT * FROM vault_nodes WHERE node_id = ?", (node_id,))

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
        row = await self._fetchone("SELECT COUNT(*) as count FROM vault_nodes")
        return row["count"] if row else 0

    async def insert_chunk(self, chunk: MemoryChunk) -> str:
        """
        Insert a memory chunk.

        Args:
            chunk: MemoryChunk object to insert

        Returns:
            chunk_id of the inserted chunk
        """
        # Serialize JSON fields
        metadata_json = json.dumps(chunk.metadata)
        
        # FIX #3: Store embedding as compressed float32 array (not pickle)
        import numpy as np
        # Convert to float32 array and compress
        embedding_array = np.array(chunk.embedding, dtype=np.float32)
        embedding_bytes = embedding_array.tobytes()
        embedding_blob = zlib.compress(embedding_bytes)

        await self._session.execute("""
            INSERT OR REPLACE INTO memory_chunks
            (chunk_id, node_id, content, embedding, embedding_blob, metadata, position)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk.chunk_id,
            chunk.node_id,
            chunk.content,
            None,  # Keep JSON field NULL (use blob instead)
            embedding_blob,
            metadata_json,
            chunk.position,
        ))
        await self._session.commit()

        return chunk.chunk_id

    async def get_chunks_for_node(self, node_id: str) -> List[MemoryChunk]:
        """
        Get all chunks for a specific node.

        Args:
            node_id: Node ID to get chunks for

        Returns:
            List of MemoryChunk objects
        """
        rows = await self._fetch(
            "SELECT * FROM memory_chunks WHERE node_id = ? ORDER BY position",
            (node_id,)
        )

        chunks = []
        for row in rows:
            # FIX #3: Read from compressed float32 blob, fallback to JSON
            import numpy as np
            if row["embedding_blob"]:
                embedding_bytes = zlib.decompress(row["embedding_blob"])
                embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
                embedding = embedding_array.tolist()
            elif row["embedding"]:
                embedding = json.loads(row["embedding"])
            else:
                embedding = []
            
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
        approved_at = edge.approved_at.isoformat() if edge.approved_at else None

        await self._session.execute("""
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
        await self._session.commit()

        return edge.edge_id

    async def insert_proposal(self, proposal: GraphEdge) -> str:
        """
        Insert a graph proposal.

        Args:
            proposal: GraphEdge object to insert as proposal

        Returns:
            proposal_id (using edge_id as proposal_id)
        """
        # Determine if user approval is required (default: True unless confidence > 0.9)
        requires_approval = proposal.confidence < 0.9

        await self._session.execute("""
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
        await self._session.commit()

        return proposal.edge_id

    async def get_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: str = None,
    ) -> Optional[GraphEdge]:
        """
        Get an edge between two nodes.

        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            edge_type: Optional edge type filter

        Returns:
            GraphEdge if found, None otherwise
        """
        if edge_type:
            row = await self._fetchone(
                "SELECT * FROM graph_edges WHERE source_node_id = ? AND target_node_id = ? AND edge_type = ?",
                (source_node_id, target_node_id, edge_type),
            )
        else:
            row = await self._fetchone(
                "SELECT * FROM graph_edges WHERE source_node_id = ? AND target_node_id = ?",
                (source_node_id, target_node_id),
            )

        if row is None:
            return None

        approved_at = datetime.fromisoformat(row["approved_at"]) if row["approved_at"] else None

        return GraphEdge(
            edge_id=row["edge_id"],
            source_node_id=row["source_node_id"],
            target_node_id=row["target_node_id"],
            edge_type=row["edge_type"],
            weight=row["weight"],
            rationale=row["rationale"],
            confidence=row["confidence"],
            is_approved=bool(row["is_approved"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            approved_at=approved_at,
        )

    async def get_proposal(self, proposal_id: str) -> Optional[GraphEdge]:
        """
        Get a proposal by ID.

        Args:
            proposal_id: Proposal ID

        Returns:
            GraphEdge if found, None otherwise
        """
        row = await self._fetchone("SELECT * FROM graph_proposals WHERE proposal_id = ?", (proposal_id,))

        if row is None:
            return None

        return GraphEdge(
            edge_id=row["proposal_id"],
            source_node_id=row["source_node_id"],
            target_node_id=row["target_node_id"],
            edge_type=row["edge_type"],
            weight=row["weight"],
            rationale=row["rationale"],
            confidence=row["confidence"],
            is_approved=False,  # Proposals are not approved
            created_at=datetime.fromisoformat(row["created_at"]),
            approved_at=None,
        )

    async def get_pending_proposals(self) -> List[GraphEdge]:
        """
        Get all pending graph proposals.

        Returns:
            List of GraphEdge objects representing proposals
        """
        rows = await self._fetch("""
            SELECT * FROM graph_proposals
            WHERE requires_user_approval = TRUE
            ORDER BY created_at DESC
        """)

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
        # Get the proposal
        row = await self._fetchone("SELECT * FROM graph_proposals WHERE proposal_id = ?", (proposal_id,))

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

        # Insert into graph_edges and delete from proposals in a transaction
        conn = await self._session.connect()
        # Insert edge
        await conn.execute("""
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
            approved_at.isoformat(),
        ))

        # Delete proposal
        await conn.execute("DELETE FROM graph_proposals WHERE proposal_id = ?", (proposal_id,))
        await self._session.commit()

        return True

    async def get_all_node_embeddings(self) -> List[Tuple[str, List[float]]]:
        """
        Get all primary chunk embeddings for similarity search (legacy, without metadata).

        Returns:
            List of (node_id, embedding) tuples
        """
        results = await self.get_all_node_embeddings_with_metadata()
        return [(node_id, embedding) for node_id, embedding, _ in results]

    async def get_all_node_embeddings_with_metadata(self) -> List[Tuple[str, List[float], Optional[dict]]]:
        """
        Get all primary chunk embeddings for similarity search with metadata.

        Returns:
            List of (node_id, embedding, metadata) tuples
        """
        # Get nodes with their primary chunk embeddings
        # Primary chunk is either the one referenced by embedding_ref or the first chunk (lowest position)
        cursor = await self._session.execute("""
            SELECT 
                vn.node_id,
                COALESCE(
                    (SELECT mc.chunk_id FROM memory_chunks mc 
                     WHERE mc.node_id = vn.node_id AND mc.chunk_id = vn.embedding_ref
                     LIMIT 1),
                    (SELECT mc.chunk_id FROM memory_chunks mc 
                     WHERE mc.node_id = vn.node_id 
                     ORDER BY mc.position ASC LIMIT 1)
                ) as chunk_id
            FROM vault_nodes vn
            WHERE EXISTS (
                SELECT 1 FROM memory_chunks mc WHERE mc.node_id = vn.node_id
            )
        """)
        node_rows = await cursor.fetchall()

        embeddings = []
        seen_nodes = set()
        
        for node_row in node_rows:
            node_id = node_row["node_id"]
            chunk_id = node_row["chunk_id"]
            
            if node_id not in seen_nodes and chunk_id:
                # Get the chunk with embedding and metadata
                chunk_row = await self._fetchone(
                    "SELECT embedding_blob, embedding, metadata FROM memory_chunks WHERE chunk_id = ?",
                    (chunk_id,)
                )
                
                if chunk_row:
                    # FIX #3: Read from compressed float32 blob, fallback to JSON
                    import numpy as np
                    if chunk_row["embedding_blob"]:
                        embedding_bytes = zlib.decompress(chunk_row["embedding_blob"])
                        embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
                        embedding = embedding_array.tolist()
                    elif chunk_row["embedding"]:
                        embedding = json.loads(chunk_row["embedding"])
                    else:
                        embedding = []
                    
                    # Parse metadata
                    metadata = json.loads(chunk_row["metadata"]) if chunk_row["metadata"] else {}
                    
                    if embedding:
                        embeddings.append((node_id, embedding, metadata))
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
        row = await self._fetchone("SELECT 1 FROM vault_nodes WHERE node_id = ?", (node_id,))
        return row is not None

    async def get_node_by_title(self, title: str) -> Optional[str]:
        """
        Get node_id by title (for wikilink resolution).

        Args:
            title: Title to search for

        Returns:
            node_id if found, None otherwise
        """
        row = await self._fetchone("SELECT node_id FROM vault_nodes WHERE title = ?", (title,))
        return row["node_id"] if row else None

    async def get_all_nodes(self) -> List[VaultNode]:
        """
        Get all vault nodes.

        Returns:
            List of all VaultNode objects
        """
        rows = await self._fetch("SELECT * FROM vault_nodes ORDER BY title")

        nodes = []
        for row in rows:
            # Deserialize JSON fields
            frontmatter = json.loads(row["frontmatter"]) if row["frontmatter"] else {}
            tags = json.loads(row["tags"]) if row["tags"] else []
            links = json.loads(row["links"]) if row["links"] else []
            backlinks = json.loads(row["backlinks"]) if row["backlinks"] else []

            nodes.append(VaultNode(
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
            ))

        return nodes

    async def get_all_edges(self, include_proposals: bool = False) -> List[GraphEdge]:
        """
        Get all graph edges (and optionally proposals).

        Args:
            include_proposals: If True, also include pending proposals

        Returns:
            List of all GraphEdge objects
        """
        edges = []

        # Get approved edges
        rows = await self._fetch("SELECT * FROM graph_edges WHERE is_approved = TRUE")
        for row in rows:
            approved_at = datetime.fromisoformat(row["approved_at"]) if row["approved_at"] else None
            edges.append(GraphEdge(
                edge_id=row["edge_id"],
                source_node_id=row["source_node_id"],
                target_node_id=row["target_node_id"],
                edge_type=row["edge_type"],
                weight=row["weight"],
                rationale=row["rationale"],
                confidence=row["confidence"],
                is_approved=True,
                created_at=datetime.fromisoformat(row["created_at"]),
                approved_at=approved_at,
            ))

        # Optionally include proposals
        if include_proposals:
            proposals = await self.get_pending_proposals()
            edges.extend(proposals)

        return edges

