"""CLI script for exploring the memory system."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from minime.memory.db import AsyncDatabase
from minime.memory.embeddings import EmbeddingModel
from minime.memory.search import MemorySearch
from minime.memory.visualizer import GraphVisualizer

app = typer.Typer(help="Explore the MiniMe memory system")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    db_path: str = typer.Option("memory.db", help="Path to SQLite database"),
    k: int = typer.Option(5, help="Number of results to return"),
):
    """
    Search the vault semantically.
    """
    async def _search():
        db_file = Path(db_path)
        if not db_file.exists():
            typer.echo(f"❌ Database not found: {db_path}", err=True)
            raise typer.Exit(1)

        db = AsyncDatabase(db_path)
        await db._session.connect()

        try:
            # Initialize search with FAISS
            embedding_model = EmbeddingModel()
            search_engine = MemorySearch(
                db,
                embedding_model,
                use_faiss=True,
                faiss_index_path=db_path.replace(".db", ".faiss")
            )

            typer.echo(f"🔍 Searching for: '{query}'\n")

            # Perform search
            results = await search_engine.search(query, k=k)

            if not results:
                typer.echo("No results found.")
                return

            typer.echo(f"Found {len(results)} results:\n")
            for i, (chunk, score) in enumerate(results, 1):
                node = await db.get_node(chunk.node_id)
                node_title = node.title if node else chunk.node_id[:8]
                
                typer.echo(f"{i}. [{node_title}] (similarity: {score:.3f})")
                typer.echo(f"   {chunk.content[:150]}...")
                typer.echo()

        finally:
            await db._session.close()

    asyncio.run(_search())


@app.command()
def stats(
    db_path: str = typer.Option("memory.db", help="Path to SQLite database"),
):
    """
    Show memory system statistics.
    """
    async def _stats():
        db_file = Path(db_path)
        if not db_file.exists():
            typer.echo(f"❌ Database not found: {db_path}", err=True)
            raise typer.Exit(1)

        db = AsyncDatabase(db_path)
        await db._session.connect()

        try:
            node_count = await db.count_nodes()
            
            # Count chunks
            cursor = await db._session.execute("SELECT COUNT(*) as count FROM memory_chunks")
            row = await cursor.fetchone()
            chunk_count = row["count"] if row else 0

            # Count edges
            cursor = await db._session.execute("SELECT COUNT(*) as count FROM graph_edges WHERE is_approved = TRUE")
            row = await cursor.fetchone()
            edge_count = row["count"] if row else 0

            # Count proposals
            proposals = await db.get_pending_proposals()
            proposal_count = len(proposals)

            typer.echo("📊 Memory System Statistics\n")
            typer.echo(f"Nodes (Notes): {node_count}")
            typer.echo(f"Chunks: {chunk_count}")
            typer.echo(f"Edges: {edge_count}")
            typer.echo(f"Pending Proposals: {proposal_count}")

        finally:
            await db._session.close()

    asyncio.run(_stats())


@app.command()
def node(
    title: str = typer.Argument(..., help="Node title to search for"),
    db_path: str = typer.Option("memory.db", help="Path to SQLite database"),
):
    """
    Show details for a specific node.
    """
    async def _node():
        db_file = Path(db_path)
        if not db_file.exists():
            typer.echo(f"❌ Database not found: {db_path}", err=True)
            raise typer.Exit(1)

        db = AsyncDatabase(db_path)
        await db._session.connect()

        try:
            # Find node by title
            node_id = await db.get_node_by_title(title)
            if not node_id:
                typer.echo(f"❌ Node not found: '{title}'", err=True)
                raise typer.Exit(1)

            node = await db.get_node(node_id)
            if not node:
                typer.echo(f"❌ Node data not found", err=True)
                raise typer.Exit(1)

            # Get chunks
            chunks = await db.get_chunks_for_node(node_id)

            # Get edges
            cursor = await db._session.execute(
                "SELECT * FROM graph_edges WHERE (source_node_id = ? OR target_node_id = ?) AND is_approved = TRUE",
                (node_id, node_id)
            )
            edges = await cursor.fetchall()

            typer.echo(f"📄 Node: {node.title}\n")
            typer.echo(f"Path: {node.path}")
            typer.echo(f"Domain: {node.domain or 'none'}")
            typer.echo(f"Scope: {node.scope}")
            if node.tags:
                typer.echo(f"Tags: {', '.join(node.tags)}")
            typer.echo(f"Created: {node.created_at}")
            typer.echo(f"Updated: {node.updated_at}")
            typer.echo(f"\nChunks: {len(chunks)}")
            typer.echo(f"Connections: {len(edges)}")

            if edges:
                typer.echo("\nConnections:")
                for edge_row in edges[:10]:  # Show first 10
                    other_id = edge_row["target_node_id"] if edge_row["source_node_id"] == node_id else edge_row["source_node_id"]
                    other_node = await db.get_node(other_id)
                    other_title = other_node.title if other_node else other_id[:8]
                    typer.echo(f"  - {other_title} ({edge_row['edge_type']})")

        finally:
            await db._session.close()

    asyncio.run(_node())


@app.command()
def explore(
    db_path: str = typer.Option("memory.db", help="Path to SQLite database"),
):
    """
    Interactive exploration mode (simple menu).
    """
    async def _explore():
        db_file = Path(db_path)
        if not db_file.exists():
            typer.echo(f"❌ Database not found: {db_path}", err=True)
            raise typer.Exit(1)

        db = AsyncDatabase(db_path)
        await db._session.connect()

        try:
            embedding_model = EmbeddingModel()
            search_engine = MemorySearch(db, embedding_model)

            typer.echo("🔍 Memory Explorer - Interactive Mode\n")
            typer.echo("Commands:")
            typer.echo("  search <query> - Search semantically")
            typer.echo("  stats - Show statistics")
            typer.echo("  node <title> - Show node details")
            typer.echo("  quit - Exit\n")

            while True:
                try:
                    command = typer.prompt("> ").strip()
                    
                    if not command or command.lower() in ["quit", "exit", "q"]:
                        if command.lower() in ["quit", "exit", "q"]:
                            break
                        continue

                    parts = command.split(maxsplit=1)
                    cmd = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""

                    if cmd == "search" and args:
                        results = await search_engine.search(args, k=5)
                        if results:
                            for i, (chunk, score) in enumerate(results, 1):
                                node = await db.get_node(chunk.node_id)
                                title = node.title if node else chunk.node_id[:8]
                                typer.echo(f"{i}. [{title}] ({score:.3f})")
                                typer.echo(f"   {chunk.content[:100]}...")
                        else:
                            typer.echo("No results.")
                    elif cmd == "stats":
                        node_count = await db.count_nodes()
                        typer.echo(f"Nodes: {node_count}")
                    elif cmd == "node" and args:
                        node_id = await db.get_node_by_title(args)
                        if node_id:
                            node = await db.get_node(node_id)
                            typer.echo(f"Found: {node.title}")
                        else:
                            typer.echo("Not found.")
                    else:
                        typer.echo("Unknown command or missing arguments.")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    typer.echo(f"Error: {e}")

        finally:
            await db._session.close()

    asyncio.run(_explore())


if __name__ == "__main__":
    app()

