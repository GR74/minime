"""Interactive demo of the MiniMe memory system."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from minime.memory.db import AsyncDatabase, init_db
from minime.memory.embeddings import EmbeddingModel
from minime.memory.search import Memory, MemorySearch
from minime.memory.vault import VaultIndexer
from minime.memory.visualizer import GraphVisualizer

app = typer.Typer(help="Interactive demo of MiniMe memory system")


@app.command()
def run(
    vault_path: str = typer.Option(".", help="Path to Obsidian vault"),
    db_path: str = typer.Option("demo_memory.db", help="Path to SQLite database"),
    reset: bool = typer.Option(False, help="Reset database before demo"),
):
    """
    Run interactive demo of the memory system.
    """
    async def _demo():
        # Initialize database
        if reset and Path(db_path).exists():
            Path(db_path).unlink()
            typer.echo("Reset database")

        init_db(db_path)
        db = AsyncDatabase(db_path)
        await db._session.connect()

        try:
            embedding_model = EmbeddingModel()
            memory = Memory(db, embedding_model)

            typer.echo("MiniMe Memory System Demo\n")
            typer.echo("=" * 60)

            # Step 1: Index vault
            typer.echo("\nStep 1: Indexing Vault")
            typer.echo("-" * 60)
            indexer = VaultIndexer(
                vault_path=vault_path,
                db=db,
                embedding_model=embedding_model,
            )

            nodes = await indexer.index()
            typer.echo(f"[OK] Indexed {len(nodes)} notes")

            if nodes:
                typer.echo("\nSample nodes:")
                for node in nodes[:5]:
                    typer.echo(f"  - {node.title} ({len(node.tags)} tags)")

            # Step 2: Write some memory
            typer.echo("\nStep 2: Writing Memory")
            typer.echo("-" * 60)
            
            sample_memories = [
                "Learned: Use async/await for I/O operations in Python",
                "Important: Always validate user input before processing",
                "Pattern: Dependency injection improves testability",
            ]

            for mem in sample_memories:
                node_id = await memory.write(mem, metadata={"tags": ["demo", "learning"]})
                typer.echo(f"[OK] Wrote: {mem[:50]}...")

            # Step 3: Search
            typer.echo("\nStep 3: Semantic Search")
            typer.echo("-" * 60)

            search_engine = MemorySearch(
                db, 
                embedding_model,
                use_faiss=True,
                faiss_index_path="demo_memory.faiss"
            )
            queries = ["Python async", "testing patterns", "user input"]

            for query in queries:
                typer.echo(f"\nSearching: '{query}'")
                results = await search_engine.search(query, k=3)
                
                if results:
                    for i, (chunk, score) in enumerate(results, 1):
                        node = await db.get_node(chunk.node_id)
                        title = node.title if node else "Unknown"
                        typer.echo(f"  {i}. [{title}] (score: {score:.3f})")
                        typer.echo(f"     {chunk.content[:80]}...")
                else:
                    typer.echo("  No results")

            # Step 4: Graph visualization
            typer.echo("\nStep 4: Graph Visualization")
            typer.echo("-" * 60)

            visualizer = GraphVisualizer(db)
            graph_data = await visualizer.load_graph(include_proposals=True)

            typer.echo(f"[OK] Loaded graph: {len(graph_data.nodes)} nodes, {len(graph_data.edges)} edges")

            stats = visualizer.get_stats()
            typer.echo(f"\nGraph Statistics:")
            typer.echo(f"  Nodes: {stats['total_nodes']}")
            typer.echo(f"  Edges: {stats['total_edges']}")
            typer.echo(f"  Wikilinks: {stats['wikilink_edges']}")
            typer.echo(f"  Similarity: {stats['similarity_edges']}")
            typer.echo(f"  Avg Connections: {stats['average_connections']}")

            # Generate visualization
            output_file = "demo_graph.html"
            visualizer.export_html(output_file)
            typer.echo(f"\n[OK] Generated visualization: {output_file}")
            typer.echo(f"   Open in browser to explore the graph!")

            # Step 5: Memory API
            typer.echo("\nStep 5: Memory API (write/read/link)")
            typer.echo("-" * 60)

            # Write
            node_a = await memory.write("Concept A: Machine Learning Basics")
            node_b = await memory.write("Concept B: Neural Networks")
            typer.echo(f"[OK] Created nodes: {node_a[:8]}... and {node_b[:8]}...")

            # Link
            await memory.link(node_a, node_b, "Concept B builds on Concept A")
            typer.echo("[OK] Created link between concepts")

            # Read
            results = await memory.read("machine learning", k=2)
            typer.echo(f"[OK] Found {len(results)} relevant memories")

            typer.echo("\n" + "=" * 60)
            typer.echo("[OK] Demo complete!")
            typer.echo("\nNext steps:")
            typer.echo("  1. Open demo_graph.html in your browser")
            typer.echo("  2. Try: python scripts/memory_explorer.py search 'your query'")
            typer.echo("  3. Try: python scripts/viz_graph.py generate")

        finally:
            await db._session.close()

    asyncio.run(_demo())


if __name__ == "__main__":
    app()

