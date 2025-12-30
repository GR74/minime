"""CLI script to generate graph visualization."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from minime.memory.db import AsyncDatabase, init_db
from minime.memory.embeddings import EmbeddingModel
from minime.memory.visualizer import GraphVisualizer

app = typer.Typer(help="Generate graph visualization of Obsidian vault")


@app.command()
def generate(
    db_path: str = typer.Option("memory.db", help="Path to SQLite database"),
    output: str = typer.Option("graph.html", help="Output file path"),
    format: str = typer.Option("html", help="Output format: html or png"),
    include_proposals: bool = typer.Option(True, help="Include pending similarity proposals"),
    open_browser: bool = typer.Option(False, help="Open HTML in browser after generation"),
):
    """
    Generate graph visualization from the memory database.
    """
    async def _generate():
        # Initialize database
        db_file = Path(db_path)
        if not db_file.exists():
            typer.echo(f"❌ Database not found: {db_path}", err=True)
            typer.echo("Run 'minime index' first to index your vault.", err=True)
            raise typer.Exit(1)

        # Initialize database connection
        db = AsyncDatabase(db_path)
        await db._session.connect()

        try:
            # Create visualizer
            visualizer = GraphVisualizer(db)

            # Load graph data
            typer.echo("Loading graph data...")
            graph_data = await visualizer.load_graph(include_proposals=include_proposals)

            if not graph_data.nodes:
                typer.echo("No nodes found in database. Index your vault first with 'minime index'")
                return

            typer.echo(f"[OK] Loaded {len(graph_data.nodes)} nodes and {len(graph_data.edges)} edges")

            # Generate visualization
            output_file = Path(output)
            if format == "html":
                typer.echo(f"Generating HTML visualization...")
                visualizer.export_html(str(output_file))
                typer.echo(f"[OK] Saved to: {output_file.absolute()}")
                
                if open_browser:
                    import webbrowser
                    webbrowser.open(f"file://{output_file.absolute()}")
            elif format == "png":
                typer.echo(f"Generating PNG image...")
                visualizer.export_image(str(output_file), format="png")
                typer.echo(f"[OK] Saved to: {output_file.absolute()}")
            else:
                typer.echo(f"ERROR: Unknown format: {format}. Use 'html' or 'png'", err=True)
                raise typer.Exit(1)

            # Show statistics
            stats = visualizer.get_stats()
            typer.echo("\nGraph Statistics:")
            typer.echo(f"  Nodes: {stats['total_nodes']}")
            typer.echo(f"  Edges: {stats['total_edges']} (Wikilinks: {stats['wikilink_edges']}, Similarity: {stats['similarity_edges']})")
            typer.echo(f"  Pending Proposals: {stats['pending_proposals']}")
            typer.echo(f"  Avg Connections: {stats['average_connections']}")
            if stats['hub_nodes']:
                typer.echo(f"  Hub Nodes: {', '.join(stats['hub_nodes'])}")

        finally:
            await db._session.close()

    asyncio.run(_generate())


@app.command()
def stats(
    db_path: str = typer.Option("memory.db", help="Path to SQLite database"),
):
    """
    Show graph statistics without generating visualization.
    """
    async def _stats():
        db_file = Path(db_path)
        if not db_file.exists():
            typer.echo(f"❌ Database not found: {db_path}", err=True)
            raise typer.Exit(1)

        db = AsyncDatabase(db_path)
        await db._session.connect()

        try:
            visualizer = GraphVisualizer(db)
            await visualizer.load_graph(include_proposals=True)
            
            stats = visualizer.get_stats()
            
            typer.echo("Graph Statistics\n")
            typer.echo(f"Nodes: {stats['total_nodes']}")
            typer.echo(f"Edges: {stats['total_edges']}")
            typer.echo(f"  - Wikilinks: {stats['wikilink_edges']}")
            typer.echo(f"  - Similarity: {stats['similarity_edges']}")
            typer.echo(f"  - Pending Proposals: {stats['pending_proposals']}")
            typer.echo(f"\nConnections:")
            typer.echo(f"  - Average: {stats['average_connections']}")
            typer.echo(f"  - Max: {stats['max_connections']}")
            
            if stats['hub_nodes']:
                typer.echo(f"\nHub Nodes (most connected):")
                for hub in stats['hub_nodes']:
                    typer.echo(f"  - {hub}")
            
            if stats['domains']:
                typer.echo(f"\nDomains:")
                for domain, count in stats['domains'].items():
                    typer.echo(f"  - {domain}: {count}")

        finally:
            await db._session.close()

    asyncio.run(_stats())


if __name__ == "__main__":
    app()

