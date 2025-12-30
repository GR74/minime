"""Script to rebuild FAISS index from existing database."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from minime.memory.db import AsyncDatabase
from minime.memory.embeddings import EmbeddingModel
from minime.memory.search import MemorySearch

app = typer.Typer(help="Rebuild FAISS index from database")


@app.command()
def rebuild(
    db_path: str = typer.Option("memory.db", help="Path to SQLite database"),
    index_path: str = typer.Option("memory.faiss", help="Path to save FAISS index"),
):
    """
    Rebuild FAISS index from all chunks in the database.
    """
    async def _rebuild():
        db_file = Path(db_path)
        if not db_file.exists():
            typer.echo(f"Database not found: {db_path}", err=True)
            raise typer.Exit(1)

        db = AsyncDatabase(db_path)
        await db._session.connect()

        try:
            typer.echo("Initializing search engine with FAISS...")
            model = EmbeddingModel()
            search = MemorySearch(
                db,
                model,
                use_faiss=True,
                faiss_index_path=index_path,
            )

            if not search.use_faiss:
                typer.echo("ERROR: FAISS not available. Install with: pip install faiss-cpu", err=True)
                raise typer.Exit(1)

            typer.echo("Building FAISS index from database...")
            await search.rebuild_index()

            index_size = search._faiss_store.size() if search._faiss_store else 0
            typer.echo(f"Index built successfully!")
            typer.echo(f"  - Index size: {index_size} vectors")
            typer.echo(f"  - Index saved to: {index_path}")
            typer.echo(f"  - ID map saved to: {index_path}.idmap")

            # Test search
            typer.echo("\nTesting search...")
            results = await search.search("test", k=3)
            typer.echo(f"  - Test search returned {len(results)} results")
            if results:
                typer.echo(f"  - Top result similarity: {results[0][1]:.3f}")

        finally:
            await db._session.close()

    asyncio.run(_rebuild())


if __name__ == "__main__":
    app()

