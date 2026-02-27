"""
main.py
────────
CLI entry point for the BI Query Agent.

Usage
-----
# Run a single query (interactive mode)
python main.py query "What were our top 10 products by revenue last quarter?"

# Seed the sample database (SQLite)
python main.py seed

# Ingest and index the schema
python main.py ingest

# Start the MCP server (stdio)
python main.py serve

Options
-------
--db-name  : Override the target database name
--no-stream: Suppress step-by-step streaming output
"""

from __future__ import annotations

import uuid

import structlog
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

load_dotenv()

app = typer.Typer(help="BI Query Agent — AI-powered SQL generation over your database")
console = Console()
logger = structlog.get_logger(__name__)


# ── seed command ──────────────────────────────────────────────────────────────

@app.command()
def seed():
    """Seed a sample SQLite database with e-commerce data for demonstration."""
    from database.seed import seed_sample_database
    console.print("[bold]Seeding sample database...[/bold]")
    seed_sample_database()
    console.print("[green]✓ Sample database ready at sample.db[/green]")


# ── ingest command ────────────────────────────────────────────────────────────

@app.command()
def ingest(
    db_name: str = typer.Option("sample", help="Database name to use as storage key"),
    force: bool = typer.Option(False, help="Force re-indexing even if already indexed"),
):
    """Extract and index the database schema into ChromaDB."""
    from database.connector import DatabaseConnector
    from database.schema_extractor import SchemaExtractor
    from mcp_server.client import MCPClient
    from retrieval.schema_retriever import SchemaIndexer

    console.print("[bold]Starting schema ingestion...[/bold]")

    connector = DatabaseConnector()
    extractor = SchemaExtractor(connector)
    schema_map = extractor.extract()

    # Print schema summary
    tbl = Table(title=f"Schema: {schema_map.db_name}")
    tbl.add_column("Table", style="cyan")
    tbl.add_column("Columns", justify="right")
    tbl.add_column("Rows (approx)", justify="right")
    for t in schema_map.tables:
        tbl.add_row(t.name, str(len(t.columns)), str(t.row_count or "?"))
    console.print(tbl)

    # Store in MCP
    mcp = MCPClient()
    mcp.store_schema(db_name, schema_map.to_dict())
    console.print(f"[green]✓ Schema stored in MCP (key: {db_name})[/green]")

    # Index in ChromaDB
    indexer = SchemaIndexer()
    n = indexer.index(schema_map, force_reindex=force)
    console.print(f"[green]✓ {n} tables indexed in ChromaDB[/green]")


# ── query command ─────────────────────────────────────────────────────────────

@app.command()
def query(
    question: str = typer.Argument(..., help="Natural-language business question"),
    db_name: str = typer.Option("sample", help="Logical database name"),
    no_stream: bool = typer.Option(False, help="Disable step-by-step output"),
):
    """Ask a natural-language question about your database."""
    from mcp_server.client import MCPClient
    from graph.workflow import run_workflow

    mcp = MCPClient()

    # Create session in MCP
    session_result = mcp.create_session(user_question=question, db_name=db_name)
    session_id = session_result["session_id"]

    console.print(Panel(
        f"[bold]Session:[/bold] {session_id}\n"
        f"[bold]Database:[/bold] {db_name}\n"
        f"[bold]Question:[/bold] {question}",
        title="[cyan]BI Query Agent[/cyan]",
    ))

    # Run the full multi-agent workflow
    final_state = run_workflow(
        user_question=question,
        db_name=db_name,
        session_id=session_id,
        stream_output=not no_stream,
    )

    # ── Display results ───────────────────────────────────────────────────────
    console.print("\n" + "=" * 60)

    if final_state.get("validation_passed"):
        # Show final SQL
        if final_state.get("final_sql"):
            console.print(Panel(
                Syntax(final_state["final_sql"], "sql", theme="monokai"),
                title="[green]Final SQL[/green]",
            ))

        # Show explanation
        if final_state.get("final_explanation"):
            console.print(Panel(
                final_state["final_explanation"],
                title="[cyan]Analysis[/cyan]",
            ))

        # Show QA findings summary
        findings = final_state.get("qa_findings", [])
        if findings:
            console.print(f"[yellow]ℹ QA Findings: {len(findings)} item(s)[/yellow]")
            for f in findings:
                icon = "⚠" if f["severity"] in ("warning", "error") else "ℹ"
                console.print(f"  {icon} [{f['severity']}] {f['description']}")
    else:
        error = final_state.get("error_message") or "Unknown error"
        console.print(Panel(
            f"[red]{error}[/red]",
            title="[red]Workflow Failed[/red]",
        ))


# ── serve command ─────────────────────────────────────────────────────────────

@app.command()
def serve(
    transport: str = typer.Option("stdio", help="MCP transport: stdio | sse"),
    host: str = typer.Option("localhost", help="SSE host (if transport=sse)"),
    port: int = typer.Option(8000, help="SSE port (if transport=sse)"),
):
    """Start the FastMCP server."""
    from mcp_server.server import mcp
    console.print(f"[bold]Starting MCP server (transport={transport})[/bold]")
    if transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run(transport="stdio")


# ── demo command ──────────────────────────────────────────────────────────────

@app.command()
def demo():
    """Run a full end-to-end demo with the sample database."""
    console.print("[bold cyan]Running full demo...[/bold cyan]")
    seed()

    demo_questions = [
        "What are the top 5 customers by total order value?",
        "Show me monthly revenue for the last 6 months",
        "Which product categories have the highest return rate?",
    ]

    for q in demo_questions:
        console.rule(f"[yellow]{q}[/yellow]")
        typer.echo("")
        from mcp_server.client import MCPClient
        from graph.workflow import run_workflow

        mcp = MCPClient()
        session = mcp.create_session(user_question=q, db_name="sample")
        run_workflow(
            user_question=q,
            db_name="sample",
            session_id=session["session_id"],
            stream_output=True,
        )
        typer.echo("")


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
