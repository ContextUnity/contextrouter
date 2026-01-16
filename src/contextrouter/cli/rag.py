"""RAG testing commands for validating retrieval and generation."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click

from contextrouter.cli.registry import register_command
from contextrouter.core import get_env

logger = logging.getLogger(__name__)


def _check_env() -> None:
    """Check required environment variables and provide helpful error messages."""
    from contextrouter.core import get_core_config

    # Ensure config is loaded (which loads environment)
    cfg = get_core_config()
    missing = []

    if not get_env("RAG_DB_NAME"):
        missing.append("RAG_DB_NAME")
    if not cfg.vertex.project_id:
        missing.append("VERTEX_PROJECT_ID")

    if missing:
        click.echo(click.style("✗ Missing required environment variables:", fg="red", bold=True))
        for var in missing:
            click.echo(f"  - {var}")
        click.echo("\nHint: Set these in your .env file or environment:")
        click.echo("  export RAG_DB_NAME=green  # or blue")
        click.echo("  export VERTEX_PROJECT_ID=your-project-id")
        raise click.ClickException("Environment configuration incomplete")


@click.group("rag")
def rag() -> None:
    """Test and validate RAG retrieval and generation."""


register_command("rag", rag)


@rag.command("query")
@click.argument("query", type=str)
@click.option("--platform", type=str, default="cli", help="Platform identifier")
@click.option("--session-id", type=str, default="test-session", help="Session ID")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def cmd_query(query: str, platform: str, session_id: str, verbose: bool) -> None:
    """Test a single query through the full RAG pipeline."""
    _check_env()
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    click.echo(click.style(f"\n🔍 Testing Query: {query}", fg="cyan", bold=True))
    click.echo(f"  Platform: {platform}")
    click.echo(f"  Session: {session_id}\n")

    async def run():
        from langchain_core.messages import HumanMessage

        from contextrouter.cortex import invoke_agent

        messages = [HumanMessage(content=query)]
        result = await invoke_agent(
            messages=messages,
            session_id=session_id,
            platform=platform,
            enable_suggestions=False,
        )

        assistant_messages = result.get("messages", [])
        citations = result.get("citations", [])

        if assistant_messages:
            last_msg = assistant_messages[-1]
            content = getattr(last_msg, "content", str(last_msg))
            click.echo(click.style("📝 Response:", fg="green", bold=True))
            click.echo(f"  {content}\n")

        if citations:
            click.echo(click.style(f"📚 Citations ({len(citations)}):", fg="yellow", bold=True))
            for i, cit in enumerate(citations[:5], 1):  # Show first 5
                title = getattr(cit, "title", "Unknown")
                source_type = getattr(cit, "source_type", "unknown")
                click.echo(f"  {i}. [{source_type}] {title}")
            if len(citations) > 5:
                click.echo(f"  ... and {len(citations) - 5} more")
        else:
            click.echo(click.style("⚠️  No citations retrieved", fg="yellow"))

        if verbose:
            retrieved_docs = result.get("retrieved_docs", [])
            if retrieved_docs:
                click.echo(f"\n📊 Retrieved {len(retrieved_docs)} documents")
                for doc in retrieved_docs[:3]:
                    click.echo(f"  - {getattr(doc, 'title', 'Unknown')}")

    asyncio.run(run())


@rag.command("retrieve")
@click.argument("query", type=str)
@click.option("--limit", type=int, default=5, help="Max documents to retrieve")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def cmd_retrieve(query: str, limit: int, verbose: bool) -> None:
    """Test retrieval only (no generation)."""
    _check_env()
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    click.echo(click.style(f"\n🔍 Retrieval Test: {query}", fg="cyan", bold=True))
    click.echo(f"  Limit: {limit}\n")

    async def run():
        from langchain_core.messages import HumanMessage

        from contextrouter.modules.retrieval import RetrievalPipeline

        pipeline = RetrievalPipeline()
        state: dict[str, object] = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "should_retrieve": True,
        }

        result = await pipeline.execute(state)

        click.echo(
            click.style(
                f"📚 Retrieved {len(result.retrieved_docs)} documents", fg="green", bold=True
            )
        )

        for i, doc in enumerate(result.retrieved_docs[:limit], 1):
            title = getattr(doc, "title", "Unknown")
            source_type = getattr(doc, "source_type", "unknown")
            score = getattr(doc, "score", None)
            score_str = f" (score: {score:.3f})" if score else ""
            click.echo(f"  {i}. [{source_type}] {title}{score_str}")

        if result.citations:
            click.echo(f"\n📎 Generated {len(result.citations)} citations")

        if result.graph_facts:
            click.echo(f"\n🔗 Found {len(result.graph_facts)} graph facts")
            if verbose:
                for fact in result.graph_facts[:3]:
                    click.echo(f"  - {fact}")

    asyncio.run(run())


@rag.command("test")
@click.argument("queries-file", type=click.Path(exists=True, path_type=Path))
@click.option("--platform", type=str, default="cli", help="Platform identifier")
@click.option("--output", type=click.Path(path_type=Path), help="Output JSON file for results")
def cmd_test(queries_file: Path, platform: str, output: Path | None) -> None:
    """Run multiple test queries from a JSON file.

    Expected format: {"queries": ["query1", "query2", ...]}
    """
    _check_env()
    with open(queries_file) as f:
        data = json.load(f)

    queries = data.get("queries", [])
    if not queries:
        click.echo(click.style("✗ No queries found in file", fg="red"))
        return

    click.echo(click.style(f"\n🧪 Running {len(queries)} test queries", fg="cyan", bold=True))

    results = []

    async def run_all():
        from langchain_core.messages import HumanMessage

        from contextrouter.cortex import invoke_agent

        for i, query in enumerate(queries, 1):
            click.echo(f"\n[{i}/{len(queries)}] {query}")
            try:
                messages = [HumanMessage(content=query)]
                result = await invoke_agent(
                    messages=messages,
                    session_id=f"test-session-{i}",
                    platform=platform,
                    enable_suggestions=False,
                )

                assistant_messages = result.get("messages", [])
                citations = result.get("citations", [])

                response = ""
                if assistant_messages:
                    last_msg = assistant_messages[-1]
                    response = getattr(last_msg, "content", str(last_msg))

                results.append(
                    {
                        "query": query,
                        "response": response,
                        "citation_count": len(citations),
                        "success": bool(response),
                    }
                )

                status = "✓" if response else "✗"
                click.echo(
                    f"  {status} Response: {len(response)} chars, Citations: {len(citations)}"
                )
            except Exception as e:
                logger.exception("Query failed")
                results.append(
                    {
                        "query": query,
                        "error": str(e),
                        "success": False,
                    }
                )
                click.echo(f"  ✗ Error: {e}")

        if output:
            with open(output, "w") as f:
                json.dump({"results": results}, f, indent=2)
            click.echo(f"\n💾 Results saved to {output}")

        success_count = sum(1 for r in results if r.get("success"))
        click.echo(f"\n📊 Summary: {success_count}/{len(results)} successful")

    asyncio.run(run_all())


@rag.command("benchmark")
@click.argument("queries-file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), help="Output JSON file")
def cmd_benchmark(queries_file: Path, output: Path | None) -> None:
    """Benchmark retrieval performance (timing and citation counts)."""
    _check_env()
    import time

    with open(queries_file) as f:
        data = json.load(f)

    queries = data.get("queries", [])
    if not queries:
        click.echo(click.style("✗ No queries found in file", fg="red"))
        return

    click.echo(click.style(f"\n⏱️  Benchmarking {len(queries)} queries", fg="cyan", bold=True))

    results = []

    async def run_benchmark():
        from langchain_core.messages import HumanMessage

        from contextrouter.modules.retrieval import RetrievalPipeline

        pipeline = RetrievalPipeline()

        for i, query in enumerate(queries, 1):
            click.echo(f"\n[{i}/{len(queries)}] {query}")

            state: dict[str, object] = {
                "messages": [HumanMessage(content=query)],
                "user_query": query,
                "should_retrieve": True,
            }

            t0 = time.perf_counter()
            try:
                result = await pipeline.execute(state)
                elapsed = time.perf_counter() - t0

                results.append(
                    {
                        "query": query,
                        "elapsed_ms": elapsed * 1000,
                        "doc_count": len(result.retrieved_docs),
                        "citation_count": len(result.citations),
                        "graph_facts_count": len(result.graph_facts),
                        "success": True,
                    }
                )

                click.echo(
                    f"  ✓ {elapsed * 1000:.1f}ms - {len(result.retrieved_docs)} docs, {len(result.citations)} citations"
                )
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.exception("Retrieval failed")
                results.append(
                    {
                        "query": query,
                        "elapsed_ms": elapsed * 1000,
                        "error": str(e),
                        "success": False,
                    }
                )
                click.echo(f"  ✗ {elapsed * 1000:.1f}ms - Error: {e}")

        if results:
            avg_time = sum(r["elapsed_ms"] for r in results if r.get("success")) / max(
                1, sum(1 for r in results if r.get("success"))
            )
            avg_docs = sum(r.get("doc_count", 0) for r in results if r.get("success")) / max(
                1, sum(1 for r in results if r.get("success"))
            )

            click.echo(f"\n📊 Average: {avg_time:.1f}ms, {avg_docs:.1f} docs per query")

        if output:
            with open(output, "w") as f:
                json.dump({"benchmark": results}, f, indent=2)
            click.echo(f"💾 Results saved to {output}")

    asyncio.run(run_benchmark())
