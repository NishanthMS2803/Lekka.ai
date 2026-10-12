"""Lekka AI Command Line Interface."""

from __future__ import annotations

import asyncio
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lekka_ai.config import get_settings
from lekka_ai.harness.direct import DirectLLMRunner
from lekka_ai.harness.models import AgentTask, Limits
from lekka_ai.harness.permissions import ToolView
from lekka_ai.harness.profiles import (
    create_analyst_profile,
    load_agent_profile,
)
from lekka_ai.harness.runtime import AgentRunResult, AgentRuntime
from lekka_ai.llm.gemini import GeminiClient
from lekka_ai.tools.base import default_registry

app = typer.Typer(
    name="Lekka AI",
    help="Lekka AI: Multi-agent financial research system CLI.",
    no_args_is_help=True,
)
console = Console()


async def _run_ask(
    question: str,
    as_of: str | None,
    variant: str = "agent",
    agent_type: str = "analyst",
    max_iterations: int = 10,
    timeout: float = 120.0,
) -> AgentRunResult:
    """Execute analyst research query asynchronously."""
    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        console.print(
            "[bold red]Error:[/bold red] GEMINI_API_KEY is not set in settings or .env"
        )
        raise typer.Exit(code=1)

    llm = GeminiClient(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
    )

    task = AgentTask(
        instruction=question,
        as_of_date=as_of,
        agent_type="direct" if variant.lower() == "direct" else agent_type,
    )

    if variant.lower() == "direct":
        return await DirectLLMRunner.run(
            task=task,
            llm=llm,
            as_of_date=as_of,
        )

    try:
        profile = load_agent_profile(agent_type)
    except Exception:
        profile = create_analyst_profile()
    tool_view = ToolView(default_registry, profile=profile)

    limits = Limits(
        max_iterations=max_iterations,
        max_seconds=timeout,
    )

    # Attempt to attach database session if DB is accessible
    session = None
    ledger = None
    try:
        from sqlalchemy import text

        from lekka_ai.db.ledger import LedgerRepository
        from lekka_ai.db.session import get_session

        db_gen = get_session()
        db_session = next(db_gen)
        # Test connectivity
        db_session.execute(text("SELECT 1"))
        session = db_session
        ledger = LedgerRepository(session)
    except Exception:
        # DB not reachable or configured; proceed gracefully in-memory
        session = None
        ledger = None

    try:
        return await AgentRuntime.run(
            task=task,
            profile=profile,
            llm=llm,
            tool_view=tool_view,
            limits=limits,
            session=session,
            ledger=ledger,
        )
    finally:
        if session:
            session.close()


def _format_claims(claims: list[Any]) -> Table:
    """Build a rich table presenting verified claims and evidence."""
    table = Table(
        title="Cited & Verified Claims",
        title_style="bold cyan",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Type", width=14)
    table.add_column("Claim Statement", style="white")
    table.add_column("Evidence & Citations", style="green")

    for i, claim in enumerate(claims, 1):
        claim_type = (
            claim.type.value if hasattr(claim.type, "value") else str(claim.type)
        )
        type_badge = f"[{claim_type.upper()}]"

        evidence_snippets: list[str] = []
        for ev in getattr(claim, "evidence", []):
            parts = [f"source={ev.source_type}"]
            if ev.tool_call_id:
                parts.append(f"tool_id={ev.tool_call_id}")
            if ev.as_of_date:
                parts.append(f"as_of={ev.as_of_date}")
            if ev.raw_value is not None:
                raw_str = str(ev.raw_value)
                if len(raw_str) > 40:
                    raw_str = raw_str[:37] + "..."
                parts.append(f"val={raw_str}")
            evidence_snippets.append(" • " + ", ".join(parts))

        ev_text = (
            "\n".join(evidence_snippets) if evidence_snippets else "[dim]None[/dim]"
        )
        table.add_row(str(i), type_badge, claim.text, ev_text)

    return table


@app.command(name="ask")
def ask(
    question: str = typer.Argument(
        ...,
        help="Financial question or research query to investigate.",
    ),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help="Point-in-time cutoff date (YYYY-MM-DD).",
    ),
    variant: str = typer.Option(
        "agent",
        "--variant",
        help=(
            "Execution variant: 'agent' (default V2 harness with tools) "
            "or 'direct' (V1 direct LLM baseline)."
        ),
    ),
    agent: str = typer.Option(
        "analyst",
        "--agent",
        "-a",
        help="Agent specialist profile to execute ('analyst', 'market').",
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        "-i",
        help="Maximum iterations for the agent runtime.",
    ),
    timeout: float = typer.Option(
        120.0,
        "--timeout",
        "-t",
        help="Maximum run timeout in seconds.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show internal execution metrics and iterations.",
    ),
) -> None:
    """Investigate a financial question using the baseline analyst agent."""
    console.print()
    is_direct = variant.strip().lower() == "direct"
    title_suffix = (
        " (Direct LLM Baseline - V1)" if is_direct else f" ({agent.capitalize()} Agent)"
    )
    header_text = Text()
    header_text.append(f"Lekka AI Financial Analyst{title_suffix}\n", style="bold blue")
    header_text.append(f"Query: {question}\n", style="white")
    if as_of:
        header_text.append(f"Cutoff Date: {as_of}\n", style="yellow")
    console.print(Panel(header_text, border_style="blue"))

    status_msg = (
        "Direct LLM generating response and extracting claims..."
        if is_direct
        else f"{agent.capitalize()} agent investigating using tools..."
    )
    with console.status(f"[bold green]{status_msg}", spinner="dots"):
        try:
            result = asyncio.run(
                _run_ask(
                    question=question,
                    as_of=as_of,
                    variant=variant,
                    agent_type=agent,
                    max_iterations=max_iterations,
                    timeout=timeout,
                )
            )
        except Exception as exc:
            console.print(f"\n[bold red]Error during investigation:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    submission = result.submission

    # Status & Confidence Panel
    status_str = result.status.upper()
    status_style = (
        "green"
        if status_str == "COMPLETED"
        else "yellow"
        if status_str == "PARTIAL"
        else "red"
    )
    conf_pct = (
        f"{int(submission.confidence * 100)}%"
        if submission and submission.confidence is not None
        else "N/A"
    )

    console.print(
        f"\n[bold]Outcome Status:[/bold] "
        f"[{status_style}]{status_str}[/{status_style}]  |  "
        f"[bold]Confidence:[/bold] [cyan]{conf_pct}[/cyan]\n"
    )

    # Answer text / Rationale
    if result.agent_type == "direct" and result.final_text:
        answer_text = result.final_text
    elif submission and submission.confidence_rationale:
        answer_text = submission.confidence_rationale
    else:
        answer_text = result.final_text or "No answer returned."
    console.print(
        Panel(
            answer_text,
            title="[bold green]Analyst Findings & Rationale[/bold green]",
            border_style="green",
        )
    )

    # Answer Extension (e.g. Market Trend & Key Stats)
    if submission and submission.extension:
        ext = submission.extension
        console.print("\n[bold magenta]Answer Extension:[/bold magenta]")
        if hasattr(ext, "trend_label"):
            console.print(f"  • [bold]Trend Label:[/bold] {ext.trend_label}")
        if hasattr(ext, "key_stats") and ext.key_stats:
            console.print("  • [bold]Key Stats:[/bold]")
            stats_dict = (
                ext.key_stats.model_dump()
                if hasattr(ext.key_stats, "model_dump")
                else ext.key_stats
            )
            for k, v in stats_dict.items():
                console.print(f"      - {k}: {v}")
        if hasattr(ext, "scenario_values") and ext.scenario_values:
            console.print("  • [bold]Scenario Values:[/bold]")
            scen_dict = (
                ext.scenario_values.model_dump()
                if hasattr(ext.scenario_values, "model_dump")
                else ext.scenario_values
            )
            for k, v in scen_dict.items():
                console.print(f"      - {k}: {v}")
        if hasattr(ext, "implied_growth") and ext.implied_growth is not None:
            pct_str = f"{ext.implied_growth * 100:.2f}%"
            console.print(f"  • [bold]Implied Growth (Reverse DCF):[/bold] {pct_str}")

    # Claims table
    if submission and submission.claims:
        console.print()
        console.print(_format_claims(submission.claims))
    else:
        console.print(
            "\n[yellow]No structured claims were submitted for this query.[/yellow]"
        )

    # Assumptions & Open Questions if present
    if submission and submission.assumptions:
        console.print("\n[bold cyan]Assumptions:[/bold cyan]")
        for k, v in submission.assumptions.items():
            console.print(f"  • [bold]{k}[/bold]: {v}")

    if submission and submission.open_questions:
        console.print("\n[bold yellow]Open Questions / Evidence Gaps:[/bold yellow]")
        for q in submission.open_questions:
            console.print(f"  • {q}")

    if verbose:
        console.print(
            f"\n[dim]Execution Metrics: {result.iterations} iterations, "
            f"{result.tool_calls_count} tool calls, "
            f"{result.tokens_in} tokens in, {result.tokens_out} tokens out[/dim]"
        )


@app.command(name="version")
def version() -> None:
    """Print the current Lekka AI version."""
    console.print("[bold blue]Lekka AI[/bold blue] version [green]0.1.0[/green]")


def main() -> None:
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
