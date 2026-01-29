# PRIVACY: Must be first import
from src.utils.privacy import disable_telemetry

disable_telemetry()


from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.core.aegean_consensus import AegeanConsensusProtocol, ConsensusResult
from src.core.emperor_brain import EmperorBrain
from src.infrastructure.database import Database
from src.infrastructure.vector_store import VectorStore
from src.memory.condensation import CondensationManager
from src.memory.context_builder import ContextBuilder
from src.memory.semantic import SemanticMemory
from src.utils.config import load_config

console = Console()

DEFAULT_USER_ID = "default_user"

PROFILE_SYNTHESIS_PROMPT = """profile_synthesis: |
  Based on the following condensed conversation summaries and psychological insights, generate a comprehensive psychological profile.

  ## Condensed Conversation History
  {condensed_history}

  ## Additional Semantic Insights
  {insights}

  ## Session Count
  {session_count} sessions analyzed

  ## Instructions
  Create a structured profile including:
  1. **Core Patterns**: Recurring themes, behaviors, and thought patterns (note frequency and intensity from summaries)
  2. **Emotional Landscape**: Predominant emotional states and triggers
  3. **Stoic Assessment**: How well the person embodies Stoic principles
  4. **Growth Areas**: Areas for development and recommended focus
  5. **Strengths**: Identified resilience factors and positive patterns
  6. **Temporal Trends**: How patterns have evolved over time

  Write as Marcus Aurelius would assess a student of Stoicism.
"""


def show_latest_profile(user_id: str) -> bool:
    config = load_config()
    db = Database(config["database"]["url"])

    profile = db.get_latest_profile(user_id)
    if not profile:
        console.print("[yellow]No profile found. Run analysis first.[/yellow]")
        return False

    console.print(
        Panel.fit(
            f"[bold]PSYCHOLOGICAL PROFILE[/bold]\n[dim]Version {profile['version']} • {profile['created_at']}[/dim]",
            border_style="blue",
        )
    )

    if profile.get("consensus_log"):
        log = profile["consensus_log"]
        if log.get("consensus_reached"):
            console.print("[green]✅ Consensus reached[/green]")
        else:
            console.print("[yellow]⚠️ No consensus[/yellow]")
        if log.get("stability_score"):
            console.print(f"[dim]Stability score: {log['stability_score']:.2f}[/dim]")

    console.print()
    console.print(
        Panel(Markdown(profile["content"]), title="[bold blue]Profile[/bold blue]", border_style="blue", padding=(1, 2))
    )
    return True


def main(user_id: str = DEFAULT_USER_ID, force: bool = False, show: bool = False) -> None:
    if show:
        show_latest_profile(user_id)
        return
    config = load_config()
    db = Database(config["database"]["url"])
    vectors = VectorStore(config["database"]["url"])
    brain = EmperorBrain(config=config)
    condensation = CondensationManager(db, config)
    _ = ContextBuilder(db, config)

    sessions_since = db.count_sessions_since_last_analysis(user_id)
    threshold = config.get("aegean_consensus", {}).get("sessions_between_analysis", 5)

    if not force and sessions_since < threshold:
        console.print(f"[yellow]Only {sessions_since} sessions since last analysis (threshold: {threshold}).[/yellow]")
        console.print("[dim]Use --force to run anyway.[/dim]")
        return

    console.print(
        Panel.fit("[bold]PSYCHOLOGICAL ANALYSIS[/bold]\n[dim]Aegean Consensus Protocol[/dim]", border_style="blue")
    )

    console.print("\n[dim]Processing unprocessed messages...[/dim]")
    semantic = SemanticMemory(db, vectors, brain)
    processed = semantic.process_unprocessed_messages(user_id)
    console.print(f"[green]Processed {processed} messages into semantic memory.[/green]")

    console.print("\n[dim]Condensing conversation history...[/dim]")
    if condensation.should_condense(user_id):
        condensation.maybe_condense(user_id, verbose=True)

    summaries = db.get_condensed_summaries(user_id)
    condensed_history = ""
    if summaries:
        summaries.sort(key=lambda s: s.period_start)
        condensed_history = "\n\n".join(
            [
                f"### Period: {s.period_start.strftime('%Y-%m-%d')} to {s.period_end.strftime('%Y-%m-%d')} "
                f"(Level {s.level}, {s.source_message_count} messages, {s.source_word_count} words)\n{s.content}"
                for s in summaries
            ]
        )
        console.print(f"[green]Found {len(summaries)} condensed summaries.[/green]")
    else:
        recent_messages = db.get_recent_messages(user_id, limit=50)
        if recent_messages:
            condensed_history = "\n\n".join(
                [
                    f"[{msg.created_at.strftime('%Y-%m-%d %H:%M')}] {msg.role.upper()}: {msg.content}"
                    for msg in recent_messages
                ]
            )
            console.print("[yellow]No condensed summaries yet. Using recent messages.[/yellow]")
        else:
            console.print("[yellow]No conversation history found.[/yellow]")
            return

    insights = db.get_user_insights(user_id)
    insights_text = ""
    if insights:
        insights_text = "\n".join([f"- {i.assertion} (confidence: {i.confidence:.2f})" for i in insights])
        console.print(f"[dim]Found {len(insights)} semantic insights.[/dim]")

    console.print("\n[dim]Running consensus analysis...[/dim]")

    prompts = {"profile_synthesis": PROFILE_SYNTHESIS_PROMPT}

    consensus = AegeanConsensusProtocol(
        model_a=config["models"]["main"],
        model_b=config["models"]["reviewer"],
        prompts=prompts,
        beta_threshold=config.get("aegean_consensus", {}).get("beta_threshold", 2),
        verbose=True,
    )

    with console.status("[dim]Dual-model consensus in progress...[/dim]"):
        result = consensus.reach_consensus(
            prompt_name="profile_synthesis",
            variables={
                "condensed_history": condensed_history,
                "insights": insights_text if insights_text else "None",
                "session_count": sessions_since,
                "source_data": condensed_history,
            },
            critical_constructs=["attachment patterns", "defense mechanisms", "core beliefs"],
        )

    _save_profile(db, user_id, result)

    console.print()
    if result.consensus_reached:
        console.print("[green]✅ Consensus reached[/green]")
    else:
        console.print("[yellow]⚠️ No consensus - using primary model output[/yellow]")

    console.print(f"[dim]Stability score: {result.stability_score:.2f}[/dim]")

    if result.critical_flags:
        console.print("[red]Critical flags:[/red]")
        for flag in result.critical_flags:
            console.print(f"  - {flag}")

    console.print()
    console.print(
        Panel(
            Markdown(result.final_output),
            title="[bold blue]Psychological Profile[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )


def _save_profile(db: Database, user_id: str, result: ConsensusResult) -> None:
    db.save_profile(user_id, result.final_output, result.to_dict())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run psychological analysis with Aegean consensus")
    parser.add_argument("--user", default=DEFAULT_USER_ID, help="User ID")
    parser.add_argument("--force", action="store_true", help="Force analysis regardless of session count")
    parser.add_argument("--show", action="store_true", help="Show latest profile without running analysis")
    args = parser.parse_args()

    main(user_id=args.user, force=args.force, show=args.show)
