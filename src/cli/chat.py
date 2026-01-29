# PRIVACY: Must be first import
from src.utils.privacy import disable_telemetry

disable_telemetry()


from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.core.emperor_brain import EmperorBrain
from src.infrastructure.database import Database
from src.infrastructure.vector_store import VectorStore
from src.memory.condensation import CondensationManager
from src.models.schemas import Message, Session
from src.utils.config import load_config

console = Console()

DEFAULT_USER_ID = "default_user"


def main(user_id: str = DEFAULT_USER_ID, session_id: str | None = None) -> None:
    config = load_config()
    db = Database(config["database"]["url"])
    vectors = VectorStore(config["database"]["url"])
    brain = EmperorBrain(config=config)
    condensation = CondensationManager(db, config)

    user = db.get_or_create_user(user_id)

    if session_id:
        session = db.get_session(session_id)
        if not session:
            console.print(f"[red]Session {session_id} not found. Creating new session.[/red]")
            session = None
    else:
        session = None

    if not session:
        session = Session(user_id=user.id)
        db.create_session(session)

    history = db.get_session_messages(session.id)

    console.print(
        Panel.fit(
            "[bold]MARCUS AURELIUS[/bold]\n"
            "[dim]Stoic Emperor • AI Persona[/dim]\n\n"
            "Type your thoughts. Press Ctrl+D or type 'exit' to end.",
            border_style="blue",
        )
    )

    if history:
        console.print(f"\n[dim]Resuming session with {len(history)} previous messages.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        user_msg = Message(session_id=session.id, role="user", content=user_input)
        db.save_message(user_msg)
        history.append(user_msg)

        retrieved_context = _retrieve_context(vectors, brain, user_input, user.id)

        with console.status("[dim]The Emperor contemplates...[/dim]", spinner="dots"):
            response = brain.respond(
                user_message=user_input, conversation_history=history, retrieved_context=retrieved_context
            )

        emperor_msg = Message(
            session_id=session.id, role="emperor", content=response.response_text, psych_update=response.psych_update
        )
        db.save_message(emperor_msg)
        history.append(emperor_msg)

        console.print()
        console.print(
            Panel(
                Markdown(response.response_text),
                title="[bold blue]Marcus Aurelius[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print()

        if condensation.should_condense(user.id):
            with console.status("[dim]Condensing conversation history...[/dim]", spinner="dots"):
                condensation.maybe_condense(user.id, verbose=False)

    console.print("\n[dim]The Emperor withdraws. May you find virtue in your path.[/dim]\n")


def _retrieve_context(vectors: VectorStore, brain: EmperorBrain, user_message: str, user_id: str) -> dict:
    context = {"stoic": [], "psych": [], "insights": [], "episodic": []}

    try:
        expanded = brain.expand_query(user_message)
        query_terms = [t.strip() for t in expanded.split(",")]
        query_text = " ".join(query_terms) if query_terms else user_message
    except Exception:
        query_text = user_message

    try:
        stoic_results = vectors.query("stoic_wisdom", query_texts=[query_text], n_results=3)
        if stoic_results.get("documents") and stoic_results["documents"][0]:
            context["stoic"] = stoic_results["documents"][0]
    except Exception:
        pass

    try:
        psych_results = vectors.query("psychoanalysis", query_texts=[query_text], n_results=3)
        if psych_results.get("documents") and psych_results["documents"][0]:
            context["psych"] = psych_results["documents"][0]
    except Exception:
        pass

    try:
        insight_results = vectors.query("semantic", query_texts=[query_text], n_results=5, where={"user_id": user_id})
        if insight_results.get("documents") and insight_results["documents"][0]:
            context["insights"] = insight_results["documents"][0]
    except Exception:
        pass

    return context


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chat with Marcus Aurelius")
    parser.add_argument("--user", default=DEFAULT_USER_ID, help="User ID")
    parser.add_argument("--session", default=None, help="Resume specific session ID")
    args = parser.parse_args()

    main(user_id=args.user, session_id=args.session)
