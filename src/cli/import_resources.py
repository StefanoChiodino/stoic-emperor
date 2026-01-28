# PRIVACY: Must be first import
from src.utils.privacy import disable_telemetry
disable_telemetry()

import uuid
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from src.infrastructure.database import Database
from src.infrastructure.vector_store import VectorStore
from src.infrastructure.ingestion_pipeline import IngestionPipeline
from src.memory.semantic import SemanticMemory
from src.core.emperor_brain import EmperorBrain
from src.models.schemas import Session, Message
from src.utils.config import load_config

console = Console()

DEFAULT_USER_ID = "default_user"


def import_journaling(
    path: str,
    user_id: str = DEFAULT_USER_ID,
    trigger_analysis: bool = False
) -> None:
    config = load_config()
    db = Database(config["database"]["url"])
    vectors = VectorStore(config["database"]["url"])
    brain = EmperorBrain(config=config)

    file_path = Path(path)

    if not file_path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        return

    console.print(Panel.fit(
        f"[bold]IMPORTING RESOURCE[/bold]\n"
        f"[dim]{file_path.name}[/dim]",
        border_style="blue"
    ))

    user = db.get_or_create_user(user_id)

    if file_path.is_file():
        _import_single_file(file_path, user.id, db, vectors, brain)
    elif file_path.is_dir():
        for f in file_path.glob("**/*.md"):
            _import_single_file(f, user.id, db, vectors, brain)
        for f in file_path.glob("**/*.txt"):
            _import_single_file(f, user.id, db, vectors, brain)

    console.print("\n[dim]Processing imported content into semantic memory...[/dim]")
    semantic = SemanticMemory(db, vectors, brain)
    processed = semantic.process_unprocessed_messages(user_id)
    console.print(f"[green]Extracted {processed} insights from imported content.[/green]")

    if trigger_analysis:
        console.print("\n[dim]Triggering re-analysis...[/dim]")
        from src.cli.analyze import main as run_analysis
        run_analysis(user_id=user_id, force=True)


def _import_single_file(
    file_path: Path,
    user_id: str,
    db: Database,
    vectors: VectorStore,
    brain: EmperorBrain
) -> None:
    console.print(f"  Importing: {file_path.name}")

    content = file_path.read_text(encoding="utf-8")

    if not content.strip():
        console.print(f"    [yellow]Empty file, skipping[/yellow]")
        return

    session = Session(
        user_id=user_id,
        metadata={"source": "import", "file": str(file_path)}
    )
    db.create_session(session)

    user_msg = Message(
        session_id=session.id,
        role="user",
        content=f"[Imported journaling entry from {file_path.name}]\n\n{content}"
    )
    db.save_message(user_msg)

    with console.status("[dim]Generating analysis...[/dim]"):
        response = brain.respond(
            user_message=content,
            conversation_history=[],
            retrieved_context=None
        )

    emperor_msg = Message(
        session_id=session.id,
        role="emperor",
        content=response.response_text,
        psych_update=response.psych_update
    )
    db.save_message(emperor_msg)

    vectors.add(
        collection="episodic",
        ids=[str(uuid.uuid4())],
        documents=[content],
        metadatas=[{
            "user_id": user_id,
            "session_id": session.id,
            "type": "imported_journal",
            "source_file": str(file_path)
        }]
    )

    console.print(f"    [green]✓ Imported and analyzed[/green]")


def import_stoic_texts(
    path: str,
    author: str,
    work: str,
    tag: bool = True
) -> None:
    config = load_config()
    vectors = VectorStore(config["database"]["url"])
    pipeline = IngestionPipeline(vectors, config=config)

    file_path = Path(path)

    console.print(Panel.fit(
        f"[bold]IMPORTING STOIC TEXT[/bold]\n"
        f"[dim]{author} - {work}[/dim]",
        border_style="blue"
    ))

    with console.status("[dim]Processing and tagging...[/dim]"):
        if file_path.is_file():
            count = pipeline.ingest_stoic_text(str(file_path), author, work, tag_with_llm=tag)
        else:
            count = pipeline.ingest_directory(str(file_path), "stoic_wisdom", author, work, tag_with_llm=tag)

    console.print(f"[green]✓ Imported {count} chunks into stoic_wisdom collection[/green]")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Import resources into Stoic Emperor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    journal_parser = subparsers.add_parser("journal", help="Import journaling entries")
    journal_parser.add_argument("path", help="Path to file or directory")
    journal_parser.add_argument("--user", default=DEFAULT_USER_ID, help="User ID")
    journal_parser.add_argument("--analyze", action="store_true", help="Trigger re-analysis after import")

    stoic_parser = subparsers.add_parser("stoic", help="Import Stoic texts")
    stoic_parser.add_argument("path", help="Path to file or directory")
    stoic_parser.add_argument("--author", required=True, help="Author name")
    stoic_parser.add_argument("--work", required=True, help="Work title")
    stoic_parser.add_argument("--no-tag", action="store_true", help="Skip LLM tagging")

    args = parser.parse_args()

    if args.command == "journal":
        import_journaling(args.path, args.user, args.analyze)
    elif args.command == "stoic":
        import_stoic_texts(args.path, args.author, args.work, tag=not args.no_tag)


if __name__ == "__main__":
    main()
