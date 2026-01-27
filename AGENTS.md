# Stoic Emperor - Agent Guidelines

## Privacy (CRITICAL)

This is a therapy application. User privacy is paramount.

### Telemetry Disabled
All telemetry is disabled via `src/utils/privacy.py`:
- ChromaDB analytics (PostHog)
- Hugging Face Hub telemetry
- Sentence Transformers telemetry

**Every CLI entry point imports `privacy.py` FIRST** before any other module.

### Data Storage
- All data stays local (SQLite + ChromaDB files)
- No cloud sync, no external APIs except the LLM provider
- `.env` file is gitignored

### LLM Provider Considerations
- OpenAI: Consider requesting Zero Data Retention (ZDR) for production
- Self-hosted: Use `LLM_BASE_URL` to point to local LLMs (Ollama, LM Studio)
- Anthropic: Review their data usage policy

### When Adding Dependencies
- **Always add to `requirements.txt` - never run `pip install` directly**
- Audit new packages for telemetry/analytics
- Add disable flags to `src/utils/privacy.py`
- Prefer local-only packages

---

## Project Overview

CLI-first therapy chatbot with Marcus Aurelius as persona. Uses a **hybrid analysis architecture**:
- **Split-brain**: Real-time psychological extraction embedded in each chat response (JSON with visible reply + hidden `psych_update`)
- **Aegean Consensus**: Periodic dual-model consensus (ported from `shrink`) for deep psychological profiling

## Architecture

```
src/
├── cli/           # Entry points: chat.py, import_resources.py, analyze.py
├── core/          # Emperor brain, Aegean consensus, hierarchical analyzer
├── memory/        # Episodic (token-window), Semantic (insights), Retrieval
├── infrastructure/# SQLite, ChromaDB, ingestion pipeline
├── models/        # Pydantic schemas
└── utils/         # LLM client, config
```

## Storage

- **SQLite**: Relational data (users, sessions, messages, profiles, semantic_insights)
- **ChromaDB**: Vector collections (episodic, semantic, stoic_wisdom, psychoanalysis)
- Migration path to PostgreSQL + pgvector is planned but deferred.

## Key Patterns

### Split-Brain Output
Every Emperor response is JSON with two layers:
```python
class EmperorResponse(BaseModel):
    response_text: str      # Shown to user
    psych_update: PsychUpdate  # Hidden analysis
```

### Token-Based Context Window
Episodic memory uses `max_context_tokens` (not fixed turn count). See `config/settings.yaml`.

### Semantic Processing Tracking
Messages have `semantic_processed_at` column. Query unprocessed: `WHERE semantic_processed_at IS NULL`.

### Dual Concept Tagging
Stoic texts are tagged with both classical ("Amor Fati") and modern ("acceptance") terms for retrieval.

## Running

```bash
source venv/bin/activate
python -m src.cli.chat          # Start conversation
python -m src.cli.analyze       # Trigger deep analysis
python -m src.cli.import_resources <path>  # Import journaling
```

## Environment

Copy `.env.example` to `.env`. Required: `OPENAI_API_KEY`. Optional: `ANTHROPIC_API_KEY` for consensus reviewer.

## Code Style

- Type hints on all function signatures
- Minimal comments; prefer readable code
- Pydantic for data validation
- No beginner explanations in code or docs
