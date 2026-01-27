# Quick Start Guide

Get Stoic Emperor running locally in 5 minutes.

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- OpenAI API key

## Local Development Setup

### 1. Clone and Setup

```bash
cd stoic-emperor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-key-here
DATABASE_URL=postgresql://stoic:changeme@localhost:5432/stoic_emperor
```

### 3. Start PostgreSQL

```bash
docker-compose up -d postgres
```

Wait for it to be ready, then enable pgvector:

```bash
docker-compose exec postgres psql -U stoic -d stoic_emperor -c "CREATE EXTENSION vector;"
```

### 4. Import Stoic Texts (Optional)

```bash
python -m src.cli.import_resources stoic ./data/stoic_texts/meditations.txt \
  --author "Marcus Aurelius" \
  --work "Meditations"
```

### 5. Start Chatting

**Option A: CLI**

```bash
python -m src.cli.chat
```

**Option B: Web API**

```bash
uvicorn src.web.api:app --reload
```

Then open http://localhost:8000 in your browser.

## Using Docker Compose (Easiest)

Run everything with Docker:

```bash
# Edit .env first with your API keys
docker-compose up

# Access web UI at http://localhost:8000
```

## Common Commands

```bash
# Chat via CLI
python -m src.cli.chat

# Run psychological analysis
python -m src.cli.analyze

# Import journal entries
python -m src.cli.import_resources journal ./my_journal.txt

# Import stoic texts
python -m src.cli.import_resources stoic ./meditations.txt \
  --author "Marcus Aurelius" \
  --work "Meditations"
```

## Deployment

Ready to deploy? See:

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Full deployment guide (Railway, Fly.io)
- **[MIGRATION.md](MIGRATION.md)** - Migrate from SQLite to PostgreSQL
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical details

## Troubleshooting

### "psycopg.OperationalError: could not connect"

PostgreSQL isn't running. Start it:

```bash
docker-compose up -d postgres
```

### "extension vector does not exist"

Enable pgvector:

```bash
docker-compose exec postgres psql -U stoic -d stoic_emperor -c "CREATE EXTENSION vector;"
```

### "OpenAI API key not found"

Set it in `.env`:

```bash
OPENAI_API_KEY=sk-your-key-here
```

## Architecture

- **Database**: PostgreSQL + pgvector
- **Vector Store**: pgvector (same database)
- **Auth**: Supabase Auth (JWT)
- **LLM**: OpenAI (configurable)
- **Deployment**: Docker, Railway, Fly.io

## Development vs Production

| Feature | Development | Production |
|---------|-------------|------------|
| Database | Local PostgreSQL | Supabase/Cloud SQL |
| Auth | Optional (default user) | Required (JWT) |
| HTTPS | Not required | Required |
| Environment | `ENVIRONMENT=development` | `ENVIRONMENT=production` |

## Next Steps

1. ✅ Get it running locally
2. 📖 Read [AGENTS.md](AGENTS.md) for architecture details
3. 🚀 Deploy to production (see DEPLOYMENT.md)
4. 🔐 Set up Supabase Auth
5. 📊 Configure monitoring

## Privacy Note

Stoic Emperor is privacy-first:

- All data stored locally (PostgreSQL)
- No telemetry or analytics
- LLM calls only to OpenAI/Anthropic
- For maximum privacy, use local LLMs (Ollama)

See [AGENTS.md](AGENTS.md) for more on privacy.

## Support

- 📚 Documentation: See README.md and linked guides
- 🐛 Issues: GitHub Issues
- 💬 Questions: GitHub Discussions

---

Happy journaling with Marcus Aurelius! 🏛️
