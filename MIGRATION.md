# Migration Guide: SQLite/ChromaDB to PostgreSQL/pgvector

This guide helps you migrate from the legacy SQLite + ChromaDB setup to the new PostgreSQL + pgvector architecture.

## Why Migrate?

- **Multi-tenancy**: Support multiple users with proper authentication
- **Scalability**: PostgreSQL handles concurrent users better than SQLite
- **Unified storage**: One database for both relational data and vectors
- **Production-ready**: Compatible with Supabase, Railway, Fly.io, and other cloud providers
- **No file-based limitations**: No ChromaDB file locking issues

## Migration Steps

### 1. Backup Your Current Data

```bash
# Backup SQLite database
cp ./data/stoic_emperor.db ./data/stoic_emperor.db.backup

# Backup ChromaDB
cp -r ./data/vector_db ./data/vector_db.backup
```

### 2. Set Up PostgreSQL

**Option A: Local PostgreSQL with Docker**

```bash
# Start PostgreSQL with pgvector
docker run -d \
  --name stoic-postgres \
  -e POSTGRES_DB=stoic_emperor \
  -e POSTGRES_USER=stoic \
  -e POSTGRES_PASSWORD=changeme \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Enable pgvector extension
docker exec -it stoic-postgres psql -U stoic -d stoic_emperor -c "CREATE EXTENSION vector;"
```

**Option B: Supabase (recommended for production)**

1. Create project at https://supabase.com
2. Get connection string from Settings → Database
3. pgvector is already installed

**Option C: Use docker-compose**

```bash
docker-compose up -d postgres
docker-compose exec postgres psql -U stoic -d stoic_emperor -c "CREATE EXTENSION vector;"
```

### 3. Update Environment Variables

Edit `.env`:

```bash
# Old (remove these)
# VECTOR_DB_PATH=./data/vector_db
# SQLITE_DB_PATH=./data/stoic_emperor.db

# New
DATABASE_URL=postgresql://stoic:changeme@localhost:5432/stoic_emperor
ENVIRONMENT=development
```

### 4. Install New Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Test Database Connection

```python
python -c "
from src.infrastructure.database import Database
db = Database()
print('Database connection successful!')
"
```

### 6. Migrate Data (Optional)

If you want to preserve existing data, use the migration script:

```bash
python scripts/migrate_sqlite_to_postgres.py \
  --sqlite ./data/stoic_emperor.db \
  --postgres "$DATABASE_URL"
```

**Note**: This script migrates:
- Users
- Sessions
- Messages (including psych_updates)
- Semantic insights
- Condensed summaries

Vector embeddings are NOT migrated automatically. You'll need to re-import your stoic texts.

### 7. Re-import Vector Data

```bash
# Import stoic texts (re-embeds everything)
python -m src.cli.import_resources ./data/stoic_texts
```

### 8. Verify Migration

```bash
# Start the app
python -m src.cli.chat

# Or start the web server
uvicorn src.web.api:app --reload
```

## Breaking Changes

### Database Initialization

**Old:**
```python
from src.infrastructure.database import Database
db = Database("./data/stoic_emperor.db")
```

**New:**
```python
from src.infrastructure.database import Database
db = Database()  # Reads DATABASE_URL from env
# Or explicitly:
db = Database("postgresql://localhost/stoic_emperor")
```

### Vector Store Initialization

**Old:**
```python
from src.infrastructure.vector_store import VectorStore
vectors = VectorStore("./data/vector_db")
```

**New:**
```python
from src.infrastructure.vector_store import VectorStore
vectors = VectorStore()  # Uses same DATABASE_URL
# Or explicitly:
vectors = VectorStore("postgresql://localhost/stoic_emperor")
```

### Configuration

**Old:**
```yaml
# config/settings.yaml
paths:
  sqlite_db: "./data/stoic_emperor.db"
  vector_db: "./data/vector_db"
```

**New:**
```yaml
# config/settings.yaml
database:
  url: "${DATABASE_URL}"
```

## Data Migration Script

Create `scripts/migrate_sqlite_to_postgres.py`:

```python
#!/usr/bin/env python3
import sqlite3
import argparse
from datetime import datetime
import psycopg

def migrate(sqlite_path: str, postgres_url: str):
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row

    # Connect to PostgreSQL
    pg_conn = psycopg.connect(postgres_url)

    with pg_conn.cursor() as cur:
        # Migrate users
        print("Migrating users...")
        sqlite_users = sqlite_conn.execute("SELECT * FROM users").fetchall()
        for user in sqlite_users:
            cur.execute(
                "INSERT INTO users (id, created_at) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (user["id"], user["created_at"])
            )

        # Migrate sessions
        print("Migrating sessions...")
        sqlite_sessions = sqlite_conn.execute("SELECT * FROM sessions").fetchall()
        for session in sqlite_sessions:
            cur.execute(
                "INSERT INTO sessions (id, user_id, created_at, metadata) VALUES (%s, %s, %s, %s::jsonb) ON CONFLICT DO NOTHING",
                (session["id"], session["user_id"], session["created_at"], session["metadata"] or '{}')
            )

        # Migrate messages
        print("Migrating messages...")
        sqlite_messages = sqlite_conn.execute("SELECT * FROM messages").fetchall()
        for msg in sqlite_messages:
            cur.execute(
                """INSERT INTO messages (id, session_id, role, content, psych_update, created_at, semantic_processed_at)
                   VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s) ON CONFLICT DO NOTHING""",
                (msg["id"], msg["session_id"], msg["role"], msg["content"],
                 msg["psych_update"], msg["created_at"], msg["semantic_processed_at"])
            )

        # Migrate semantic insights
        print("Migrating semantic insights...")
        sqlite_insights = sqlite_conn.execute("SELECT * FROM semantic_insights").fetchall()
        for insight in sqlite_insights:
            cur.execute(
                """INSERT INTO semantic_insights (id, user_id, source_message_id, assertion, confidence, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                (insight["id"], insight["user_id"], insight["source_message_id"],
                 insight["assertion"], insight["confidence"], insight["created_at"])
            )

        pg_conn.commit()

    print("Migration complete!")
    sqlite_conn.close()
    pg_conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True, help="Path to SQLite database")
    parser.add_argument("--postgres", required=True, help="PostgreSQL connection URL")
    args = parser.parse_args()

    migrate(args.sqlite, args.postgres)
```

Make it executable:

```bash
chmod +x scripts/migrate_sqlite_to_postgres.py
```

## Rollback Plan

If you need to rollback:

1. Restore backups:
   ```bash
   cp ./data/stoic_emperor.db.backup ./data/stoic_emperor.db
   cp -r ./data/vector_db.backup ./data/vector_db
   ```

2. Revert code to previous commit:
   ```bash
   git checkout <previous-commit>
   ```

3. Reinstall old dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Troubleshooting

### "psycopg.OperationalError: could not connect"

Check DATABASE_URL format:
```bash
echo $DATABASE_URL
# Should be: postgresql://user:password@host:port/database
```

### "relation does not exist"

Run migrations:
```python
from src.infrastructure.database import Database
Database()  # Auto-runs migrations
```

### "extension vector does not exist"

Enable pgvector:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Performance Issues

Add indexes:
```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_semantic ON messages(semantic_processed_at);
```

## Post-Migration Checklist

- [ ] Database connection works
- [ ] Migrations applied successfully
- [ ] Stoic texts imported
- [ ] CLI chat works
- [ ] Web API works
- [ ] Authentication works (if configured)
- [ ] Vector search returns results
- [ ] Old SQLite/ChromaDB files backed up
- [ ] `.env` updated with DATABASE_URL

## Questions?

See `DEPLOYMENT.md` for production deployment or open an issue on GitHub.
