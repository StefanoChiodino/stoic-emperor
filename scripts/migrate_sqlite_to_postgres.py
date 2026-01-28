#!/usr/bin/env python3
"""
Migration script: SQLite + ChromaDB → PostgreSQL + pgvector

Migrates relational data from SQLite to PostgreSQL.
Vector data must be re-imported after migration.
"""

import sqlite3
import argparse
import sys
from datetime import datetime

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    print("Error: psycopg not installed. Run: pip install psycopg[binary]")
    sys.exit(1)


def migrate(sqlite_path: str, postgres_url: str, dry_run: bool = False):
    print(f"{'[DRY RUN] ' if dry_run else ''}Migrating from SQLite to PostgreSQL...")
    print(f"  Source: {sqlite_path}")
    print(f"  Target: {postgres_url.split('@')[1] if '@' in postgres_url else postgres_url}")
    print()

    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row

    if dry_run:
        print("[DRY RUN] Skipping actual migration. Use --execute to perform migration.")
        return analyze_migration(sqlite_conn)

    pg_conn = psycopg.connect(postgres_url, row_factory=dict_row)

    try:
        with pg_conn.cursor() as cur:
            print("📊 Migrating users...")
            sqlite_users = sqlite_conn.execute("SELECT * FROM users").fetchall()
            user_count = 0
            for user in sqlite_users:
                cur.execute(
                    "INSERT INTO users (id, created_at) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (user["id"], user["created_at"])
                )
                user_count += 1
            print(f"  ✓ Migrated {user_count} users")

            print("📝 Migrating sessions...")
            sqlite_sessions = sqlite_conn.execute("SELECT * FROM sessions").fetchall()
            session_count = 0
            for session in sqlite_sessions:
                metadata = session["metadata"] if session["metadata"] else '{}'
                cur.execute(
                    "INSERT INTO sessions (id, user_id, created_at, metadata) VALUES (%s, %s, %s, %s::jsonb) ON CONFLICT (id) DO NOTHING",
                    (session["id"], session["user_id"], session["created_at"], metadata)
                )
                session_count += 1
            print(f"  ✓ Migrated {session_count} sessions")

            print("💬 Migrating messages...")
            sqlite_messages = sqlite_conn.execute("SELECT * FROM messages").fetchall()
            message_count = 0
            for msg in sqlite_messages:
                psych_update = msg["psych_update"] if msg["psych_update"] else None
                cur.execute(
                    """INSERT INTO messages (id, session_id, role, content, psych_update, created_at, semantic_processed_at)
                       VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s) ON CONFLICT (id) DO NOTHING""",
                    (msg["id"], msg["session_id"], msg["role"], msg["content"],
                     psych_update, msg["created_at"], msg["semantic_processed_at"])
                )
                message_count += 1
            print(f"  ✓ Migrated {message_count} messages")

            print("🧠 Migrating semantic insights...")
            sqlite_insights = sqlite_conn.execute("SELECT * FROM semantic_insights").fetchall()
            insight_count = 0
            for insight in sqlite_insights:
                cur.execute(
                    """INSERT INTO semantic_insights (id, user_id, source_message_id, assertion, confidence, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING""",
                    (insight["id"], insight["user_id"], insight["source_message_id"],
                     insight["assertion"], insight["confidence"], insight["created_at"])
                )
                insight_count += 1
            print(f"  ✓ Migrated {insight_count} insights")

            try:
                print("📚 Migrating profiles...")
                sqlite_profiles = sqlite_conn.execute("SELECT * FROM profiles").fetchall()
                profile_count = 0
                for profile in sqlite_profiles:
                    consensus_log = profile["consensus_log"] if profile.get("consensus_log") else None
                    cur.execute(
                        """INSERT INTO profiles (id, user_id, version, content, consensus_log, created_at)
                           VALUES (%s, %s, %s, %s, %s::jsonb, %s) ON CONFLICT (id) DO NOTHING""",
                        (profile["id"], profile["user_id"], profile["version"],
                         profile["content"], consensus_log, profile["created_at"])
                    )
                    profile_count += 1
                print(f"  ✓ Migrated {profile_count} profiles")
            except Exception as e:
                print(f"  ⚠ Profiles table not found or error: {e}")

            try:
                print("📦 Migrating condensed summaries...")
                sqlite_summaries = sqlite_conn.execute("SELECT * FROM condensed_summaries").fetchall()
                summary_count = 0
                for summary in sqlite_summaries:
                    source_ids = summary["source_summary_ids"] if summary.get("source_summary_ids") else '[]'
                    consensus = summary["consensus_log"] if summary.get("consensus_log") else None
                    cur.execute(
                        """INSERT INTO condensed_summaries
                           (id, user_id, level, content, period_start, period_end,
                            source_message_count, source_word_count, source_summary_ids,
                            consensus_log, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                           ON CONFLICT (id) DO NOTHING""",
                        (summary["id"], summary["user_id"], summary["level"], summary["content"],
                         summary["period_start"], summary["period_end"],
                         summary["source_message_count"], summary["source_word_count"],
                         source_ids, consensus, summary["created_at"])
                    )
                    summary_count += 1
                print(f"  ✓ Migrated {summary_count} summaries")
            except Exception as e:
                print(f"  ⚠ Condensed summaries table not found or error: {e}")

            pg_conn.commit()

        print("\n✅ Migration complete!")
        print("\n⚠️  IMPORTANT: Vector data NOT migrated automatically.")
        print("   Re-import your stoic texts using:")
        print("   python -m src.cli.import_resources stoic ./data/stoic_texts")

    except Exception as e:
        pg_conn.rollback()
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
    finally:
        sqlite_conn.close()
        pg_conn.close()


def analyze_migration(sqlite_conn):
    """Analyze what would be migrated without actually migrating."""
    print("Analyzing SQLite database...\n")

    tables = [
        ("users", "SELECT COUNT(*) as count FROM users"),
        ("sessions", "SELECT COUNT(*) as count FROM sessions"),
        ("messages", "SELECT COUNT(*) as count FROM messages"),
        ("semantic_insights", "SELECT COUNT(*) as count FROM semantic_insights"),
    ]

    optional_tables = [
        ("profiles", "SELECT COUNT(*) as count FROM profiles"),
        ("condensed_summaries", "SELECT COUNT(*) as count FROM condensed_summaries"),
    ]

    print("📊 Required Tables:")
    for table_name, query in tables:
        try:
            result = sqlite_conn.execute(query).fetchone()
            count = result[0]
            print(f"  {table_name}: {count} rows")
        except Exception as e:
            print(f"  {table_name}: ERROR - {e}")

    print("\n📦 Optional Tables:")
    for table_name, query in optional_tables:
        try:
            result = sqlite_conn.execute(query).fetchone()
            count = result[0]
            print(f"  {table_name}: {count} rows")
        except Exception:
            print(f"  {table_name}: Not found (OK)")

    print("\n⚠️  Vector data in ChromaDB will NOT be migrated.")
    print("   You will need to re-import stoic texts after migration.")

    sqlite_conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Stoic Emperor from SQLite to PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (analyze only)
  python scripts/migrate_sqlite_to_postgres.py \\
    --sqlite ./data/stoic_emperor.db \\
    --postgres "postgresql://localhost/stoic_emperor"

  # Execute migration
  python scripts/migrate_sqlite_to_postgres.py \\
    --sqlite ./data/stoic_emperor.db \\
    --postgres "postgresql://localhost/stoic_emperor" \\
    --execute

Note: Vector data must be re-imported after migration.
        """
    )
    parser.add_argument(
        "--sqlite",
        required=True,
        help="Path to SQLite database file"
    )
    parser.add_argument(
        "--postgres",
        required=True,
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the migration (default is dry-run)"
    )

    args = parser.parse_args()

    migrate(args.sqlite, args.postgres, dry_run=not args.execute)


if __name__ == "__main__":
    main()
