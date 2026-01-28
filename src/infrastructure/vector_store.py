import os
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from urllib.parse import urlparse
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


class VectorStore:
    COLLECTIONS = ["episodic", "semantic", "stoic_wisdom", "psychoanalysis"]

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", "sqlite:///./data/stoic_emperor.db")

        parsed = urlparse(self.database_url)
        self.is_postgres = parsed.scheme in ("postgresql", "postgres")

        if self.is_postgres:
            import psycopg2
            from psycopg2 import pool as pg_pool
            from pgvector.psycopg2 import register_vector
            self._psycopg2 = psycopg2
            self._register_vector = register_vector
            self._pool = pg_pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                dsn=self.database_url
            )
        else:
            db_path = self.database_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._sqlite_path = db_path

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._ensure_extension()
        self._ensure_collections()

    @contextmanager
    def _connection(self):
        if self.is_postgres:
            conn = self._pool.getconn()
            try:
                self._register_vector(conn)
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._pool.putconn(conn)
        else:
            conn = sqlite3.connect(self._sqlite_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _ensure_extension(self) -> None:
        if self.is_postgres:
            conn = self._pool.getconn()
            try:
                cursor = conn.cursor()
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.close()
                conn.commit()
            finally:
                self._pool.putconn(conn)

    def _ensure_collections(self) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            for name in self.COLLECTIONS:
                if self.is_postgres:
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS vector_{name} (
                            id TEXT PRIMARY KEY,
                            document TEXT NOT NULL,
                            embedding vector(384),
                            metadata JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                else:
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS vector_{name} (
                            id TEXT PRIMARY KEY,
                            document TEXT NOT NULL,
                            embedding TEXT,
                            metadata TEXT,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
            cursor.close()

    def add(
        self,
        collection: str,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        if embeddings is None:
            embeddings = self.embedding_model.encode(documents).tolist()

        with self._connection() as conn:
            cursor = conn.cursor()
            for i, doc_id in enumerate(ids):
                metadata = metadatas[i] if metadatas else {}
                if self.is_postgres:
                    cursor.execute(
                        f"""INSERT INTO vector_{collection} (id, document, embedding, metadata)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (id) DO UPDATE
                           SET document = EXCLUDED.document,
                               embedding = EXCLUDED.embedding,
                               metadata = EXCLUDED.metadata""",
                        (doc_id, documents[i], embeddings[i], json.dumps(metadata))
                    )
                else:
                    cursor.execute(
                        f"""INSERT OR REPLACE INTO vector_{collection} (id, document, embedding, metadata)
                           VALUES (?, ?, ?, ?)""",
                        (doc_id, documents[i], json.dumps(embeddings[i]), json.dumps(metadata))
                    )
            cursor.close()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

    def query(
        self,
        collection: str,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if query_embeddings is None and query_texts:
            query_embeddings = self.embedding_model.encode(query_texts).tolist()

        if not query_embeddings:
            raise ValueError("Either query_texts or query_embeddings must be provided")

        results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        with self._connection() as conn:
            if self.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                for query_embedding in query_embeddings:
                    where_clause = ""
                    params: List[Any] = [query_embedding, n_results]

                    if where:
                        conditions = []
                        for key, value in where.items():
                            conditions.append(f"metadata->>'{key}' = %s")
                            params.insert(-1, str(value))
                        where_clause = "WHERE " + " AND ".join(conditions)

                    cursor.execute(
                        f"""SELECT id, document, metadata,
                               1 - (embedding <=> %s::vector) as similarity
                           FROM vector_{collection}
                           {where_clause}
                           ORDER BY embedding <=> %s::vector
                           LIMIT %s""",
                        params
                    )

                    rows = cursor.fetchall()
                    results["ids"][0].extend([row["id"] for row in rows])
                    results["documents"][0].extend([row["document"] for row in rows])
                    results["metadatas"][0].extend([row["metadata"] if row["metadata"] else {} for row in rows])
                    results["distances"][0].extend([1 - row["similarity"] for row in rows])
            else:
                cursor = conn.cursor()
                where_clause = ""
                params: List[Any] = []

                if where:
                    conditions = []
                    for key, value in where.items():
                        conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                        params.append(str(value))
                    where_clause = "WHERE " + " AND ".join(conditions)

                cursor.execute(
                    f"SELECT id, document, embedding, metadata FROM vector_{collection} {where_clause}",
                    params
                )
                rows = cursor.fetchall()

                scored_rows = []
                for row in rows:
                    embedding = json.loads(row["embedding"]) if row["embedding"] else None
                    if embedding:
                        for query_embedding in query_embeddings:
                            similarity = self._cosine_similarity(query_embedding, embedding)
                            scored_rows.append((row, similarity))

                scored_rows.sort(key=lambda x: x[1], reverse=True)
                scored_rows = scored_rows[:n_results]

                for row, similarity in scored_rows:
                    results["ids"][0].append(row["id"])
                    results["documents"][0].append(row["document"])
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                    results["metadatas"][0].append(metadata)
                    results["distances"][0].append(1 - similarity)

            cursor.close()

        return results

    def get(
        self,
        collection: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        results = {"ids": [], "documents": [], "metadatas": []}

        with self._connection() as conn:
            if self.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                where_clause = ""
                params: List[Any] = []

                if ids:
                    placeholders = ",".join(["%s"] * len(ids))
                    where_clause = f"WHERE id IN ({placeholders})"
                    params.extend(ids)
                elif where:
                    conditions = []
                    for key, value in where.items():
                        conditions.append(f"metadata->>'{key}' = %s")
                        params.append(str(value))
                    where_clause = "WHERE " + " AND ".join(conditions)

                limit_clause = f"LIMIT {limit}" if limit else ""

                cursor.execute(
                    f"SELECT id, document, metadata FROM vector_{collection} {where_clause} {limit_clause}",
                    params
                )

                rows = cursor.fetchall()
                results["ids"] = [row["id"] for row in rows]
                results["documents"] = [row["document"] for row in rows]
                results["metadatas"] = [row["metadata"] if row["metadata"] else {} for row in rows]
            else:
                cursor = conn.cursor()
                where_clause = ""
                params: List[Any] = []

                if ids:
                    placeholders = ",".join(["?"] * len(ids))
                    where_clause = f"WHERE id IN ({placeholders})"
                    params.extend(ids)
                elif where:
                    conditions = []
                    for key, value in where.items():
                        conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                        params.append(str(value))
                    where_clause = "WHERE " + " AND ".join(conditions)

                limit_clause = f"LIMIT {limit}" if limit else ""

                cursor.execute(
                    f"SELECT id, document, metadata FROM vector_{collection} {where_clause} {limit_clause}",
                    params
                )

                rows = cursor.fetchall()
                results["ids"] = [row["id"] for row in rows]
                results["documents"] = [row["document"] for row in rows]
                results["metadatas"] = [json.loads(row["metadata"]) if row["metadata"] else {} for row in rows]

            cursor.close()

        return results

    def delete(
        self,
        collection: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            where_clause = ""
            params: List[Any] = []

            if self.is_postgres:
                if ids:
                    placeholders = ",".join(["%s"] * len(ids))
                    where_clause = f"WHERE id IN ({placeholders})"
                    params.extend(ids)
                elif where:
                    conditions = []
                    for key, value in where.items():
                        conditions.append(f"metadata->>'{key}' = %s")
                        params.append(str(value))
                    where_clause = "WHERE " + " AND ".join(conditions)
            else:
                if ids:
                    placeholders = ",".join(["?"] * len(ids))
                    where_clause = f"WHERE id IN ({placeholders})"
                    params.extend(ids)
                elif where:
                    conditions = []
                    for key, value in where.items():
                        conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                        params.append(str(value))
                    where_clause = "WHERE " + " AND ".join(conditions)

            cursor.execute(f"DELETE FROM vector_{collection} {where_clause}", params)
            cursor.close()

    def count(self, collection: str) -> int:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) as count FROM vector_{collection}")
            result = cursor.fetchone()
            cursor.close()
            if self.is_postgres:
                return result["count"] if result else 0
            return result[0] if result else 0
