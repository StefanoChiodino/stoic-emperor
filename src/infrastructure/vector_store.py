# PRIVACY: Must import before chromadb
from src.utils.privacy import disable_telemetry
disable_telemetry()

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class VectorStore:
    COLLECTIONS = ["episodic", "semantic", "stoic_wisdom", "psychoanalysis"]

    def __init__(self, db_path: str = "./data/vector_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        for name in self.COLLECTIONS:
            self.client.get_or_create_collection(name=name)

    def add(
        self,
        collection: str,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        coll = self.client.get_collection(collection)
        kwargs: Dict[str, Any] = {"ids": ids, "documents": documents}
        if metadatas:
            kwargs["metadatas"] = metadatas
        if embeddings:
            kwargs["embeddings"] = embeddings
        coll.add(**kwargs)

    def query(
        self,
        collection: str,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        coll = self.client.get_collection(collection)
        kwargs: Dict[str, Any] = {"n_results": n_results}
        if query_texts:
            kwargs["query_texts"] = query_texts
        if query_embeddings:
            kwargs["query_embeddings"] = query_embeddings
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document
        return coll.query(**kwargs)

    def get(
        self,
        collection: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        coll = self.client.get_collection(collection)
        kwargs: Dict[str, Any] = {}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where
        if limit:
            kwargs["limit"] = limit
        return coll.get(**kwargs)

    def delete(
        self,
        collection: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> None:
        coll = self.client.get_collection(collection)
        kwargs: Dict[str, Any] = {}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where
        coll.delete(**kwargs)

    def count(self, collection: str) -> int:
        coll = self.client.get_collection(collection)
        return coll.count()
