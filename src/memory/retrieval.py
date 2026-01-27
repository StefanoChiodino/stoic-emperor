from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from src.models.schemas import Message
from src.infrastructure.database import Database
from src.infrastructure.vector_store import VectorStore
from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.core.emperor_brain import EmperorBrain


@dataclass
class RetrievalContext:
    recent_messages: List[Message]
    episodic_matches: List[str]
    semantic_insights: List[str]
    stoic_wisdom: List[str]
    psychoanalysis: List[str]
    expanded_query: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episodic": self.episodic_matches,
            "insights": self.semantic_insights,
            "stoic": self.stoic_wisdom,
            "psych": self.psychoanalysis,
        }


class UnifiedRetriever:
    def __init__(
        self,
        db: Database,
        vectors: VectorStore,
        brain: EmperorBrain,
        max_context_tokens: int = 4000
    ):
        self.db = db
        self.vectors = vectors
        self.brain = brain
        self.episodic = EpisodicMemory(db, vectors, max_context_tokens)
        self.semantic = SemanticMemory(db, vectors, brain)

    def retrieve(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        n_results: int = 5
    ) -> RetrievalContext:
        expanded_query = self._expand_query(user_message)

        recent_messages = self.episodic.get_recent_context(
            session_id=session_id,
            current_message=user_message
        )

        episodic_matches = self.episodic.search_past_conversations(
            user_id=user_id,
            query=expanded_query,
            n_results=n_results
        )

        semantic_insights = self.semantic.get_relevant_insights(
            user_id=user_id,
            query=expanded_query,
            n_results=n_results
        )

        stoic_wisdom = self._query_collection("stoic_wisdom", expanded_query, n_results)
        psychoanalysis = self._query_collection("psychoanalysis", expanded_query, n_results)

        return RetrievalContext(
            recent_messages=recent_messages,
            episodic_matches=episodic_matches,
            semantic_insights=semantic_insights,
            stoic_wisdom=stoic_wisdom,
            psychoanalysis=psychoanalysis,
            expanded_query=expanded_query
        )

    def _expand_query(self, user_message: str) -> str:
        try:
            expanded = self.brain.expand_query(user_message)
            terms = [t.strip() for t in expanded.split(",")]
            return " ".join(terms) if terms else user_message
        except Exception:
            return user_message

    def _query_collection(
        self,
        collection: str,
        query: str,
        n_results: int
    ) -> List[str]:
        try:
            results = self.vectors.query(
                collection=collection,
                query_texts=[query],
                n_results=n_results
            )
            if results.get("documents") and results["documents"][0]:
                return results["documents"][0]
        except Exception:
            pass
        return []

    def process_new_insights(self, user_id: str) -> int:
        return self.semantic.process_unprocessed_messages(user_id)
