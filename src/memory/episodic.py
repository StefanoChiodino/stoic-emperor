import tiktoken

from src.infrastructure.database import Database
from src.infrastructure.vector_store import VectorStore
from src.models.schemas import Message


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


class EpisodicMemory:
    def __init__(self, db: Database, vectors: VectorStore, max_context_tokens: int = 4000):
        self.db = db
        self.vectors = vectors
        self.max_context_tokens = max_context_tokens

    def get_recent_context(self, session_id: str, current_message: str | None = None) -> list[Message]:
        messages = self.db.get_session_messages(session_id)

        if not messages:
            return []

        selected: list[Message] = []
        total_tokens = 0

        if current_message:
            total_tokens += estimate_tokens(current_message)

        for msg in reversed(messages):
            msg_tokens = estimate_tokens(msg.content)
            if total_tokens + msg_tokens > self.max_context_tokens:
                break
            selected.insert(0, msg)
            total_tokens += msg_tokens

        return selected

    def store_turn(self, user_id: str, session_id: str, user_message: str, emperor_response: str) -> None:
        turn_text = f"User: {user_message}\nMarcus: {emperor_response}"
        turn_id = f"{session_id}_{len(self.db.get_session_messages(session_id))}"

        self.vectors.add(
            collection="episodic",
            ids=[turn_id],
            documents=[turn_text],
            metadatas=[{"user_id": user_id, "session_id": session_id, "type": "conversation_turn"}],
        )

    def search_past_conversations(self, user_id: str, query: str, n_results: int = 5) -> list[str]:
        results = self.vectors.query(
            collection="episodic", query_texts=[query], n_results=n_results, where={"user_id": user_id}
        )

        if results.get("documents") and results["documents"][0]:
            return results["documents"][0]
        return []
