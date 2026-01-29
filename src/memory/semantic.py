from src.core.emperor_brain import EmperorBrain
from src.infrastructure.database import Database
from src.infrastructure.vector_store import VectorStore
from src.models.schemas import Message, SemanticInsight


class SemanticMemory:
    def __init__(self, db: Database, vectors: VectorStore, brain: EmperorBrain):
        self.db = db
        self.vectors = vectors
        self.brain = brain

    def process_unprocessed_messages(self, user_id: str) -> int:
        unprocessed = self.db.get_unprocessed_messages(user_id)
        processed_count = 0

        for msg in unprocessed:
            if msg.psych_update and msg.role == "emperor":
                user_msg = self._find_preceding_user_message(msg)
                if user_msg:
                    self._extract_and_store_insights(user_id, user_msg, msg)
                    processed_count += 1

            self.db.mark_message_processed(msg.id)

        return processed_count

    def _find_preceding_user_message(self, emperor_msg: Message) -> str:
        messages = self.db.get_session_messages(emperor_msg.session_id)
        for i, msg in enumerate(messages):
            if msg.id == emperor_msg.id and i > 0:
                prev = messages[i - 1]
                if prev.role == "user":
                    return prev.content
        return ""

    def _extract_and_store_insights(self, user_id: str, user_message: str, emperor_msg: Message) -> None:
        if not emperor_msg.psych_update:
            return

        assertions = self.brain.extract_semantic_insights(
            user_message=user_message, psych_update=emperor_msg.psych_update
        )

        for assertion in assertions:
            text = assertion.get("text", "")
            confidence = assertion.get("confidence", 0.5)

            if not text or confidence < 0.5:
                continue

            insight = SemanticInsight(
                user_id=user_id, source_message_id=emperor_msg.id, assertion=text, confidence=confidence
            )

            self.db.save_semantic_insight(insight)

            self.vectors.add(
                collection="semantic",
                ids=[insight.id],
                documents=[text],
                metadatas=[{"user_id": user_id, "source_message_id": emperor_msg.id, "confidence": confidence}],
            )

    def get_relevant_insights(
        self, user_id: str, query: str, n_results: int = 5, min_confidence: float = 0.5
    ) -> list[str]:
        results = self.vectors.query(
            collection="semantic", query_texts=[query], n_results=n_results * 2, where={"user_id": user_id}
        )

        if not results.get("documents") or not results["documents"][0]:
            return []

        filtered = []
        metadatas = results.get("metadatas", [[]])[0]
        documents = results["documents"][0]

        for doc, meta in zip(documents, metadatas, strict=False):
            if meta.get("confidence", 0) >= min_confidence:
                filtered.append(doc)
                if len(filtered) >= n_results:
                    break

        return filtered

    def get_all_insights(self, user_id: str) -> list[SemanticInsight]:
        return self.db.get_user_insights(user_id)
