from unittest.mock import MagicMock

from src.memory.semantic import SemanticMemory
from src.models.schemas import Message, PsychUpdate


class TestSemanticMemory:
    def test_initialization(self):
        mock_db = MagicMock()
        mock_vectors = MagicMock()
        mock_brain = MagicMock()

        memory = SemanticMemory(mock_db, mock_vectors, mock_brain)

        assert memory.db == mock_db
        assert memory.vectors == mock_vectors
        assert memory.brain == mock_brain

    def test_process_unprocessed_messages_none(self):
        mock_db = MagicMock()
        mock_db.get_unprocessed_messages.return_value = []
        mock_vectors = MagicMock()
        mock_brain = MagicMock()

        memory = SemanticMemory(mock_db, mock_vectors, mock_brain)
        count = memory.process_unprocessed_messages("user123")

        assert count == 0
        mock_db.get_unprocessed_messages.assert_called_once_with("user123")

    def test_process_unprocessed_messages_with_psych_update(self):
        mock_db = MagicMock()
        mock_vectors = MagicMock()
        mock_brain = MagicMock()

        psych_update = PsychUpdate(
            detected_patterns=["pattern1"],
            emotional_state="calm",
            stoic_principle_applied="acceptance",
            suggested_next_direction="continue",
            confidence=0.8,
        )

        emperor_msg = Message(
            id="msg2",
            session_id="session1",
            role="emperor",
            content="Response text",
            psych_update=psych_update,
        )

        user_msg = Message(id="msg1", session_id="session1", role="user", content="User question")

        mock_db.get_unprocessed_messages.return_value = [emperor_msg]
        mock_db.get_session_messages.return_value = [user_msg, emperor_msg]
        mock_brain.extract_semantic_insights.return_value = [{"text": "insight", "confidence": 0.9}]

        memory = SemanticMemory(mock_db, mock_vectors, mock_brain)
        count = memory.process_unprocessed_messages("user123")

        assert count == 1
        mock_db.save_semantic_insight.assert_called_once()
        mock_vectors.add.assert_called_once()

    def test_process_skips_low_confidence_insights(self):
        mock_db = MagicMock()
        mock_vectors = MagicMock()
        mock_brain = MagicMock()

        psych_update = PsychUpdate(
            detected_patterns=[],
            emotional_state="neutral",
            stoic_principle_applied="none",
            suggested_next_direction="none",
            confidence=0.3,
        )

        emperor_msg = Message(
            id="msg2", session_id="session1", role="emperor", content="Response", psych_update=psych_update
        )
        user_msg = Message(id="msg1", session_id="session1", role="user", content="Question")

        mock_db.get_unprocessed_messages.return_value = [emperor_msg]
        mock_db.get_session_messages.return_value = [user_msg, emperor_msg]
        mock_brain.extract_semantic_insights.return_value = [{"text": "weak insight", "confidence": 0.3}]

        memory = SemanticMemory(mock_db, mock_vectors, mock_brain)
        memory.process_unprocessed_messages("user123")

        mock_db.save_semantic_insight.assert_not_called()
        mock_vectors.add.assert_not_called()

    def test_find_preceding_user_message(self):
        mock_db = MagicMock()
        mock_vectors = MagicMock()
        mock_brain = MagicMock()

        user_msg = Message(id="msg1", session_id="session1", role="user", content="User question")
        emperor_msg = Message(id="msg2", session_id="session1", role="emperor", content="Response")

        mock_db.get_session_messages.return_value = [user_msg, emperor_msg]

        memory = SemanticMemory(mock_db, mock_vectors, mock_brain)
        result = memory._find_preceding_user_message(emperor_msg)

        assert result == "User question"

    def test_find_preceding_user_message_none_found(self):
        mock_db = MagicMock()
        mock_vectors = MagicMock()
        mock_brain = MagicMock()

        emperor_msg = Message(id="msg1", session_id="session1", role="emperor", content="Response")
        mock_db.get_session_messages.return_value = [emperor_msg]

        memory = SemanticMemory(mock_db, mock_vectors, mock_brain)
        result = memory._find_preceding_user_message(emperor_msg)

        assert result == ""

    def test_get_relevant_insights_empty(self):
        mock_db = MagicMock()
        mock_vectors = MagicMock()
        mock_brain = MagicMock()

        mock_vectors.query.return_value = {"documents": [[]], "metadatas": [[]]}

        memory = SemanticMemory(mock_db, mock_vectors, mock_brain)
        results = memory.get_relevant_insights("user123", "test query")

        assert results == []

    def test_get_relevant_insights_filters_by_confidence(self):
        mock_db = MagicMock()
        mock_vectors = MagicMock()
        mock_brain = MagicMock()

        mock_vectors.query.return_value = {
            "documents": [["high confidence insight", "low confidence insight"]],
            "metadatas": [[{"confidence": 0.9}, {"confidence": 0.3}]],
        }

        memory = SemanticMemory(mock_db, mock_vectors, mock_brain)
        results = memory.get_relevant_insights("user123", "test query", min_confidence=0.5)

        assert len(results) == 1
        assert results[0] == "high confidence insight"

    def test_get_relevant_insights_respects_n_results(self):
        mock_db = MagicMock()
        mock_vectors = MagicMock()
        mock_brain = MagicMock()

        mock_vectors.query.return_value = {
            "documents": [["insight1", "insight2", "insight3"]],
            "metadatas": [[{"confidence": 0.9}, {"confidence": 0.8}, {"confidence": 0.7}]],
        }

        memory = SemanticMemory(mock_db, mock_vectors, mock_brain)
        results = memory.get_relevant_insights("user123", "query", n_results=2)

        assert len(results) == 2

    def test_get_all_insights(self):
        mock_db = MagicMock()
        mock_vectors = MagicMock()
        mock_brain = MagicMock()

        mock_db.get_user_insights.return_value = ["insight1", "insight2"]

        memory = SemanticMemory(mock_db, mock_vectors, mock_brain)
        results = memory.get_all_insights("user123")

        assert results == ["insight1", "insight2"]
        mock_db.get_user_insights.assert_called_once_with("user123")
