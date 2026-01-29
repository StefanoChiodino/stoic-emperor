from datetime import datetime

from src.models.schemas import EmperorResponse, Message, PsychUpdate, SemanticInsight, Session, User


class TestPsychUpdate:
    def test_create_valid(self):
        psych = PsychUpdate(
            detected_patterns=["catastrophizing", "avoidance"],
            emotional_state="anxious",
            stoic_principle_applied="Dichotomy of Control",
            suggested_next_direction="Explore what is within control",
            confidence=0.85,
        )

        assert psych.confidence == 0.85
        assert len(psych.detected_patterns) == 2

    def test_serialization(self):
        psych = PsychUpdate(
            detected_patterns=["test"],
            emotional_state="calm",
            stoic_principle_applied="Virtue",
            suggested_next_direction="Continue",
            confidence=0.9,
        )

        json_str = psych.model_dump_json()
        assert "detected_patterns" in json_str
        assert "0.9" in json_str


class TestEmperorResponse:
    def test_create_valid(self):
        psych = PsychUpdate(
            detected_patterns=[],
            emotional_state="neutral",
            stoic_principle_applied="",
            suggested_next_direction="",
            confidence=0.5,
        )

        response = EmperorResponse(response_text="This is the Emperor's response.", psych_update=psych)

        assert "Emperor" in response.response_text
        assert response.psych_update.confidence == 0.5


class TestMessage:
    def test_create_user_message(self):
        msg = Message(session_id="session_123", role="user", content="Hello, Emperor.")

        assert msg.role == "user"
        assert msg.psych_update is None
        assert msg.semantic_processed_at is None

    def test_create_emperor_message_with_psych(self):
        psych = PsychUpdate(
            detected_patterns=["seeking_guidance"],
            emotional_state="curious",
            stoic_principle_applied="Wisdom",
            suggested_next_direction="Provide guidance",
            confidence=0.7,
        )

        msg = Message(session_id="session_123", role="emperor", content="Greetings, citizen.", psych_update=psych)

        assert msg.role == "emperor"
        assert msg.psych_update is not None
        assert msg.psych_update.confidence == 0.7

    def test_auto_generated_id(self):
        msg1 = Message(session_id="s1", role="user", content="Test")
        msg2 = Message(session_id="s1", role="user", content="Test")

        assert msg1.id != msg2.id

    def test_auto_generated_timestamp(self):
        msg = Message(session_id="s1", role="user", content="Test")

        assert msg.created_at is not None
        assert isinstance(msg.created_at, datetime)


class TestSession:
    def test_create_session(self):
        session = Session(user_id="user_123")

        assert session.user_id == "user_123"
        assert session.metadata == {}

    def test_session_with_metadata(self):
        session = Session(user_id="user_123", metadata={"source": "cli", "version": "1.0"})

        assert session.metadata["source"] == "cli"


class TestUser:
    def test_create_user(self):
        user = User(id="custom_id")

        assert user.id == "custom_id"

    def test_auto_generated_id(self):
        user = User()

        assert user.id is not None
        assert len(user.id) > 0


class TestSemanticInsight:
    def test_create_insight(self):
        insight = SemanticInsight(
            user_id="user_123",
            source_message_id="msg_456",
            assertion="User struggles with authority figures",
            confidence=0.85,
        )

        assert insight.confidence == 0.85
        assert "authority" in insight.assertion
