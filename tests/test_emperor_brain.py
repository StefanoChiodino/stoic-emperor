import json

import pytest

from src.models.schemas import EmperorResponse, PsychUpdate


class TestEmperorResponseParsing:
    def test_parse_valid_response(self, mock_emperor_response):
        psych_data = mock_emperor_response["psych_update"]
        psych = PsychUpdate(**psych_data)
        response = EmperorResponse(response_text=mock_emperor_response["response_text"], psych_update=psych)

        assert "father" in response.response_text.lower()
        assert len(response.psych_update.detected_patterns) > 0

    def test_psych_update_validation(self):
        psych = PsychUpdate(
            detected_patterns=["pattern1", "pattern2"],
            emotional_state="anxious",
            stoic_principle_applied="Amor Fati",
            suggested_next_direction="Explore acceptance",
            confidence=0.75,
        )

        assert psych.confidence == 0.75
        assert "pattern1" in psych.detected_patterns

    def test_psych_update_serialization(self, sample_psych_update):
        json_str = sample_psych_update.model_dump_json()
        data = json.loads(json_str)

        assert "detected_patterns" in data
        assert "confidence" in data

        restored = PsychUpdate(**data)
        assert restored.confidence == sample_psych_update.confidence


class TestEmperorBrainUnit:
    def test_brain_initialization(self):
        from unittest.mock import MagicMock

        from src.core.emperor_brain import EmperorBrain

        mock_llm = MagicMock()
        brain = EmperorBrain(llm_client=mock_llm)

        assert brain.llm == mock_llm

    def test_parse_response_valid_json(self):
        from unittest.mock import MagicMock

        from src.core.emperor_brain import EmperorBrain

        brain = EmperorBrain(llm_client=MagicMock())

        json_response = json.dumps(
            {
                "response_text": "Test response from Marcus",
                "psych_update": {
                    "detected_patterns": ["test_pattern"],
                    "emotional_state": "calm",
                    "stoic_principle_applied": "Virtue",
                    "suggested_next_direction": "Continue",
                    "confidence": 0.9,
                },
            }
        )

        result = brain._parse_response(json_response)

        assert isinstance(result, EmperorResponse)
        assert result.response_text == "Test response from Marcus"
        assert result.psych_update.confidence == 0.9

    def test_parse_response_invalid_json(self):
        from unittest.mock import MagicMock

        from src.core.emperor_brain import EmperorBrain

        brain = EmperorBrain(llm_client=MagicMock())

        with pytest.raises(json.JSONDecodeError):
            brain._parse_response("This is not JSON")

    def test_parse_response_with_semantic_assertions(self):
        from unittest.mock import MagicMock

        from src.core.emperor_brain import EmperorBrain

        brain = EmperorBrain(llm_client=MagicMock())

        json_response = json.dumps(
            {
                "response_text": "Test response",
                "psych_update": {
                    "detected_patterns": [],
                    "emotional_state": "neutral",
                    "stoic_principle_applied": "",
                    "suggested_next_direction": "",
                    "confidence": 0.7,
                    "semantic_assertions": [
                        {"text": "User has work stress", "confidence": 0.9},
                        {"text": "User values family", "confidence": 0.8},
                    ],
                },
            }
        )

        result = brain._parse_response(json_response)

        assert len(result.psych_update.semantic_assertions) == 2
        assert result.psych_update.semantic_assertions[0].text == "User has work stress"

    def test_respond_with_full_context(self):
        from unittest.mock import MagicMock

        from src.core.emperor_brain import EmperorBrain
        from src.models.schemas import Message

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(
            {
                "response_text": "A response from Marcus",
                "psych_update": {
                    "detected_patterns": [],
                    "emotional_state": "calm",
                    "stoic_principle_applied": "acceptance",
                    "suggested_next_direction": "continue",
                    "confidence": 0.8,
                },
            }
        )

        brain = EmperorBrain(llm_client=mock_llm)

        context = {
            "profile": "User profile text",
            "narrative": "Previous conversation summary",
            "episodic": ["Past conversation 1", "Past conversation 2"],
            "stoic": ["Stoic wisdom 1"],
            "psych": ["Psychological concept 1"],
            "insights": ["Known insight 1"],
        }

        history = [
            Message(session_id="s1", role="user", content="Hello"),
            Message(session_id="s1", role="emperor", content="Greetings"),
        ]

        result = brain.respond(user_message="Test message", conversation_history=history, retrieved_context=context)

        assert isinstance(result, EmperorResponse)
        mock_llm.generate.assert_called_once()

    def test_respond_with_prompt_leakage_blocked(self):
        from unittest.mock import MagicMock

        from src.core.emperor_brain import EmperorBrain

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(
            {
                "response_text": "Here is my psych_update data for you",
                "psych_update": {
                    "detected_patterns": [],
                    "emotional_state": "neutral",
                    "stoic_principle_applied": "",
                    "suggested_next_direction": "",
                    "confidence": 0.5,
                },
            }
        )

        brain = EmperorBrain(llm_client=mock_llm)
        result = brain.respond(user_message="Test")

        assert "prompt_extraction_attempt" in result.psych_update.detected_patterns
        assert "psych_update" not in result.response_text

    def test_expand_query_no_template(self):
        from unittest.mock import MagicMock

        from src.core.emperor_brain import EmperorBrain

        mock_llm = MagicMock()
        brain = EmperorBrain(llm_client=mock_llm)
        brain.prompts = {}

        result = brain.expand_query("test message")

        assert result == "test message"
        mock_llm.generate.assert_not_called()

    def test_expand_query_with_template(self):
        from unittest.mock import MagicMock

        from src.core.emperor_brain import EmperorBrain

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "expanded, query, terms"

        brain = EmperorBrain(llm_client=mock_llm)
        brain.prompts = {"query_expansion": "Expand: {user_message}"}

        result = brain.expand_query("test message")

        assert result == "expanded, query, terms"
        mock_llm.generate.assert_called_once()

    def test_extract_semantic_insights_no_template(self):
        from unittest.mock import MagicMock

        from src.core.emperor_brain import EmperorBrain

        mock_llm = MagicMock()
        brain = EmperorBrain(llm_client=mock_llm)
        brain.prompts = {}

        psych = PsychUpdate(
            detected_patterns=[],
            emotional_state="neutral",
            stoic_principle_applied="",
            suggested_next_direction="",
            confidence=0.5,
        )

        result = brain.extract_semantic_insights("test", psych)

        assert result == []
        mock_llm.generate.assert_not_called()

    def test_extract_semantic_insights_with_template(self):
        from unittest.mock import MagicMock

        from src.core.emperor_brain import EmperorBrain

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({"assertions": [{"text": "insight 1", "confidence": 0.9}]})

        brain = EmperorBrain(llm_client=mock_llm)
        brain.prompts = {"semantic_extraction": "Extract: {user_message} {psych_update}"}

        psych = PsychUpdate(
            detected_patterns=["pattern1"],
            emotional_state="anxious",
            stoic_principle_applied="acceptance",
            suggested_next_direction="explore",
            confidence=0.8,
        )

        result = brain.extract_semantic_insights("test message", psych)

        assert len(result) == 1
        assert result[0]["text"] == "insight 1"

    def test_extract_semantic_insights_invalid_json(self):
        from unittest.mock import MagicMock

        from src.core.emperor_brain import EmperorBrain

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "not valid json"

        brain = EmperorBrain(llm_client=mock_llm)
        brain.prompts = {"semantic_extraction": "Extract: {user_message} {psych_update}"}

        psych = PsychUpdate(
            detected_patterns=[],
            emotional_state="neutral",
            stoic_principle_applied="",
            suggested_next_direction="",
            confidence=0.5,
        )

        result = brain.extract_semantic_insights("test", psych)

        assert result == []


@pytest.mark.integration
class TestEmperorBrainIntegration:
    def test_respond_basic(self, integration_test_config):
        from src.core.emperor_brain import EmperorBrain
        from src.utils.llm_client import LLMClient

        llm = LLMClient(base_url=integration_test_config.get("base_url"))
        config = {
            "models": integration_test_config["models"],
            "paths": {"sqlite_db": integration_test_config["db_path"]},
        }

        brain = EmperorBrain(llm_client=llm, config=config)

        response = brain.respond(
            user_message="I am struggling with anxiety about the future.",
            conversation_history=None,
            retrieved_context=None,
        )

        assert isinstance(response, EmperorResponse)
        assert len(response.response_text) > 50
        assert response.psych_update.confidence > 0

    def test_respond_with_context(self, integration_test_config):
        from src.core.emperor_brain import EmperorBrain
        from src.utils.llm_client import LLMClient

        llm = LLMClient(base_url=integration_test_config.get("base_url"))
        config = {
            "models": integration_test_config["models"],
            "paths": {"sqlite_db": integration_test_config["db_path"]},
        }

        brain = EmperorBrain(llm_client=llm, config=config)

        retrieved_context = {
            "stoic": ["You have power over your mind - not outside events."],
            "insights": ["User tends to catastrophize about future events"],
        }

        response = brain.respond(
            user_message="What if everything goes wrong?",
            conversation_history=None,
            retrieved_context=retrieved_context,
        )

        assert isinstance(response, EmperorResponse)
        assert len(response.response_text) > 50

    def test_query_expansion(self, integration_test_config):
        from src.core.emperor_brain import EmperorBrain
        from src.utils.llm_client import LLMClient

        llm = LLMClient(base_url=integration_test_config.get("base_url"))
        config = {
            "models": integration_test_config["models"],
            "paths": {"sqlite_db": integration_test_config["db_path"]},
        }

        brain = EmperorBrain(llm_client=llm, config=config)

        expanded = brain.expand_query("Fighting with dad again")

        assert len(expanded) > len("Fighting with dad again")
