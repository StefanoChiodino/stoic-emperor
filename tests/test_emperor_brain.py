import pytest
import json

from src.models.schemas import EmperorResponse, PsychUpdate


class TestEmperorResponseParsing:
    def test_parse_valid_response(self, mock_emperor_response):
        psych_data = mock_emperor_response["psych_update"]
        psych = PsychUpdate(**psych_data)
        response = EmperorResponse(
            response_text=mock_emperor_response["response_text"],
            psych_update=psych
        )
        
        assert "father" in response.response_text.lower()
        assert len(response.psych_update.detected_patterns) > 0

    def test_psych_update_validation(self):
        psych = PsychUpdate(
            detected_patterns=["pattern1", "pattern2"],
            emotional_state="anxious",
            stoic_principle_applied="Amor Fati",
            suggested_next_direction="Explore acceptance",
            confidence=0.75
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
        from src.core.emperor_brain import EmperorBrain
        from unittest.mock import MagicMock
        
        mock_llm = MagicMock()
        brain = EmperorBrain(llm_client=mock_llm)
        
        assert brain.llm == mock_llm

    def test_parse_response_valid_json(self):
        from src.core.emperor_brain import EmperorBrain
        from unittest.mock import MagicMock
        
        brain = EmperorBrain(llm_client=MagicMock())
        
        json_response = json.dumps({
            "response_text": "Test response from Marcus",
            "psych_update": {
                "detected_patterns": ["test_pattern"],
                "emotional_state": "calm",
                "stoic_principle_applied": "Virtue",
                "suggested_next_direction": "Continue",
                "confidence": 0.9
            }
        })
        
        result = brain._parse_response(json_response)
        
        assert isinstance(result, EmperorResponse)
        assert result.response_text == "Test response from Marcus"
        assert result.psych_update.confidence == 0.9

    def test_parse_response_invalid_json(self):
        from src.core.emperor_brain import EmperorBrain
        from unittest.mock import MagicMock
        
        brain = EmperorBrain(llm_client=MagicMock())
        
        result = brain._parse_response("This is not JSON")
        
        assert isinstance(result, EmperorResponse)
        assert "parse_error" in result.psych_update.detected_patterns


@pytest.mark.integration
class TestEmperorBrainIntegration:
    def test_respond_basic(self, integration_test_config):
        from src.core.emperor_brain import EmperorBrain
        from src.utils.llm_client import LLMClient
        
        llm = LLMClient(base_url=integration_test_config.get("base_url"))
        config = {
            "models": integration_test_config["models"],
            "paths": {"sqlite_db": integration_test_config["db_path"]}
        }
        
        brain = EmperorBrain(llm_client=llm, config=config)
        
        response = brain.respond(
            user_message="I am struggling with anxiety about the future.",
            conversation_history=None,
            retrieved_context=None
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
            "paths": {"sqlite_db": integration_test_config["db_path"]}
        }
        
        brain = EmperorBrain(llm_client=llm, config=config)
        
        retrieved_context = {
            "stoic": ["You have power over your mind - not outside events."],
            "insights": ["User tends to catastrophize about future events"]
        }
        
        response = brain.respond(
            user_message="What if everything goes wrong?",
            conversation_history=None,
            retrieved_context=retrieved_context
        )
        
        assert isinstance(response, EmperorResponse)
        assert len(response.response_text) > 50

    def test_query_expansion(self, integration_test_config):
        from src.core.emperor_brain import EmperorBrain
        from src.utils.llm_client import LLMClient
        
        llm = LLMClient(base_url=integration_test_config.get("base_url"))
        config = {
            "models": integration_test_config["models"],
            "paths": {"sqlite_db": integration_test_config["db_path"]}
        }
        
        brain = EmperorBrain(llm_client=llm, config=config)
        
        expanded = brain.expand_query("Fighting with dad again")
        
        assert len(expanded) > len("Fighting with dad again")
