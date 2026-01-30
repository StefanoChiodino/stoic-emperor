import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.core.aegean_consensus import AegeanConsensusProtocol, ConsensusResult, ConsensusRound


class TestConsensusRound:
    def test_round_initialization(self):
        round = ConsensusRound(
            round_number=1,
            model_a_output="Output A",
            model_b_output="Output B",
        )

        assert round.round_number == 1
        assert round.model_a_output == "Output A"
        assert round.model_b_output == "Output B"
        assert round.model_a_review is None
        assert round.model_b_review is None
        assert round.consensus_reached is False
        assert isinstance(round.timestamp, datetime)

    def test_round_with_reviews(self):
        round = ConsensusRound(
            round_number=2,
            model_a_output="Output A",
            model_b_output="Output B",
            model_a_review={"approved": True, "strengths": ["good"]},
            model_b_review={"approved": True, "reasoning": "excellent"},
            consensus_reached=True,
        )

        assert round.model_a_review["approved"] is True
        assert round.model_b_review["approved"] is True
        assert round.consensus_reached is True


class TestConsensusResult:
    def test_result_initialization(self):
        rounds = [
            ConsensusRound(round_number=1, model_a_output="A", model_b_output="B"),
        ]
        result = ConsensusResult(
            final_output="Final output",
            consensus_reached=True,
            rounds=rounds,
            model_a_name="gpt-4",
            model_b_name="claude-3",
            stability_score=1.0,
        )

        assert result.final_output == "Final output"
        assert result.consensus_reached is True
        assert len(result.rounds) == 1
        assert result.model_a_name == "gpt-4"
        assert result.stability_score == 1.0
        assert result.critical_flags == []
        assert result.metadata == {}

    def test_to_dict(self):
        rounds = [
            ConsensusRound(round_number=1, model_a_output="A", model_b_output="B", consensus_reached=True),
            ConsensusRound(round_number=2, model_a_output="A2", model_b_output="B2", consensus_reached=True),
        ]
        result = ConsensusResult(
            final_output="Final",
            consensus_reached=True,
            rounds=rounds,
            model_a_name="model-a",
            model_b_name="model-b",
            stability_score=0.75,
            critical_flags=["flag1"],
            metadata={"key": "value"},
        )

        d = result.to_dict()

        assert d["consensus_reached"] is True
        assert d["rounds"] == 2
        assert d["model_a"] == "model-a"
        assert d["model_b"] == "model-b"
        assert d["stability_score"] == 0.75
        assert d["critical_flags"] == ["flag1"]
        assert d["metadata"] == {"key": "value"}


class TestAegeanConsensusProtocol:
    @pytest.fixture
    def mock_openai_client(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test output"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def consensus_protocol(self, temp_dir):
        with patch("src.core.aegean_consensus.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            protocol = AegeanConsensusProtocol(
                model_a="gpt-4",
                model_b="claude-3",
                prompts={"test_prompt": "Test: {input}"},
                beta_threshold=2,
                verbose=False,
                output_folder=str(temp_dir / "consensus_logs"),
            )
            protocol.openai_client = mock_client
            return protocol

    def test_initialization(self, temp_dir):
        with patch("src.core.aegean_consensus.OpenAI"):
            protocol = AegeanConsensusProtocol(
                model_a="gpt-4",
                model_b="claude-3",
                prompts={"test": "template"},
                beta_threshold=3,
                alpha_quorum=0.8,
                verbose=False,
                output_folder=str(temp_dir / "logs"),
            )

            assert protocol.model_a == "gpt-4"
            assert protocol.model_b == "claude-3"
            assert protocol.beta_threshold == 3
            assert protocol.alpha_quorum == 0.8
            assert protocol.prompts == {"test": "template"}

    def test_load_prompts_file_exists(self, temp_dir):
        prompts_content = "test_prompt: 'Test template'\n"
        prompts_path = temp_dir / "config"
        prompts_path.mkdir()
        (prompts_path / "prompts.yaml").write_text(prompts_content)

        with patch("src.core.aegean_consensus.OpenAI"), patch("src.core.aegean_consensus.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            protocol = AegeanConsensusProtocol(
                model_a="gpt-4",
                model_b="claude-3",
                verbose=False,
                output_folder=str(temp_dir / "logs"),
            )
            assert isinstance(protocol.prompts, dict)

    def test_generate(self, consensus_protocol):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated content"
        consensus_protocol.openai_client.chat.completions.create.return_value = mock_response

        result = consensus_protocol._generate(
            "gpt-4",
            "test_prompt",
            {"input": "test input"},
            0.7,
        )

        assert result == "Generated content"
        consensus_protocol.openai_client.chat.completions.create.assert_called_once()

    def test_generate_missing_prompt(self, consensus_protocol):
        with pytest.raises(ValueError, match="Prompt 'nonexistent' not found"):
            consensus_protocol._generate(
                "gpt-4",
                "nonexistent",
                {},
                0.7,
            )

    def test_review_output_valid_json(self, consensus_protocol):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "approved": True,
                "strengths": ["clear", "accurate"],
                "concerns": [],
                "reasoning": "Good analysis",
            }
        )
        consensus_protocol.openai_client.chat.completions.create.return_value = mock_response

        result = consensus_protocol._review_output(
            "gpt-4",
            "Output to review",
            "context",
            ["critical1"],
            "original data",
        )

        assert result["approved"] is True
        assert "strengths" in result

    def test_review_output_invalid_json(self, consensus_protocol):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Not valid JSON at all"
        consensus_protocol.openai_client.chat.completions.create.return_value = mock_response

        result = consensus_protocol._review_output(
            "gpt-4",
            "Output to review",
            "context",
            None,
            "",
        )

        assert result["approved"] is False
        assert "reasoning" in result

    def test_merge_outputs_a_stronger(self, consensus_protocol):
        review_a = {"strengths": ["1"]}
        review_b = {"strengths": ["1", "2", "3"]}

        result = consensus_protocol._merge_outputs("Output A", "Output B", review_a, review_b)

        assert result == "Output A"

    def test_merge_outputs_b_stronger(self, consensus_protocol):
        review_a = {"strengths": ["1", "2", "3", "4"]}
        review_b = {"strengths": ["1"]}

        result = consensus_protocol._merge_outputs("Output A", "Output B", review_a, review_b)

        assert result == "Output B"

    def test_update_variables_with_feedback(self, consensus_protocol):
        variables = {"input": "original", "other": "value"}
        review_a = {"reasoning": "Review A reasoning"}
        review_b = {"reasoning": "Review B reasoning"}

        updated = consensus_protocol._update_variables_with_feedback(
            variables, "output_a", "output_b", review_a, review_b
        )

        assert updated["input"] == "original"
        assert "previous_feedback" in updated
        assert "Review A" in updated["previous_feedback"]

    def test_create_no_consensus_output(self, consensus_protocol):
        rounds = [ConsensusRound(round_number=1, model_a_output="Model A analysis", model_b_output="Model B analysis")]

        result = consensus_protocol._create_no_consensus_output(rounds)

        assert "Model A" in result
        assert "Model B" in result
        assert "Manual Review Required" in result

    def test_check_critical_disagreements_empty(self, consensus_protocol):
        rounds = []
        result = consensus_protocol._check_critical_disagreements(rounds, ["construct1"])
        assert result == []

    def test_check_critical_disagreements_no_constructs(self, consensus_protocol):
        rounds = [ConsensusRound(round_number=1, model_a_output="A", model_b_output="B")]
        result = consensus_protocol._check_critical_disagreements(rounds, [])
        assert result == []

    def test_check_critical_disagreements_with_flags(self, consensus_protocol):
        rounds = [
            ConsensusRound(
                round_number=1,
                model_a_output="A",
                model_b_output="B",
                model_a_review={"concerns": [{"issue": "depression risk", "severity": "critical"}]},
                model_b_review={"concerns": []},
            )
        ]

        result = consensus_protocol._check_critical_disagreements(rounds, ["depression"])

        assert len(result) == 1
        assert "depression" in result[0]

    def test_calculate_stability_score_empty(self, consensus_protocol):
        result = consensus_protocol._calculate_stability_score([])
        assert result == 0.0

    def test_calculate_stability_score_all_consensus(self, consensus_protocol):
        rounds = [
            ConsensusRound(round_number=1, model_a_output="A", model_b_output="B", consensus_reached=True),
            ConsensusRound(round_number=2, model_a_output="A", model_b_output="B", consensus_reached=True),
        ]

        result = consensus_protocol._calculate_stability_score(rounds)

        assert result == 1.0

    def test_calculate_stability_score_mixed(self, consensus_protocol):
        rounds = [
            ConsensusRound(round_number=1, model_a_output="A", model_b_output="B", consensus_reached=True),
            ConsensusRound(round_number=2, model_a_output="A", model_b_output="B", consensus_reached=False),
        ]

        result = consensus_protocol._calculate_stability_score(rounds)

        assert result == 0.5

    def test_log_consensus(self, consensus_protocol, temp_dir):
        result = ConsensusResult(
            final_output="Final",
            consensus_reached=True,
            rounds=[],
            model_a_name="gpt-4",
            model_b_name="claude-3",
            stability_score=1.0,
        )

        log_path = consensus_protocol._log_consensus(result, "test_prompt")

        assert log_path.exists()
        with open(log_path) as f:
            log_data = json.load(f)
        assert log_data["consensus_reached"] is True

    def test_reach_consensus_success(self, consensus_protocol):
        def make_generate_response():
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = "Generated output"
            return r

        def make_review_response():
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = json.dumps(
                {
                    "approved": True,
                    "strengths": ["good"],
                    "concerns": [],
                    "reasoning": "OK",
                }
            )
            return r

        call_count = {"calls": 0}

        def side_effect(*args, **kwargs):
            call_count["calls"] += 1
            if call_count["calls"] % 2 == 1:
                return make_generate_response()
            return make_review_response()

        consensus_protocol.openai_client.chat.completions.create.side_effect = side_effect

        result = consensus_protocol.reach_consensus(
            prompt_name="test_prompt",
            variables={"input": "test"},
            max_rounds=2,
        )

        assert isinstance(result, ConsensusResult)

    def test_reach_consensus_no_consensus(self, consensus_protocol):
        def make_generate_response():
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = "Generated output"
            return r

        def make_review_response():
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = json.dumps(
                {
                    "approved": False,
                    "strengths": [],
                    "concerns": [{"issue": "bad", "severity": "critical"}],
                    "reasoning": "Not good",
                }
            )
            return r

        call_count = {"calls": 0}

        def side_effect(*args, **kwargs):
            call_count["calls"] += 1
            if call_count["calls"] % 2 == 1:
                return make_generate_response()
            return make_review_response()

        consensus_protocol.openai_client.chat.completions.create.side_effect = side_effect

        result = consensus_protocol.reach_consensus(
            prompt_name="test_prompt",
            variables={"input": "test"},
            max_rounds=1,
            use_model_a_on_failure=True,
        )

        assert result.consensus_reached is False
        assert result.final_output == "Generated output"

    def test_reach_consensus_no_consensus_fallback(self, consensus_protocol):
        def make_generate_response():
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = "Model output"
            return r

        def make_review_response():
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = json.dumps(
                {
                    "approved": False,
                    "strengths": [],
                    "concerns": [],
                    "reasoning": "Not approved",
                }
            )
            return r

        call_count = {"calls": 0}

        def side_effect(*args, **kwargs):
            call_count["calls"] += 1
            if call_count["calls"] % 2 == 1:
                return make_generate_response()
            return make_review_response()

        consensus_protocol.openai_client.chat.completions.create.side_effect = side_effect

        result = consensus_protocol.reach_consensus(
            prompt_name="test_prompt",
            variables={"input": "test"},
            max_rounds=1,
            use_model_a_on_failure=False,
        )

        assert result.consensus_reached is False
        assert "Manual Review Required" in result.final_output
