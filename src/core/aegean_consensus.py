import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import yaml
from openai import OpenAI

from src.utils.llm_adapter import AnthropicAdapter, LLMAdapter, OpenAIChatAdapter

logger = logging.getLogger(__name__)


@dataclass
class ConsensusRound:
    round_number: int
    model_a_output: str
    model_b_output: str
    model_a_review: dict | None = None
    model_b_review: dict | None = None
    consensus_reached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsensusResult:
    final_output: str
    consensus_reached: bool
    rounds: list[ConsensusRound]
    model_a_name: str
    model_b_name: str
    stability_score: float
    critical_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "consensus_reached": self.consensus_reached,
            "rounds": len(self.rounds),
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "stability_score": self.stability_score,
            "critical_flags": self.critical_flags,
            "metadata": self.metadata,
        }


def _is_claude_model(model: str) -> bool:
    model_lower = model.lower()
    return "claude" in model_lower or "sonnet" in model_lower or "opus" in model_lower or "haiku" in model_lower


class AegeanConsensusProtocol:
    def __init__(
        self,
        model_a: str,
        model_b: str,
        prompts: dict[str, str] | None = None,
        beta_threshold: int = 2,
        alpha_quorum: float = 1.0,
        verbose: bool = True,
        output_folder: str = "./data/output/consensus_logs",
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.prompts = prompts or self._load_prompts()
        self.beta_threshold = beta_threshold
        self.alpha_quorum = alpha_quorum
        self.verbose = verbose
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        base_url = os.getenv("LLM_BASE_URL")
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        self.openai_client: OpenAI | None = None
        self.anthropic_client: anthropic.Anthropic | None = None

        if openai_key:
            self.openai_client = OpenAI(
                api_key=openai_key,
                base_url=base_url if base_url and "anthropic" not in base_url else None,
            )
        if anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)

    def _load_prompts(self) -> dict[str, str]:
        prompts_path = Path("config/prompts.yaml")
        if prompts_path.exists():
            with open(prompts_path) as f:
                return yaml.safe_load(f)
        return {}

    def reach_consensus(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        context: str = "",
        max_rounds: int | None = None,
        temperature: float = 0.7,
        critical_constructs: list[str] | None = None,
        use_model_a_on_failure: bool = True,
    ) -> ConsensusResult:
        if max_rounds is None:
            max_rounds = self.beta_threshold

        if self.verbose:
            print(f"\n🔄 Aegean Consensus: {prompt_name}")
            print(f"   Models: {self.model_a} ⇄ {self.model_b}")

        rounds: list[ConsensusRound] = []
        consensus_reached = False
        final_output = ""
        consecutive_approvals = 0

        for round_num in range(1, max_rounds + 1):
            if self.verbose:
                print(f"\n   Round {round_num}/{max_rounds}")

            output_a = self._generate(self.model_a, prompt_name, variables, temperature)
            output_b = self._generate(self.model_b, prompt_name, variables, temperature)

            original_data = variables.get("source_data", "")

            review_a_of_b = self._review_output(self.model_a, output_b, context, critical_constructs, original_data)
            review_b_of_a = self._review_output(self.model_b, output_a, context, critical_constructs, original_data)

            current_round = ConsensusRound(
                round_number=round_num,
                model_a_output=output_a,
                model_b_output=output_b,
                model_a_review=review_a_of_b,
                model_b_review=review_b_of_a,
            )

            a_approves = review_b_of_a.get("approved", False)
            b_approves = review_a_of_b.get("approved", False)

            if self.verbose:
                print(f"      {self.model_a} approval: {a_approves}")
                print(f"      {self.model_b} approval: {b_approves}")

            if a_approves and b_approves:
                consecutive_approvals += 1
                current_round.consensus_reached = True

                if consecutive_approvals >= self.beta_threshold:
                    consensus_reached = True
                    final_output = self._merge_outputs(output_a, output_b, review_a_of_b, review_b_of_a)
                    rounds.append(current_round)
                    if self.verbose:
                        print(f"   ✅ Consensus reached after {round_num} round(s)")
                    break
            else:
                consecutive_approvals = 0
                current_round.consensus_reached = False

            rounds.append(current_round)

            if round_num < max_rounds and not consensus_reached:
                variables = self._update_variables_with_feedback(
                    variables, output_a, output_b, review_a_of_b, review_b_of_a
                )

        if not consensus_reached:
            if use_model_a_on_failure:
                if self.verbose:
                    print(f"   ⚠️  No consensus - using {self.model_a} output")
                final_output = rounds[-1].model_a_output if rounds else ""
            else:
                final_output = self._create_no_consensus_output(rounds)

        critical_flags = self._check_critical_disagreements(rounds, critical_constructs or [])
        stability_score = self._calculate_stability_score(rounds)

        result = ConsensusResult(
            final_output=final_output,
            consensus_reached=consensus_reached,
            rounds=rounds,
            model_a_name=self.model_a,
            model_b_name=self.model_b,
            stability_score=stability_score,
            critical_flags=critical_flags,
            metadata={"rounds_needed": len(rounds), "max_rounds": max_rounds},
        )

        self._log_consensus(result, prompt_name)
        return result

    def _get_adapter_for_model(self, model: str) -> tuple[LLMAdapter, str]:
        is_claude = _is_claude_model(model)
        if is_claude and self.anthropic_client:
            return AnthropicAdapter(self.anthropic_client), "anthropic"
        elif self.openai_client:
            return OpenAIChatAdapter(self.openai_client), "openai"
        elif self.anthropic_client:
            return AnthropicAdapter(self.anthropic_client), "anthropic"
        else:
            raise ValueError("No LLM client configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")

    def _log_usage(self, model: str, client_type: str, input_tokens: int | None, output_tokens: int | None) -> None:
        if input_tokens is None and output_tokens is None:
            return
        logger.info(
            "LLM tokens: model=%s client=%s input=%s output=%s",
            model,
            client_type,
            input_tokens,
            output_tokens,
        )

    def _call_llm(self, model: str, prompt: str, temperature: float, max_tokens: int = 4000) -> str:
        adapter, client_type = self._get_adapter_for_model(model)
        result = adapter.generate(
            prompt=prompt,
            system_prompt="",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
        )
        self._log_usage(model, client_type, result.input_tokens, result.output_tokens)
        return result.content

    def _generate(self, model: str, prompt_name: str, variables: dict[str, Any], temperature: float) -> str:
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")

        prompt_template = self.prompts[prompt_name]
        prompt = prompt_template.format(**variables)
        return self._call_llm(model, prompt, temperature)

    def _review_output(
        self,
        model: str,
        output_to_review: str,
        context: str,
        critical_constructs: list[str] | None,
        original_data: str = "",
    ) -> dict:
        review_prompt = f"""Review the following analysis for accuracy and completeness.

Analysis to review:
{output_to_review}

Original source data:
{original_data[:2000] if original_data else "Not provided"}

Critical areas to check: {", ".join(critical_constructs) if critical_constructs else "general quality"}

Respond with JSON:
{{
  "approved": true/false,
  "strengths": ["strength 1", ...],
  "concerns": [{{"issue": "...", "severity": "minor/moderate/critical"}}],
  "reasoning": "Brief explanation"
}}"""

        review_text = self._call_llm(model, review_prompt, temperature=0.3, max_tokens=1000)

        try:
            start = review_text.find("{")
            end = review_text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(review_text[start : end + 1])
        except json.JSONDecodeError:
            pass

        return {"approved": False, "reasoning": review_text, "concerns": []}

    def _merge_outputs(self, output_a: str, output_b: str, review_a: dict, review_b: dict) -> str:
        strengths_a = len(review_b.get("strengths", []))
        strengths_b = len(review_a.get("strengths", []))
        return output_a if strengths_a >= strengths_b else output_b

    def _update_variables_with_feedback(
        self, variables: dict[str, Any], output_a: str, output_b: str, review_a: dict, review_b: dict
    ) -> dict[str, Any]:
        updated = variables.copy()
        updated["previous_feedback"] = f"Feedback: {review_a.get('reasoning', '')} | {review_b.get('reasoning', '')}"
        return updated

    def _create_no_consensus_output(self, rounds: list[ConsensusRound]) -> str:
        last = rounds[-1]
        return f"""# Analysis - Manual Review Required

## Model A ({self.model_a})
{last.model_a_output}

## Model B ({self.model_b})
{last.model_b_output}
"""

    def _check_critical_disagreements(self, rounds: list[ConsensusRound], critical_constructs: list[str]) -> list[str]:
        if not rounds or not critical_constructs:
            return []

        flags = []
        last = rounds[-1]

        for construct in critical_constructs:
            concerns_a = last.model_a_review.get("concerns", []) if last.model_a_review else []
            concerns_b = last.model_b_review.get("concerns", []) if last.model_b_review else []

            for concern in concerns_a + concerns_b:
                if isinstance(concern, dict) and construct.lower() in concern.get("issue", "").lower():
                    flags.append(f"Critical disagreement: {construct}")
                    break

        return flags

    def _calculate_stability_score(self, rounds: list[ConsensusRound]) -> float:
        if not rounds:
            return 0.0
        return sum(1 for r in rounds if r.consensus_reached) / len(rounds)

    def _log_consensus(self, result: ConsensusResult, prompt_name: str) -> Path:
        timestamp = datetime.now()
        log_id = f"{prompt_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        log_file = self.output_folder / f"{log_id}.json"

        log_data = {"log_id": log_id, "timestamp": timestamp.isoformat(), **result.to_dict()}

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        return log_file
