import json
import logging
from pathlib import Path
from typing import Any

import yaml

from src.models.schemas import EmperorResponse, Message, PsychUpdate, SemanticAssertion
from src.utils.config import load_config
from src.utils.llm_client import LLMClient
from src.utils.response_guard import guard_response

logger = logging.getLogger(__name__)


class EmperorBrain:
    def __init__(self, llm_client: LLMClient | None = None, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.llm = llm_client or LLMClient()
        self.prompts = self._load_prompts()
        self.model = self.config["models"]["main"]
        self._system_prompt = self.prompts.get("marcus_aurelius_system", "")

    def _load_prompts(self) -> dict[str, str]:
        prompts_path = Path("config/prompts.yaml")
        if prompts_path.exists():
            with open(prompts_path) as f:
                return yaml.safe_load(f)
        return {}

    def respond(
        self,
        user_message: str,
        conversation_history: list[Message] | None = None,
        retrieved_context: dict[str, Any] | None = None,
    ) -> EmperorResponse:
        prompt_parts = []

        profile_text = "No profile yet - this is a new user."
        narrative_text = "No conversation history yet."
        if retrieved_context:
            if retrieved_context.get("profile"):
                profile_text = retrieved_context["profile"]

            if retrieved_context.get("narrative"):
                narrative_text = retrieved_context["narrative"]

            if retrieved_context.get("episodic"):
                prompt_parts.append("## Relevant Past Conversations")
                for item in retrieved_context["episodic"][:3]:
                    prompt_parts.append(f"- {item}")

            if retrieved_context.get("stoic"):
                prompt_parts.append("\n## Relevant Stoic Wisdom")
                for item in retrieved_context["stoic"][:3]:
                    prompt_parts.append(f"- {item}")

            if retrieved_context.get("psych"):
                prompt_parts.append("\n## Relevant Psychological Concepts")
                for item in retrieved_context["psych"][:3]:
                    prompt_parts.append(f"- {item}")

            if retrieved_context.get("insights"):
                prompt_parts.append("\n## Known About This Person")
                for item in retrieved_context["insights"][:5]:
                    prompt_parts.append(f"- {item}")

        if conversation_history:
            prompt_parts.append("\n## Recent Conversation")
            for msg in conversation_history[-10:]:
                role = "User" if msg.role == "user" else "Marcus"
                prompt_parts.append(f"{role}: {msg.content}")

        prompt_parts.append(f"\n## Current Message\nUser: {user_message}")
        prompt_parts.append("\nRespond with the JSON object as specified.")

        full_prompt = "\n".join(prompt_parts)
        system_prompt = self._system_prompt.format(profile=profile_text, narrative=narrative_text)

        max_attempts = 2
        last_error = None

        for attempt in range(max_attempts):
            try:
                response_text = self.llm.generate(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    model=self.model,
                    temperature=0.7 + (attempt * 0.1),
                    max_tokens=2000,
                    json_mode=True,
                )

                response = self._parse_response(response_text)
                guarded_text, was_blocked = guard_response(response.response_text, self._system_prompt)
                if was_blocked:
                    response.response_text = guarded_text
                    response.psych_update.detected_patterns.append("prompt_extraction_attempt")

                return response

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = e
                logger.warning(f"Response parse attempt {attempt + 1} failed: {e}")
                continue

        logger.error(f"All {max_attempts} attempts failed. Last error: {last_error}")
        return EmperorResponse(
            response_text="Something disrupted my thoughts. Speak again, and I shall attend more carefully.",
            psych_update=self._empty_psych_update(["response_generation_failed"]),
        )

    def _strip_markdown_fences(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text.strip()

    def _parse_response(self, response_text: str) -> EmperorResponse:
        try:
            cleaned = self._strip_markdown_fences(response_text)
            data = json.loads(cleaned)

            psych_data = data.get("psych_update", {})
            raw_assertions = psych_data.get("semantic_assertions", [])
            semantic_assertions = [
                SemanticAssertion(text=a.get("text", ""), confidence=a.get("confidence", 0.5))
                for a in raw_assertions
                if a.get("text")
            ]

            psych_update = PsychUpdate(
                detected_patterns=psych_data.get("detected_patterns", []),
                emotional_state=psych_data.get("emotional_state", "unknown"),
                stoic_principle_applied=psych_data.get("stoic_principle_applied", ""),
                suggested_next_direction=psych_data.get("suggested_next_direction", ""),
                confidence=psych_data.get("confidence", 0.5),
                semantic_assertions=semantic_assertions,
            )

            user_response = data.get("response_text") or data.get("text") or data.get("reply")
            if not user_response:
                logger.error(
                    f"LLM returned JSON without response_text. Keys: {list(data.keys())}. Full: {cleaned[:500]}"
                )
                raise ValueError("Missing response_text in LLM output")

            return EmperorResponse(response_text=user_response, psych_update=psych_update)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}. Raw response: {response_text[:500]}")
            raise

        except (KeyError, ValueError) as e:
            logger.error(f"Parse error: {e}. Raw response: {response_text[:500]}")
            raise

    def _empty_psych_update(self, patterns: list[str] | None = None) -> PsychUpdate:
        return PsychUpdate(
            detected_patterns=patterns or [],
            emotional_state="unknown",
            stoic_principle_applied="",
            suggested_next_direction="",
            confidence=0.0,
        )

    def expand_query(self, user_message: str) -> str:
        prompt_template = self.prompts.get("query_expansion", "")
        if not prompt_template:
            return user_message

        prompt = prompt_template.format(user_message=user_message)

        return self.llm.generate(
            prompt=prompt,
            system_prompt="You are a search query expansion assistant.",
            model=self.config["models"].get("light", self.config["models"]["main"]),
            temperature=0.3,
            max_tokens=200,
        )

    def extract_semantic_insights(self, user_message: str, psych_update: PsychUpdate) -> list[dict[str, Any]]:
        prompt_template = self.prompts.get("semantic_extraction", "")
        if not prompt_template:
            return []

        prompt = prompt_template.format(user_message=user_message, psych_update=psych_update.model_dump_json())

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are extracting psychological insights for long-term memory.",
            model=self.config["models"]["main"],
            temperature=0.3,
            max_tokens=500,
            json_mode=True,
        )

        try:
            data = json.loads(response)
            return data.get("assertions", [])
        except json.JSONDecodeError:
            return []
