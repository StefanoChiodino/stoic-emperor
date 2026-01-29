import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.models.schemas import EmperorResponse, PsychUpdate, Message
from src.utils.llm_client import LLMClient
from src.utils.config import load_config
from src.utils.response_guard import guard_response


class EmperorBrain:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or load_config()
        self.llm = llm_client or LLMClient()
        self.prompts = self._load_prompts()
        self.model = self.config["models"]["main"]
        self._system_prompt = self.prompts.get("marcus_aurelius_system", "")

    def _load_prompts(self) -> Dict[str, str]:
        prompts_path = Path("config/prompts.yaml")
        if prompts_path.exists():
            with open(prompts_path) as f:
                return yaml.safe_load(f)
        return {}

    def respond(
        self,
        user_message: str,
        conversation_history: Optional[List[Message]] = None,
        retrieved_context: Optional[Dict[str, Any]] = None
    ) -> EmperorResponse:
        prompt_parts = []

        profile_text = "No profile yet - this is a new user."
        if retrieved_context:
            if retrieved_context.get("profile"):
                profile_text = retrieved_context["profile"]

            if retrieved_context.get("stoic"):
                prompt_parts.append("## Relevant Stoic Wisdom")
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
        system_prompt = self._system_prompt.format(profile=profile_text)

        response_text = self.llm.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            model=self.model,
            temperature=0.7,
            max_tokens=2000,
            json_mode=True
        )

        response = self._parse_response(response_text)
        guarded_text, was_blocked = guard_response(
            response.response_text,
            self._system_prompt
        )
        if was_blocked:
            response.response_text = guarded_text
            response.psych_update.detected_patterns.append("prompt_extraction_attempt")

        return response

    def _parse_response(self, response_text: str) -> EmperorResponse:
        try:
            data = json.loads(response_text)

            psych_data = data.get("psych_update", {})
            psych_update = PsychUpdate(
                detected_patterns=psych_data.get("detected_patterns", []),
                emotional_state=psych_data.get("emotional_state", "unknown"),
                stoic_principle_applied=psych_data.get("stoic_principle_applied", ""),
                suggested_next_direction=psych_data.get("suggested_next_direction", ""),
                confidence=psych_data.get("confidence", 0.5)
            )

            return EmperorResponse(
                response_text=data.get("response_text", "I must reflect further on this."),
                psych_update=psych_update
            )
        except (json.JSONDecodeError, KeyError) as e:
            return EmperorResponse(
                response_text=response_text if isinstance(response_text, str) else "I must reflect further on this.",
                psych_update=PsychUpdate(
                    detected_patterns=["parse_error"],
                    emotional_state="unknown",
                    stoic_principle_applied="",
                    suggested_next_direction="Retry with clearer structure",
                    confidence=0.0
                )
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
            max_tokens=200
        )

    def extract_semantic_insights(
        self,
        user_message: str,
        psych_update: PsychUpdate
    ) -> List[Dict[str, Any]]:
        prompt_template = self.prompts.get("semantic_extraction", "")
        if not prompt_template:
            return []

        prompt = prompt_template.format(
            user_message=user_message,
            psych_update=psych_update.model_dump_json()
        )

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are extracting psychological insights for long-term memory.",
            model=self.config["models"]["main"],
            temperature=0.3,
            max_tokens=500,
            json_mode=True
        )

        try:
            data = json.loads(response)
            return data.get("assertions", [])
        except json.JSONDecodeError:
            return []
