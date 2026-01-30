import json
import logging
import os
from typing import Any

import anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.llm_adapter import AnthropicAdapter, LLMAdapter, OpenAIChatAdapter

logger = logging.getLogger(__name__)


def _is_claude_model(model: str) -> bool:
    model_lower = model.lower()
    return "claude" in model_lower or "sonnet" in model_lower or "opus" in model_lower or "haiku" in model_lower


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.timeout = timeout

        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        self.openai_client: OpenAI | None = None
        self.anthropic_client: anthropic.Anthropic | None = None

        if openai_key:
            self.openai_client = OpenAI(
                api_key=openai_key,
                base_url=self.base_url if self.base_url and "anthropic" not in self.base_url else None,
                timeout=timeout,
            )

        if anthropic_key:
            self.anthropic_client = anthropic.Anthropic(
                api_key=anthropic_key,
                timeout=timeout,
            )

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

    def _get_adapter_for_model(self, model: str) -> tuple[LLMAdapter, str]:
        is_claude = _is_claude_model(model)

        if is_claude and self.anthropic_client:
            return AnthropicAdapter(self.anthropic_client), "anthropic"
        elif self.openai_client:
            return OpenAIChatAdapter(self.openai_client), "openai"
        elif self.anthropic_client:
            return AnthropicAdapter(self.anthropic_client), "anthropic"
        else:
            raise ValueError("No LLM client configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        json_mode: bool = False,
    ) -> str:
        adapter, client_type = self._get_adapter_for_model(model)
        is_claude = _is_claude_model(model)

        logger.debug(f"LLM request: model={model}, client={client_type}, json_mode={json_mode}")

        adapter_json_mode = json_mode if client_type == "anthropic" else json_mode and not is_claude
        result = adapter.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=adapter_json_mode,
        )
        content = result.content.strip()

        self._log_usage(model, client_type, result.input_tokens, result.output_tokens)
        logger.debug(f"LLM response length: {len(content)}, first 200 chars: {content[:200]}")

        if not content:
            logger.error("Empty response from LLM")

        return content

    def generate_structured(
        self,
        prompt: str,
        response_model: Any,  # Pydantic model class
        system_prompt: str = "You are a helpful assistant.",
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ) -> Any:
        """Generate structured data using Pydantic model (requires instructor or similar, but here we use JSON mode + parsing)"""
        # Note: In a real prod app, I'd use the `instructor` library or OpenAI's new structured outputs.
        # For now, we'll use JSON mode and manual parsing for simplicity and compatibility.

        schema = response_model.model_json_schema()
        json_prompt = (
            f"{prompt}\n\nRespond with a valid JSON object matching this schema:\n{json.dumps(schema, indent=2)}"
        )

        response_text = self.generate(
            prompt=json_prompt, system_prompt=system_prompt, model=model, temperature=temperature, json_mode=True
        )

        try:
            data = json.loads(response_text)
            return response_model(**data)
        except Exception as e:
            print(f"Failed to parse JSON or validate model: {e}")
            print(f"Raw response: {response_text}")
            raise

    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        if not self.openai_client:
            raise ValueError("Embeddings require OpenAI API. Set OPENAI_API_KEY environment variable.")
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(input=[text], model=model).data[0].embedding
