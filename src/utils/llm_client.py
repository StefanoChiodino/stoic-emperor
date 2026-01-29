import json
import logging
import os
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.base_url,
            timeout=timeout,
        )

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
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        is_claude = "sonnet" in model.lower() or "claude" in model.lower() or "opus" in model.lower()
        use_json_format = json_mode and not is_claude

        logger.debug(f"LLM request: model={model}, json_mode={json_mode}, use_json_format={use_json_format}")

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if use_json_format:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        content = (response.choices[0].message.content or "").strip()
        logger.debug(f"LLM response length: {len(content)}, first 200 chars: {content[:200]}")

        if not content:
            logger.error(f"Empty response from LLM. Full response object: {response}")

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
        """Get embedding for text"""
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding
