from dataclasses import dataclass
from typing import Protocol

import anthropic
from anthropic.types import Message, TextBlock
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


@dataclass(frozen=True)
class LLMResult:
    content: str
    input_tokens: int | None
    output_tokens: int | None


class LLMAdapter(Protocol):
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult: ...


class OpenAIChatAdapter:
    def __init__(self, client: OpenAI):
        self.client = client

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        if json_mode:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        content = (response.choices[0].message.content or "").strip()
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        return LLMResult(content=content, input_tokens=input_tokens, output_tokens=output_tokens)


class AnthropicAdapter:
    def __init__(self, client: anthropic.Anthropic):
        self.client = client

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult:
        json_instruction = "\n\nRespond with valid JSON only." if json_mode else ""
        message_content = prompt + json_instruction
        if system_prompt:
            response: Message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": message_content}],
            )
        else:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": message_content}],
            )
        content = "".join(block.text for block in response.content if isinstance(block, TextBlock))
        usage = response.usage
        input_tokens = usage.input_tokens if usage else None
        output_tokens = usage.output_tokens if usage else None
        return LLMResult(content=content.strip(), input_tokens=input_tokens, output_tokens=output_tokens)
