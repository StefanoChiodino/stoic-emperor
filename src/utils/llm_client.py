import json
import os
from typing import Dict, Any, Optional, List, Union
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.base_url
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        json_mode: bool = False
    ) -> str:
        """Generate text from LLM"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response_format = {"type": "json_object"} if json_mode else None
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        
        return response.choices[0].message.content.strip()

    def generate_structured(
        self,
        prompt: str,
        response_model: Any,  # Pydantic model class
        system_prompt: str = "You are a helpful assistant.",
        model: str = "gpt-4o",
        temperature: float = 0.7
    ) -> Any:
        """Generate structured data using Pydantic model (requires instructor or similar, but here we use JSON mode + parsing)"""
        # Note: In a real prod app, I'd use the `instructor` library or OpenAI's new structured outputs.
        # For now, we'll use JSON mode and manual parsing for simplicity and compatibility.
        
        schema = response_model.model_json_schema()
        json_prompt = f"{prompt}\n\nRespond with a valid JSON object matching this schema:\n{json.dumps(schema, indent=2)}"
        
        response_text = self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            json_mode=True
        )
        
        try:
            data = json.loads(response_text)
            return response_model(**data)
        except Exception as e:
            print(f"Failed to parse JSON or validate model: {e}")
            print(f"Raw response: {response_text}")
            raise

    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Get embedding for text"""
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding
