
import os
from openai import OpenAI


class LLMClient:
    """Shared client for the local Docker Model Runner."""

    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv(
                "LLM_BASE_URL",
                "http://localhost:12434/engines/v1"
            ),
            api_key=os.getenv("LLM_API_KEY", "local"),
        )

        self.model = os.getenv(
            "LLM_MODEL",
            "ai/qwen2.5:3B-Q4_K_M"
        )

    def generate(
        self,
        prompt: str,
        system_message: str = "You are a helpful customer support assistant.",
        temperature: float = 0.3,
        max_tokens: int = 300,
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""
