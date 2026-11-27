
import json
from typing import Any

from src.llm_client import LLMClient


class TriageAgent:
    """Identifies customer intent and classifies support requests."""

    ALLOWED_PRIORITIES = {"low", "medium", "high"}
    ALLOWED_CATEGORIES = {
        "ORDER",
        "REFUND",
        "DELIVERY",
        "PRODUCT",
        "OTHER",
    }

    def __init__(self):
        self.llm = LLMClient()

    def classify(self, message: str) -> dict[str, Any]:
        if not message or not message.strip():
            raise ValueError("Customer message cannot be empty.")

        prompt = f"""
Classify this customer support message.

Return ONLY a valid JSON object with these fields:
- intent: short snake_case intent
- category: one of ORDER, REFUND, DELIVERY, PRODUCT, OTHER
- priority: one of low, medium, high
- requires_human: true or false

Customer message:
{message}

JSON:
"""

        result = self.llm.generate(
            prompt=prompt,
            system_message=(
                "You are the Triage Agent in a customer support system. "
                "Classify customer requests accurately. "
                "Return only valid JSON."
            ),
            temperature=0,
            max_tokens=150,
        )

        try:
            data = json.loads(result)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Triage Agent returned invalid JSON: {result}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError("Triage output must be a JSON object.")

        required_fields = {
            "intent",
            "category",
            "priority",
            "requires_human",
        }

        if not required_fields.issubset(data):
            raise ValueError("Triage output is missing required fields.")

        if data["category"] not in self.ALLOWED_CATEGORIES:
            raise ValueError("Invalid category returned by Triage Agent.")

        if data["priority"] not in self.ALLOWED_PRIORITIES:
            raise ValueError("Invalid priority returned by Triage Agent.")

        if not isinstance(data["requires_human"], bool):
            raise ValueError("requires_human must be true or false.")

        return data
