
import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-intent"


class IntentClassifier:
    """Loads the fine-tuned DistilBERT model for intent prediction."""

    def __init__(self):
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_DIR}"
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR
        ).to(self.device)

        self.model.eval()

        with open(MODEL_DIR / "labels.json", encoding="utf-8") as file:
            self.labels = json.load(file)

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=96,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)

        confidence, predicted_id = torch.max(probabilities, dim=1)

        return {
            "intent": self.labels[predicted_id.item()],
            "confidence": round(confidence.item(), 4),
        }
