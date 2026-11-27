
from src.ml.intent_classifier import IntentClassifier
import torch


def main():
    classifier = IntentClassifier()

    messages = [
        "I want to cancel my order.",
        "Please cancel my purchase immediately.",
        "I need to cancel an order I placed yesterday.",
        "Where is my package?",
        "I want a refund for my purchase.",
    ]

    for message in messages:
        inputs = classifier.tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=96,
        )

        inputs = {
            key: value.to(classifier.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = classifier.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]

        top_values, top_indices = torch.topk(probabilities, 3)

        print("\nMessage:", message)
        print("Top predictions:")

        for score, index in zip(top_values, top_indices):
            print(
                f"{classifier.labels[index.item()]}: "
                f"{score.item():.4f}"
            )


if __name__ == "__main__":
    main()
