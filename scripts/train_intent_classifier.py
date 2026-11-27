
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/raw/customer_support_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models/distilbert-intent"

CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 96
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


def evaluate(model, loader, device):
    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            actual.extend(labels.cpu().numpy())

    accuracy = accuracy_score(actual, predictions)
    macro_f1 = f1_score(
        actual,
        predictions,
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1, actual, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    set_seed()

    torch.set_num_threads(4)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")
    print(f"Dataset: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df = df[["instruction", "intent"]].dropna()
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()

    df = df[
        (df["instruction"] != "") &
        (df["intent"] != "")
    ]

    # Remove exact duplicate text-label pairs.
    df = df.drop_duplicates(
        subset=["instruction", "intent"]
    ).reset_index(drop=True)

    # Keep only text-label pairs that have one unambiguous label.
    label_counts = df.groupby("instruction")["intent"].nunique()
    conflicting_texts = label_counts[label_counts > 1].index
    df = df[~df["instruction"].isin(conflicting_texts)]

    labels = sorted(df["intent"].unique().tolist())
    label_to_id = {
        label: index for index, label in enumerate(labels)
    }

    df["label_id"] = df["intent"].map(label_to_id)

    print(f"Usable samples: {len(df)}")
    print(f"Number of intents: {len(labels)}")

    train_df, temporary_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label_id"],
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temporary_df["label_id"],
    )

    if args.smoke_test:
        train_df, _ = train_test_split(
            train_df,
            train_size=min(1500, len(train_df)),
            random_state=SEED,
            stratify=train_df["label_id"],
        )
        validation_df, _ = train_test_split(
            validation_df,
            train_size=min(500, len(validation_df)),
            random_state=SEED,
            stratify=validation_df["label_id"],
        )
        test_df, _ = train_test_split(
            test_df,
            train_size=min(500, len(test_df)),
            random_state=SEED,
            stratify=test_df["label_id"],
        )

    print(
        f"Train: {len(train_df)} | "
        f"Validation: {len(validation_df)} | "
        f"Test: {len(test_df)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(labels),
        id2label={i: label for i, label in enumerate(labels)},
        label2id=label_to_id,
    )
    model.to(device)

    train_dataset = IntentDataset(
        train_df["instruction"].tolist(),
        train_df["label_id"].tolist(),
        tokenizer,
    )
    validation_dataset = IntentDataset(
        validation_df["instruction"].tolist(),
        validation_df["label_id"].tolist(),
        tokenizer,
    )
    test_dataset = IntentDataset(
        test_df["instruction"].tolist(),
        test_df["label_id"].tolist(),
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_f1 = -1.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for batch in progress:
            labels_batch = batch.pop("labels").to(device)
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(
                **batch,
                labels=labels_batch,
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(train_loader)

        val_accuracy, val_f1, _, _ = evaluate(
            model,
            validation_loader,
            device,
        )

        print(f"\nEpoch {epoch + 1}")
        print(f"Training loss: {average_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1

            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            with open(
                MODEL_DIR / "labels.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(labels, file, indent=2)

            print("Best model checkpoint saved.")

    print("\nEvaluating best model on test data...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    ).to(device)

    test_accuracy, test_f1, actual, predictions = evaluate(
        model,
        test_loader,
        device,
    )

    report = classification_report(
        actual,
        predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_f1),
        "number_of_intents": len(labels),
        "training_samples": len(train_df),
        "validation_samples": len(validation_df),
        "test_samples": len(test_df),
        "classification_report": report,
    }

    with open(
        MODEL_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\nFinal test results")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {test_f1:.4f}")
    print(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
