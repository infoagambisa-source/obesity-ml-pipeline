from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.etl import load_obesity_ucimlrepo, split_features_target


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved obesity model on a fresh split.")
    parser.add_argument("--model", default="models/model.joblib", help="Path to saved joblib model.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for splitting.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    # Load full dataset
    df = load_obesity_ucimlrepo()
    X, y = split_features_target(df)

    # Fresh stratified split (for evaluation reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # Evaluate
    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)

    print("Evaluation results")
    print("------------------")
    print("Test Macro F1:", round(macro_f1, 4))
    print("Test Accuracy:", round(acc, 4))

    # Confusion matrix
    plt.figure(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=90, cmap="Blues")
    plt.title("Confusion Matrix — Saved Model")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
