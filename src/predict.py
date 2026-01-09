from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run inference using a saved obesity model pipeline.")
    parser.add_argument("--model", default="models/model.joblib", help="Path to saved joblib model.")
    parser.add_argument("--input", required=True, help="Path to input CSV containing feature columns.")
    parser.add_argument("--output", default="predictions.csv", help="Path to save predictions CSV.")
    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    model = joblib.load(model_path)
    X_new = pd.read_csv(input_path)

    # Predict class labels
    preds = model.predict(X_new)

    out = X_new.copy()
    out["prediction"] = preds
    out.to_csv(output_path, index=False)

    print(f"Loaded model: {model_path}")
    print(f"Read input: {input_path} | rows: {len(X_new)}")
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
