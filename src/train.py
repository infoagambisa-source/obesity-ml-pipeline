from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.etl import load_obesity_ucimlrepo, split_features_target
from src.preprocessing import get_feature_lists, build_preprocessor


@dataclass(frozen=True)
class TrainConfig:
    dataset_id: int = 544
    target_col: str = "NObeyesdad"
    random_state: int = 42
    test_size: float = 0.2
    val_size_within_train: float = 0.25  # 0.25 * 0.8 = 0.2 total

    n_iter: int = 15
    cv: int = 5
    scoring: str = "f1_macro"

    model_dir: str = "models"
    model_filename: str = "model.joblib"


def main():
    cfg = TrainConfig()

    # Load data
    df = load_obesity_ucimlrepo(dataset_id=cfg.dataset_id)
    X, y = split_features_target(df, target_col=cfg.target_col)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y
    )

    # Stratified train/val split (from training portion)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=cfg.val_size_within_train,
        random_state=cfg.random_state,
        stratify=y_train
    )

    # For preprocessing
    num_cols, cat_cols = get_feature_lists(X_train)
    preprocess = build_preprocessor(num_cols, cat_cols)

    # Pipeline
    gb_pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", GradientBoostingClassifier(random_state=cfg.random_state))
    ])

    # Hyperparameter search space (same as notebook)
    param_dist = {
        "clf__n_estimators": [100, 150, 200, 300],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth": [2, 3, 4],
        "clf__subsample": [0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=gb_pipeline,
        param_distributions=param_dist,
        n_iter=cfg.n_iter,
        scoring=cfg.scoring,
        cv=cfg.cv,
        random_state=cfg.random_state,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Validation evaluation 
    y_val_pred = best_model.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")
    val_acc = accuracy_score(y_val, y_val_pred)

    # Test evaluation (kept for logging, but test remains untouched by tuning)
    y_test_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    test_acc = accuracy_score(y_test, y_test_pred)

    print("Best parameters:", search.best_params_)
    print("Best CV Macro F1:", round(search.best_score_, 4))
    print("Validation Macro F1:", round(val_f1, 4), "| Validation Accuracy:", round(val_acc, 4))
    print("Test Macro F1:", round(test_f1, 4), "| Test Accuracy:", round(test_acc, 4))

    # Save model
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / cfg.model_filename

    joblib.dump(best_model, out_path)
    print(f"Saved trained pipeline to: {out_path}")


if __name__ == "__main__":
    main()
