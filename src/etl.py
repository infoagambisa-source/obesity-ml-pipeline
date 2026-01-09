from __future__ import annotations

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_obesity_ucimlrepo(dataset_id: int = 544) -> pd.DataFrame:
    """
    Fetch the obesity dataset from UCI via ucimlrepo and return a single DataFrame
    containing both features and the target column `NObeyesdad`.

    Returns
    -------
    pd.DataFrame
        DataFrame with 16 feature columns + target column `NObeyesdad`.
    """
    obesity = fetch_ucirepo(id=dataset_id)
    X = obesity.data.features
    y = obesity.data.targets

    df = X.copy()
    # Targets come as a 1-col DataFrame; keep the column name consistent with UCI docs
    df["NObeyesdad"] = y.iloc[:, 0]
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "NObeyesdad"):
    """
    Split a DataFrame into features X and target y.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame columns.")

    X = df.drop(columns=[target_col])
    y = df[target_col]  # Series for sklearn compatibility
    return X, y
