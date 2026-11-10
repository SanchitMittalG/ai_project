from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import FEATURES, TARGET


def validate_features(frame, allow_missing=True):
    if set(frame.columns) != set(FEATURES) or frame.columns.duplicated().any():
        raise ValueError(f"Exactly these features are required: {FEATURES}")
    numeric = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if numeric.empty or np.isinf(numeric.to_numpy()).any():
        raise ValueError("Feature rows must be nonempty and cannot contain infinity")
    if not allow_missing and numeric.isna().any().any():
        raise ValueError("Prediction inputs must contain seven finite numeric values")
    for feature in ("N", "P", "K", "rainfall"):
        if (numeric[feature] < 0).any():
            raise ValueError(f"{feature} cannot be negative")
    for feature, upper in (("humidity", 100), ("ph", 14)):
        if ((numeric[feature] < 0) | (numeric[feature] > upper)).any():
            raise ValueError(f"{feature} must be in [0, {upper}]")
    return numeric


def load_dataset(path):
    frame = pd.read_csv(Path(path))
    if set(frame.columns) != {*FEATURES, TARGET}:
        raise ValueError("Dataset must contain exactly seven features and label")
    features = validate_features(frame.drop(columns=TARGET))
    if frame[TARGET].isna().any():
        raise ValueError("Crop labels cannot be missing")
    labels = frame[TARGET].astype(str).str.strip()
    if (labels == "").any() or labels.nunique() < 2:
        raise ValueError("At least two nonempty crop labels are required")
    clean = features.assign(label=labels)
    duplicates = int(clean.duplicated().sum())
    clean = clean.drop_duplicates().reset_index(names="source_row")
    if clean.duplicated(subset=list(FEATURES), keep=False).any():
        raise ValueError("Identical input rows have conflicting crop labels")
    if clean[list(FEATURES)].isna().all().any():
        raise ValueError("No input feature may be entirely missing")
    return clean, duplicates


def split_dataset(frame, seed=42):
    training, remainder = train_test_split(
        frame.index.to_numpy(), test_size=0.30, stratify=frame[TARGET], random_state=seed
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, stratify=frame.loc[remainder, TARGET], random_state=seed
    )
    return {"train": training, "validation": validation, "test": test}
