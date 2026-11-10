import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import validate_features


def recommend(bundle, inputs, threshold=0.5, top_k=3):
    if not 0 <= threshold <= 1 or top_k < 1:
        raise ValueError("Threshold must be in [0, 1] and top_k must be positive")
    frame = validate_features(pd.DataFrame([inputs]), allow_missing=False)
    probabilities = bundle["model"].predict_proba(frame)[0]
    predicted_label = bundle["model"].predict(frame)[0]
    predicted_index = int(np.flatnonzero(bundle["model"].classes_ == predicted_label)[0])
    ranking = np.argsort(-probabilities, kind="stable")[:min(top_k, len(probabilities))]
    warnings = []
    for feature in bundle["features"]:
        value = float(frame.iloc[0][feature])
        if not bundle["training_min"][feature] <= value <= bundle["training_max"][feature]:
            warnings.append(f"{feature} lies outside the observed training range")
    if ranking[0] != predicted_index:
        warnings.append("Estimator decision differs from probability ranking (possible for SVM); explanation targets the estimator decision")
    confidence = float(probabilities[predicted_index])
    accepted = confidence >= threshold
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "inputs": frame.iloc[0].to_dict(),
        "model": bundle["metadata"]["selected_model"], "threshold": threshold,
        "dataset_sha256": bundle["metadata"]["data_sha256"],
        "cohort": bundle["metadata"].get("cohort", "legacy"),
        "status": "recommended" if accepted else "abstained_low_confidence",
        "predicted_crop": bundle["class_names"][predicted_index], "confidence": confidence,
        "recommended_crop": bundle["class_names"][predicted_index] if accepted else None,
        "ranking": [{"crop": bundle["class_names"][index], "probability": float(probabilities[index])}
                    for index in ranking], "warnings": warnings,
        "disclaimer": "Uncalibrated model scores, not yield forecasts or fertilizer prescriptions. Consult an agronomist.",
    }
    return result


def append_prediction(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, allow_nan=False) + "\n")
