import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    log_loss, mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score,
)


def save_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Cannot serialize {type(item)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding="utf-8")


def metrics(labels, predictions, probabilities=None, classes=None):
    scores = {"accuracy": accuracy_score(labels, predictions)}
    for average in ("macro", "weighted"):
        for name, function in (("precision", precision_score), ("recall", recall_score), ("f1", f1_score)):
            scores[f"{name}_{average}"] = function(labels, predictions, average=average, zero_division=0)
    if probabilities is not None:
        scores["log_loss"] = log_loss(labels, probabilities, labels=classes)
    return scores


def label_diagnostics(labels, predictions):
    return {
        "label_code_rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
        "label_code_mae": mean_absolute_error(labels, predictions),
        "label_code_r2": r2_score(labels, predictions),
        "warning": "Arbitrary alphabetical class codes; not yield errors or meaningful crop distances.",
    }


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def export_evaluation(model, features, labels, class_names, directory):
    directory.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(features)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    scores = metrics(labels, predictions, probabilities, model.classes_)
    save_json(directory / "metrics.json", scores)
    save_json(directory / "label_code_diagnostics.json", label_diagnostics(labels, predictions))
    report = classification_report(
        labels, predictions, labels=np.arange(len(class_names)), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    save_json(directory / "classification_report.json", report)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(class_names)))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(directory / "confusion_matrix.csv")
    plt.figure(figsize=(11, 10))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted crop")
    plt.ylabel("True crop")
    plt.colorbar(label="Samples")
    save_figure(directory / "confusion_matrix.png")
    rows = pd.DataFrame({
        "row_index": features.index, "true_crop": np.asarray(class_names)[labels],
        "predicted_crop": np.asarray(class_names)[predictions],
        "confidence": probabilities.max(axis=1),
    })
    for class_index, class_name in enumerate(class_names):
        rows[f"probability_{class_name}"] = probabilities[:, class_index]
    rows.to_csv(directory / "predictions.csv", index=False)
    return scores


def export_eda(training, directory):
    directory.mkdir(parents=True, exist_ok=True)
    numeric = training.drop(columns=["label", "source_row"])
    correlation = numeric.corr()
    correlation.to_csv(directory / "pearson_correlation.csv")
    plt.figure(figsize=(8, 7))
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(numeric.columns)), numeric.columns, rotation=45)
    plt.yticks(range(len(numeric.columns)), numeric.columns)
    plt.colorbar(label="Pearson correlation (training only)")
    save_figure(directory / "pearson_correlation.png")
    numeric.hist(figsize=(12, 8), bins=20)
    save_figure(directory / "feature_distributions.png")
    training["label"].value_counts().sort_index().plot.bar(figsize=(12, 5))
    plt.ylabel("Training samples")
    save_figure(directory / "crop_distribution.png")
    zscores = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, np.nan)
    save_json(directory / "diagnostics.json", {
        "missing_training_values": numeric.isna().sum().to_dict(),
        "training_zscore_abs_gt_3_counts": (zscores.abs() > 3).sum().to_dict(),
        "note": "Diagnostics only; no rows removed. All fitted preprocessing is fold-local.",
    })
