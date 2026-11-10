import hashlib
import importlib.metadata
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder

from . import FEATURES, PAPER_CROPS, TARGET
from .data import load_dataset, split_dataset
from .evaluation import export_eda, export_evaluation, metrics, save_figure, save_json
from .models import make_pipeline, model_definitions


def run_experiment(data_path, output, seed=42, folds=5, jobs=1, quick=False, explain=True,
                   shap_samples=12, lime_samples=2500, cohort="paper22"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a new run name to avoid stale evidence")
    if folds < 2 or jobs == 0 or shap_samples < 1 or lime_samples < 100:
        raise ValueError("Require folds >= 2, jobs != 0, shap_samples >= 1, lime_samples >= 100")
    frame, duplicates = load_dataset(data_path)
    available_rows = len(frame)
    available_classes = frame[TARGET].nunique()
    if cohort == "paper22":
        if not set(PAPER_CROPS).issubset(set(frame[TARGET])):
            raise ValueError("The paper22 cohort requires all 22 canonical crop labels")
        frame = frame[frame[TARGET].isin(PAPER_CROPS)].reset_index(drop=True)
    elif cohort != "all":
        raise ValueError("Cohort must be paper22 or all")
    print(f"Cohort {cohort}: {len(frame)} rows, {frame[TARGET].nunique()} crops "
          f"(source: {available_rows} rows, {available_classes} crops)", flush=True)
    partitions = split_dataset(frame, seed)
    features = frame.loc[:, FEATURES]
    encoder = LabelEncoder().fit(frame[TARGET])
    labels = encoder.transform(frame[TARGET])
    training = partitions["train"]
    validation = partitions["validation"]
    testing = partitions["test"]
    if frame.loc[training, TARGET].value_counts().min() < folds:
        raise ValueError("Every training class needs at least one sample per CV fold")
    if features.loc[training].isna().all().any():
        raise ValueError("Training split contains an entirely missing input feature")
    output.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    save_json(output / "splits.json", {
        name: {"row_indices": indices, "source_csv_rows": frame.loc[indices, "source_row"].tolist()}
        for name, indices in partitions.items()
    })
    save_json(output / "cv_splits.json", [
        {"fold": fold + 1, "train_rows": training[train_rows], "validation_rows": training[val_rows]}
        for fold, (train_rows, val_rows) in enumerate(cv.split(features.loc[training], labels[training]))
    ])
    export_eda(frame.loc[training], output / "eda")
    models, grids = model_definitions(seed, quick)
    candidates = {}
    summaries = []
    fold_records = []
    started = time.perf_counter()
    scoring = {"accuracy": "accuracy",
               "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
               "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
               "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
               "neg_label_code_rmse": "neg_root_mean_squared_error"}
    for name, estimator in models.items():
        print(f"Training and cross-validating {name}", flush=True)
        pipeline = make_pipeline(estimator)
        baseline_cv = cross_validate(
            pipeline, features.loc[training], labels[training], cv=cv,
            scoring=scoring, n_jobs=jobs, error_score="raise",
        )
        pipeline.fit(features.loc[training], labels[training])
        candidates[name] = pipeline
        validation_scores = metrics(labels[validation], pipeline.predict(features.loc[validation]))
        summaries.append({"model": name, "kind": "baseline", **validation_scores,
                          "cv_accuracy_mean": baseline_cv["test_accuracy"].mean(),
                          "cv_accuracy_std": baseline_cv["test_accuracy"].std(),
                          "cv_f1_macro_mean": baseline_cv["test_f1_macro"].mean()})
        for fold in range(folds):
            fold_records.append({"model": name, "fold": fold + 1,
                                 **{metric: baseline_cv[f"test_{metric}"][fold] for metric in scoring}})
        search = GridSearchCV(
            make_pipeline(clone(estimator)),
            {f"classifier__{key}": value for key, value in grids[name].items()},
            scoring=scoring, cv=cv, n_jobs=jobs, refit="f1_macro", error_score="raise",
            return_train_score=True,
        ).fit(features.loc[training], labels[training])
        search_dir = output / "search"
        search_dir.mkdir(exist_ok=True)
        pd.DataFrame(search.cv_results_).to_csv(search_dir / f"{name}.csv", index=False)
        tuned_name = f"grid_{name}"
        candidates[tuned_name] = search.best_estimator_
        validation_scores = metrics(labels[validation], search.predict(features.loc[validation]))
        summaries.append({"model": tuned_name, "kind": "grid_search", **validation_scores,
                          "cv_accuracy_mean": search.cv_results_["mean_test_accuracy"][search.best_index_],
                          "cv_accuracy_std": search.cv_results_["std_test_accuracy"][search.best_index_],
                          "cv_f1_macro_mean": search.best_score_})
        for fold in range(folds):
            fold_records.append({"model": tuned_name, "fold": fold + 1,
                                 **{metric: search.cv_results_[f"split{fold}_test_{metric}"][search.best_index_]
                                    for metric in scoring}})
        save_json(search_dir / f"{name}_best.json", {
            "parameters": search.best_params_, "cv_f1_macro": search.best_score_,
            "grid": grids[name], "candidates": len(search.cv_results_["params"]),
        })
    comparison = pd.DataFrame(summaries).sort_values(
        ["f1_macro", "cv_f1_macro_mean", "model"], ascending=[False, False, True]
    )
    comparison.to_csv(output / "validation_comparison.csv", index=False)
    pd.DataFrame(fold_records).to_csv(output / "cv_fold_metrics.csv", index=False)
    selected_name = comparison.iloc[0]["model"]
    selected = candidates[selected_name]
    print(f"Selected using validation macro F1: {selected_name}", flush=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "folds": folds,
        "mode": "smoke" if quick else "full", "selected_model": selected_name,
        "selection_rule": "Validation macro F1, then training CV macro F1, then model name; never test scores",
        "split_sizes": {name: len(indices) for name, indices in partitions.items()},
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "data_rows": len(frame), "removed_exact_duplicates": duplicates,
        "cohort": cohort, "source_data_rows": available_rows, "source_data_classes": available_classes,
        "class_counts": frame[TARGET].value_counts().sort_index().to_dict(),
        "classes": encoder.classes_.tolist(), "features": list(FEATURES),
        "grids": grids, "historical_yield_available": False,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "shap", "lime", "matplotlib", "joblib")},
        "explanations_requested": explain, "status": "running",
    }
    try:
        metadata["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
        ).strip()
        metadata["source_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, cwd=Path(__file__).resolve().parents[1]
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        metadata["source_commit"] = None
        metadata["source_dirty"] = None
    save_json(output / "run.json", metadata)
    test_records = []
    for name, model in candidates.items():
        scores = export_evaluation(model, features.loc[testing], labels[testing],
                                   encoder.classes_, output / "evaluation" / name)
        test_records.append({"model": name, "selected_before_test": name == selected_name, **scores})
    test_table = pd.DataFrame(test_records)
    test_table.to_csv(output / "test_comparison.csv", index=False)
    plot_columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    test_table.set_index("model")[plot_columns].plot.bar(figsize=(14, 6), ylim=(0, 1.03))
    save_figure(output / "model_comparison.png")
    bundle = {"model": selected, "class_names": encoder.classes_.tolist(),
              "features": list(FEATURES), "metadata": metadata,
              "training_min": features.loc[training].min().to_dict(),
              "training_max": features.loc[training].max().to_dict(),
              "training_reference": features.loc[training].copy()}
    joblib.dump(bundle, output / "model.joblib")
    ablations = []
    full_score = metrics(labels[testing], selected.predict(features.loc[testing]))
    import matplotlib.pyplot as plt
    confidence = selected.predict_proba(features.loc[testing]).max(axis=1)
    plt.figure(figsize=(8, 5))
    plt.hist(confidence, bins=20, range=(0, 1), edgecolor="black")
    plt.xlabel("Maximum uncalibrated class probability")
    plt.ylabel("Test samples")
    save_figure(output / "prediction_confidence.png")
    for removed in [None, *FEATURES]:
        remaining = [feature for feature in FEATURES if feature != removed]
        if removed is None:
            scores = full_score
        else:
            reduced = clone(selected).fit(features.loc[training, remaining], labels[training])
            scores = metrics(labels[testing], reduced.predict(features.loc[testing, remaining]))
        ablations.append({"removed_feature": removed or "none", **scores,
                          "accuracy_drop": full_score["accuracy"] - scores["accuracy"]})
    pd.DataFrame(ablations).to_csv(output / "ablation.csv", index=False)
    pd.DataFrame(ablations).set_index("removed_feature")["accuracy"].plot.bar(figsize=(9, 5), ylim=(0, 1.03))
    save_figure(output / "ablation.png")
    if explain:
        from .explain import explain_model
        explain_model(bundle, features.loc[training], features.loc[testing], output / "explanations",
                      seed, shap_samples, lime_samples)
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["status"] = "complete"
    save_json(output / "run.json", metadata)
    joblib.dump(bundle, output / "model.joblib")
    from .evidence import write_evidence
    write_evidence(output)
    print(f"Completed. Evidence: {output.resolve()}", flush=True)
    return bundle
