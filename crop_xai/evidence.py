import hashlib
import json

import pandas as pd

from .evaluation import save_json


def write_evidence(output):
    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    table = pd.read_csv(output / "test_comparison.csv")
    selected = table[table["selected_before_test"]].iloc[0]
    text = [
        "# Reproduction run evidence", "",
        f"- Mode: {metadata['mode']} (smoke results must not be called final experiments).",
        f"- Source commit: {metadata['source_commit']}; dirty source: {metadata['source_dirty']}.",
        f"- Dataset SHA256: {metadata['data_sha256']}.",
        f"- Split sizes: {metadata['split_sizes']}; CV folds: {metadata['folds']}.",
        f"- Selected before test: {metadata['selected_model']}.",
        f"- Selection rule: {metadata['selection_rule']}.",
        f"- Held-out accuracy: {selected['accuracy']:.6f}; macro F1: {selected['f1_macro']:.6f}.",
        "", "## Authoritative evidence", "",
        "- run.json: status, versions, seeds, grids, source and dataset provenance.",
        "- splits.json and cv_splits.json: exact disjoint partitions and CV row indices.",
        "- cv_fold_metrics.csv: training-only baseline fold scores.",
        "- search/: all tuning candidates, fold scores and selected parameters.",
        "- validation_comparison.csv: model selection evidence.",
        "- test_comparison.csv and evaluation/: untouched-test classification results.",
        "- ablation.csv: descriptive held-out comparisons, not used for model selection.",
        "- eda/: training-only correlations, distributions and diagnostics.",
        "- explanations/: SHAP values, class alignment, LIME fidelity and figures (if requested).",
        "", "## Interpretation limits", "",
        "This is crop classification, not historical yield prediction. No yield or time columns exist.",
        "The paper does not publish enough settings for exact numerical replication.",
        "Encoded-label RMSE, MAE and R2 are arbitrary-order compatibility diagnostics only.",
        "XAI contributions and ablations are associative model diagnostics, not causal agronomy.",
        "No geographic/temporal generalization, calibrated probabilities or cloud deployment is established.",
        "The original exploratory notebook results are not this run's results.",
    ]
    (output / "REPORT_EVIDENCE.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    files = {
        path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"
    }
    save_json(output / "manifest.json", {"algorithm": "sha256", "files": files})
