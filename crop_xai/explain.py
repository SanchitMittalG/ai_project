from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import save_figure, save_json


def explain_model(bundle, training, samples, output, seed=42, sample_count=12, lime_samples=2500,
                  sample_role="held-out test"):
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = bundle["model"]
    feature_names = bundle["features"]
    background = training.sample(min(32, len(training)), random_state=seed)
    background = pd.DataFrame(model.named_steps["imputer"].transform(background), columns=feature_names)
    selected = samples.sample(min(sample_count, len(samples)), random_state=seed)
    selected = pd.DataFrame(model.named_steps["imputer"].transform(selected),
                            columns=feature_names, index=selected.index)

    def predict(values):
        return model.predict_proba(pd.DataFrame(values, columns=feature_names))

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        explainer = shap.KernelExplainer(predict, background.to_numpy())
        raw_values = explainer.shap_values(selected.to_numpy(), nsamples=128, l1_reg=0.0, silent=True)
    finally:
        np.random.set_state(state)
    values = np.stack(raw_values, axis=-1) if isinstance(raw_values, list) else np.asarray(raw_values)
    expected_shape = (len(selected), len(feature_names), len(bundle["class_names"]))
    if values.shape != expected_shape:
        raise ValueError(f"Unexpected multiclass SHAP shape {values.shape}; expected {expected_shape}")
    probabilities = predict(selected.to_numpy())
    predicted_classes = probabilities.argmax(axis=1)
    predicted_values = np.stack([values[row, :, class_index] for row, class_index in enumerate(predicted_classes)])
    np.savez_compressed(output / "shap_values.npz", values=values,
                        expected_values=explainer.expected_value, row_indices=selected.index.to_numpy(),
                        feature_names=np.asarray(feature_names), class_names=np.asarray(bundle["class_names"]))
    importance = pd.Series(np.abs(values).mean(axis=(0, 2)), index=feature_names).sort_values(ascending=False)
    importance.rename("mean_absolute_shap_all_classes").to_csv(output / "shap_global.csv")
    importance.sort_values().plot.barh(figsize=(8, 5))
    plt.xlabel("Mean absolute SHAP contribution (all classes, sampled rows)")
    save_figure(output / "shap_global.png")
    shap.summary_plot(predicted_values, selected, show=False)
    plt.xlabel("SHAP contribution to each row's predicted class")
    save_figure(output / "shap_beeswarm.png")
    local_records = []
    for row, class_index in enumerate(predicted_classes):
        base = float(np.asarray(explainer.expected_value)[class_index])
        contributions = predicted_values[row]
        local_records.append({
            "row_index": int(selected.index[row]), "crop": bundle["class_names"][class_index],
            "inputs": selected.iloc[row].to_dict(), "probability": float(probabilities[row, class_index]),
            "base_value": base, "contributions": dict(zip(feature_names, contributions.tolist())),
            "additivity_residual": float(probabilities[row, class_index] - base - contributions.sum()),
        })
    save_json(output / "shap_local.json", local_records)
    lime_training = pd.DataFrame(model.named_steps["imputer"].transform(training), columns=feature_names)
    lime = LimeTabularExplainer(
        lime_training.to_numpy(), feature_names=feature_names, class_names=bundle["class_names"],
        mode="classification", discretize_continuous=True, random_state=seed,
    )
    class_index = int(predicted_classes[0])
    explanation = lime.explain_instance(selected.iloc[0].to_numpy(), predict, labels=(class_index,),
                                        num_features=len(feature_names), num_samples=lime_samples)
    explanation.save_to_file(str(output / "lime_local.html"))
    save_json(output / "lime_local.json", {
        "row_index": int(selected.index[0]), "crop": bundle["class_names"][class_index],
        "inputs": selected.iloc[0].to_dict(), "probability": float(probabilities[0, class_index]),
        "intercept": float(explanation.intercept[class_index]),
        "surrogate_r2": float(explanation.score), "surrogate_prediction": float(explanation.local_pred[0]),
        "conditions": [{"condition": condition, "weight": weight}
                       for condition, weight in explanation.as_list(label=class_index)],
        "num_perturbations": lime_samples,
        "warning": "Local surrogate weights are not causal effects; inspect surrogate fidelity.",
    })
    explanation.as_pyplot_figure(label=class_index)
    save_figure(output / "lime_local.png")
    save_json(output / "method.json", {
        "shap_method": "KernelExplainer on raw units, identity link, training-only background",
        "background_rows": len(background), "rows_explained": len(selected), "sample_role": sample_role,
        "shap_nsamples": 128, "lime_training_rows": len(training), "seed": seed,
        "global_aggregation": "Mean absolute SHAP over sampled rows and all output classes",
        "beeswarm": "Each row's predicted-class contributions; class may vary by row",
        "warning": "Sampled model behavior, not causal explanations or validated agronomic advice.",
    })
