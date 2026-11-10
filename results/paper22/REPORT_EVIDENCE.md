# Reproduction run evidence

- Mode: full (smoke results must not be called final experiments).
- Source commit: 080177d755432fcbb3c56985c1d76ee228f2ab2c; dirty source: False.
- Dataset SHA256: c602b137c8ad697fcbfc8b77007e0fd25d458e583173bae37caab3fe3e04dc68.
- Split sizes: {'train': 1540, 'validation': 330, 'test': 330}; CV folds: 5.
- Selected before test: grid_random_forest.
- Selection rule: Validation macro F1, then training CV macro F1, then model name; never test scores.
- Held-out accuracy: 0.996970; macro F1: 0.996966.

## Authoritative evidence

- run.json: status, versions, seeds, grids, source and dataset provenance.
- splits.json and cv_splits.json: exact disjoint partitions and CV row indices.
- cv_fold_metrics.csv: training-only baseline fold scores.
- search/: all tuning candidates, fold scores and selected parameters.
- validation_comparison.csv: model selection evidence.
- test_comparison.csv and evaluation/: untouched-test classification results.
- ablation.csv: descriptive held-out comparisons, not used for model selection.
- eda/: training-only correlations, distributions and diagnostics.
- explanations/: SHAP values, class alignment, LIME fidelity and figures (if requested).

## Interpretation limits

This is crop classification, not historical yield prediction. No yield or time columns exist.
The paper does not publish enough settings for exact numerical replication.
Encoded-label RMSE, MAE and R2 are arbitrary-order compatibility diagnostics only.
XAI contributions and ablations are associative model diagnostics, not causal agronomy.
No geographic/temporal generalization, calibrated probabilities or cloud deployment is established.
The original exploratory notebook results are not this run's results.
