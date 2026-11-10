# Prompt to give Claude after completing an experiment

Copy the prompt below. Attach the reference PDF and the repository, plus the
complete experiment directory (or its evidence, CSV/JSON files and figures).
Replace RUN_DIRECTORY with the actual full, completed run path. Do not give
Claude only the old notebook or README and ask it to invent results.

---

You are my research-report writing assistant. Produce a rigorous final-year
project report for **Explainable Crop Recommendation Using Soil Nutrient and
Climate Profiles**, based on this repository and the attached reference paper:
Surendra Kumar and Mohit Kumar, *Developing an XAI-Based Crop Recommendation
Framework Using Soil Nutrient Profiles and Historical Crop Yields*, IEEE
Transactions on Consumer Electronics, Vol. 71, No. 2, May 2025. Verify remaining
bibliographic details in the attached PDF before citing them.

Team: Aadi Singhal (202352301), Piyush Jain (202352324), Sanchit Mittal
(202352328). Leave institution, supervisor and submission date as clearly marked
placeholders unless I provide them.

For the completed experiment in this repository, set `RUN_DIRECTORY` to
`results/paper22` (absolute path in the original checkout:
`E:/aiiii/ai_project/results/paper22`). The fitted binary is intentionally not
included in that evidence snapshot and is not needed to write the report.

## Inputs you must inspect before drafting

1. The attached PDF, particularly Sections III-V, Algorithm 1 and Tables II-III.
2. `docs/PAPER_IMPLEMENTATION.md`, `README.md`, `docs/VALIDATION.md` and the
   actual `crop_xai/` implementation, not just saved legacy notebook outputs.
3. RUN_DIRECTORY/`run.json` and `REPORT_EVIDENCE.md`: verify status is complete,
   mode is full, cohort is identified, and folds/seed/versions/hash are known.
4. `splits.json`, `cv_splits.json`, `cv_fold_metrics.csv`, `search/*_best.json`,
   candidate search CSVs and `validation_comparison.csv`.
5. `test_comparison.csv`, `evaluation/*/metrics.json`, per-class reports,
   confusion matrices, row-level predictions and `ablation.csv`.
6. `explanations/method.json`, `shap_global.csv`, `shap_local.json`,
   `lime_local.json`, saved explanation figures and training-only EDA artifacts.
7. `manifest.json` for artifact integrity when available. Avoid opening/loading
   a joblib model unless it is trusted and necessary; use JSON/CSV evidence first.

If any evidence is missing, list it explicitly. Do not manufacture measurements,
cloud deployments, agronomic field validation, screenshots, timings, significance,
citations, crop-success probabilities or claims of author-code equivalence.
If you cannot read a file, ask for that file rather than guess its contents.

## Scientific integrity requirements

- The repository CSV has 6,596 rows and 51 classes. The default paper22 cohort
  selects 2,200 rows and 22 canonical labels. Keep these populations separate.
  It is a reconstructed cohort, not proven identical author-data provenance.
- There is no historical-yield target, temporal index, location or farm grouping.
  Our task is multiclass crop recommendation, not yield forecasting. Discuss the
  title's historical-yield component as a data limitation, not an implementation.
- Report stratified 70/15/15 splitting as our explicit interpretation of the
  paper's unspecified 70/30 train/remainder protocol. Five-fold CV occurs inside
  training only. Imputation and scaling are fitted within each fold.
- Explain GridSearchCV as exhaustive enumeration, not Bayesian optimization or
  learning from previous evaluations. List actual grids and selection metrics.
- The selected model is chosen by validation macro F1, then training CV macro F1,
  then model name. Its test score is not used for selection. The final stored
  model is not retrained on the test set. Distinguish baseline and tuned variants;
  do not assume the selected model is RF or a tuned model without evidence.
- Explain balanced paper22 versus the imbalanced all-class extension. SMOTE and
  the legacy notebook's engineered features are not part of this pipeline.
- Use measured accuracy, macro/weighted precision, recall, F1 and log loss with
  correct sample counts and percentage conversions. Do not silently replace
  macro by weighted scores. Training CV is not nested-CV generalization evidence;
  best-grid CV estimates are optimistic after tuning, unlike held-out evaluation.
- Class-code RMSE, MAE and R2 depend on alphabetical label ordering. If required
  for paper comparison, isolate them in a clearly warned compatibility appendix,
  never call them yield errors, percentage accuracy or physical crop distances.
- The paper's 99.73% accuracy and other quoted claims are reference values, not
  our results. Different splits/cohorts/settings prevent exact comparability.
- SHAP is model-agnostic KernelExplainer in raw feature units with a training-only
  background. Global mean absolute importance averages sampled rows and output
  classes. The beeswarm combines each sampled row's predicted-class attribution;
  do not mislabel it a single fixed-class or all-test-set explanation.
- LIME targets the actual predicted class. State row, crop, probability, number
  of perturbations, conditions, signed weights and surrogate R2. Low fidelity is
  a limitation, not proof of a faithful explanation. Do not invent simple
  agronomic rules from signed weights or conflate LIME with SHAP scales.
- Ablation freezes the selected algorithm/hyperparameters and refits after
  removing one feature using identical splits. Report observed accuracy drops,
  including negative drops. This is descriptive association, not causal nutrient
  importance, and is not used to select a new winning configuration.
- Model scores are uncalibrated. The abstention threshold is a configurable
  engineering choice, not a validated agricultural safety boundary. Range
  warnings do not prove out-of-distribution detection.
- Optional GCS export is backup/storage integration, not hosted inference or
  verified cloud deployment. Claim actual uploads only with execution evidence.
- Historical yields, temporal/geographic validation, causal nutrient adjustments,
  calibrated uncertainty and external field trials remain future work.

## Report structure

Write an original, coherent report, not copied passages from the paper:

1. Title page placeholders, abstract (200-300 words) and keywords.
2. Introduction, agricultural motivation, problem definition and objectives.
3. Related work: only verified sources; separate the reference method from our
   reproducible implementation. Include a comparison table grounded in sources.
4. Dataset and provenance: source audit, feature dictionary with unit uncertainty,
   cohort reconstruction, class distribution, missingness, duplicates and ethics.
5. Methodology: architecture/data-flow description, preprocessing, splits, CV,
   baselines, grids, validation selection, saved model and mathematical metrics.
6. Explainability: SHAP and LIME mechanics, class alignment, reference data,
   approximation limits, local fidelity and interpretation cautions.
7. Implementation: module responsibilities, reproducible commands, dependencies,
   provenance manifests, CLI inference, thresholding/logging and optional cloud.
8. Experiments and results: training-fold mean/std, validation selection, held-out
   baseline/tuned table, per-class failure analysis, confusion and confidence
   plots, XAI example and feature ablation. Read actual files for every number.
9. Paper comparison: table of paper claims, our observations, protocol differences
   and implemented/unsupported components, with no forced accuracy matching.
10. Discussion, threats to validity, limitations, conclusion and future scope.
11. Verified references and appendices: full grids, environment/seed/data hash,
    exact run commands, artifact index and compatibility-only numeric diagnostics.

Provide numbered figure/table captions tied to real artifact filenames. Insert
explicit placeholders when images cannot be embedded. Maintain a claim-to-evidence
table mapping every numerical claim to an artifact and column/key. End with a
short viva-preparation Q&A covering leakage, grid search, dataset mismatch,
macro/weighted metrics, SHAP/LIME fidelity and why this is not yield forecasting.

First return a concise evidence audit and proposed outline. Then draft the full
report once evidence is sufficient. If institution formatting or page limits are
unknown, use a conventional academic structure without inventing requirements.
