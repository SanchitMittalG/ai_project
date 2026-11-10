# Explainable Crop Recommendation

A reproducible implementation of the **measurable crop-classification framework**
in Kumar and Kumar, *Developing an XAI-Based Crop Recommendation Framework Using
Soil Nutrient Profiles and Historical Crop Yields*, IEEE TCE, May 2025.

| Team member | Enrollment |
| --- | --- |
| Aadi Singhal | 202352301 |
| Piyush Jain | 202352324 |
| Sanchit Mittal | 202352328 |

## Important dataset correction

The committed CSV contains **6,596 rows and 51 crop classes**, not the 2,200 rows
and 22 classes previously stated here. The default `paper22` cohort explicitly
selects the 22 canonical crop labels, yielding 2,200 rows (100 per class). Use
`--cohort all` for the separate imbalanced, expanded-data experiment. Source
provenance beyond this repository is unverified; cohort selection does not prove
that these are the authors' exact original samples.

There are seven measured inputs: `N`, `P`, `K`, `temperature`, `humidity`, `ph`,
`rainfall`; the target is `label`. Preserve the CSV's measurement conventions.
Neither nutrient units nor rainfall aggregation period are verified by dataset
metadata. The dataset has **no historical yield or timestamp columns**, so this
project does not claim to predict yield or establish temporal generalization.

## What is implemented

- Strict input checks, exact-duplicate removal and contradictory-row rejection.
- Saved stratified 70/15/15 train/validation/test partitions and five training-only CV folds.
- Fold-local median imputation and min-max normalization; Pearson/EDA and z-score diagnostics.
- Logistic Regression, Decision Tree, Random Forest, KNN, SVM and Gradient Boosting.
- Explicit exhaustive grids for all six algorithms; saved parameters and candidate/fold scores.
- Model selection by **validation macro F1**, with training CV as tie-breaker, never by test accuracy.
- Accuracy, macro/weighted precision, recall and F1, log loss, per-class reports and confusion matrices.
- Separately labelled class-code RMSE/MAE/R2 for paper compatibility, not meaningful yield errors.
- Frozen-configuration leave-one-feature-out ablations, including N, P and K.
- Multiclass SHAP global/local explanations and LIME explanations with surrogate fidelity.
- Trusted-model save/reload, ranked recommendations, confidence-based abstention and range warnings.
- Structured prediction logs and optional create-only authenticated GCS backups.
- Evidence summaries, artifact SHA256 manifests and a grounded Claude report-writing prompt.

See [the paper mapping](docs/PAPER_IMPLEMENTATION.md) for assumptions, deviations,
and unsupported claims. Exact author grids/splits are unavailable. No result is
forced to match the paper's reported 99.73% accuracy.

## Setup (Python 3.12 recommended)

```powershell
git clone https://github.com/SanchitMittalG/ai_project.git
cd ai_project
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-paper.txt
```

On Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`.
The original exploratory notebook has heavier optional dependencies in
`requirements.txt`; those are not needed for the paper pipeline.

## Run the complete experiment

Run from the repository root. Output must be a new or empty directory, so stale
artifacts cannot silently mix with a new run.

```powershell
.venv\Scripts\python.exe -m crop_xai train --output artifacts/paper22 --jobs 2
```

This executes five-fold baseline CV, full grids, validation-based selection,
held-out evaluation of all baseline/tuned variants, seven ablations and XAI.
It can take several minutes. The selected artifact remains trained on the 70%
partition; it is not silently refit using the validation/test rows. The optional
expanded experiment is independent:

```powershell
.venv\Scripts\python.exe -m crop_xai train --cohort all --output artifacts/expanded --jobs 2
```

For a reduced-grid smoke check (not final report measurements):

```powershell
.venv\Scripts\python.exe -m crop_xai train --quick --output artifacts/smoke22 --shap-samples 3 --lime-samples 1000
```

`--folds`, `--seed`, `--shap-samples`, `--lime-samples` and `--no-explain` are
explicitly configurable and recorded. Default SHAP global figures explain a
random 12-row test sample, not the entire population. LIME explains the first
sampled row for the predicted class; its JSON records local-surrogate R2.

## Recommend a crop

```powershell
.venv\Scripts\python.exe -m crop_xai predict --model artifacts/paper22/model.joblib --input examples/soil_sample.json --threshold 0.5 --top-k 3 --log artifacts/predictions.jsonl --explain-output artifacts/sample-explanation
```

The seven keys must match exactly. Invalid domains, missing fields and nonfinite
inputs are rejected. Low maximum probability produces `abstained_low_confidence`
and a null recommendation; alternatives remain visible. Out-of-training-range
inputs produce warnings. Scores are **not calibrated crop-success probabilities**
and neither explanations nor recommendations prescribe nutrient interventions.
Only load trusted joblib files: Python model deserialization can execute code.

## Evidence and reporting

Each complete run produces:

```text
run.json                      provenance, dataset cohort, versions, grids, status
splits.json / cv_splits.json   source rows and disjoint split/fold membership
cv_fold_metrics.csv           baseline and selected-grid per-fold metrics
search/                       every candidate and selected hyperparameters
validation_comparison.csv     selection evidence
test_comparison.csv            test scores, selected-before-test flag
evaluation/<model>/            metrics, per-class report, predictions, confusion
ablation.csv / ablation.png    feature-removal experiment
eda/                          training-only statistics and plots
explanations/                 SHAP arrays/plots/JSON, LIME HTML/JSON/plot
model.joblib                  selected fitted pipeline and training reference
REPORT_EVIDENCE.md             report entry point and limitations
manifest.json                 artifact SHA256 hashes
```

Use [`docs/CLAUDE_REPORT_PROMPT.md`](docs/CLAUDE_REPORT_PROMPT.md) with the supplied
paper and a complete run's evidence. The prompt requires measured results and
distinguishes the paper cohort from the expanded dataset. An optional
`notebooks/paper_reproduction.ipynb` wraps the same pipeline without duplicating
the implementation. The original `crop_recommendation_pipeline.ipynb` is a legacy
extension: its test-selected model and saved results are not reproduction evidence.

## Optional Google Cloud Storage backup

No cloud credentials, public bucket or deployment is created automatically.
Install the optional requirements and configure Application Default Credentials
with permission to create objects in your own existing bucket. Use a fresh prefix
for each backup; existing objects cannot be overwritten.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-cloud.txt
.venv\Scripts\python.exe -m crop_xai upload --run artifacts/paper22 --bucket YOUR_BUCKET --prefix crop-xai/RUN_ID
```

This stores the artifact bundle, including structured predictions and inputs.
It is an export/backup integration, **not a deployed cloud inference service**.
Do not add credentials or `.env` files to Git. See
[`docs/VALIDATION.md`](docs/VALIDATION.md) for the actual verification record.
