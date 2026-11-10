# Validation record

Validation date: November 10, 2026. Local environment: Windows 11, Python 3.12.2.
Exact resolved packages are recorded in `requirements-paper-lock.txt`.

## Executed checks

This repository had no test suite. Rather than introducing an unrelated testing
framework, validation used executed Python assertions, CLI error-path checks and
actual end-to-end experiments. These are observed checks, not a claim of exhaustive
coverage or formal verification.

- All `crop_xai` modules compile; the new notebook's Python cells compile and
  its JSON parses. The legacy notebook retains its exploratory results, with
  an explicit warning and a corrected repository-relative CSV path.
- CSV audited: 6,596 rows, 51 classes, no missing numeric values and no exact
  duplicate rows. The canonical cohort has 2,200 rows and 22 labels, 100 each.
- Default partition sizes are 1,540 / 330 / 330. Assertions verify disjointness,
  full row coverage, repeatability and representation of all 22 classes.
- Negative nutrients, invalid humidity/pH, infinity, missing inference values,
  extra target/input columns and invalid thresholds are rejected.
- Synthetic duplicate rows are removed, conflicting labels on identical inputs
  are rejected, and a pipeline trained with missing values imputes successfully.
- Six baselines and six tuned variants execute; the full grids comprise 27
  candidate configurations and five CV folds. Baseline/tuned fold metrics contain
  60 rows. All searches, evaluations and ablations run on recorded partitions.
- An expanded-data reduced-grid smoke experiment completed, including
  multiclass SHAP with shape `(3, 7, 51)`. It is not used as final paper evidence.
- Full paper-cohort execution includes SHAP shape `(12, 7, 22)`, local additive
  reconstruction, predicted-class LIME alignment and all seven feature ablations.
- Model reload and CLI prediction execute with structured JSONL logging and
  inference-time SHAP/LIME. The supplied example predicts rice with score 0.52
  in the recorded run; it abstains when the threshold is raised to 1.0.
- Assertions verify normalized probabilities, range warnings and SVM-style
  estimator-decision/probability-ranking disagreements. Explanation targets,
  reported labels and threshold scores follow the estimator's actual decision.
- Nonempty experiment destinations and invalid CV-fold arguments produce CLI
  errors instead of overwriting or silently mixing artifacts.
- SHA256 artifact manifests are verified against the generated files; SHAP
  predicted-class additivity residuals are checked below 1e-6.
- GCS upload interface is exercised with a mocked client: incomplete runs are
  rejected before client creation, checksums are attached, bucket URLs are
  rejected as names, and every write uses `if_generation_match=0`.
- `git diff --check` is run before final commit.

## Authoritative result bundle

The final full experiment is `artifacts/paper22-final`; its Git-friendly evidence
snapshot is `results/paper22`. The trusted fitted binary stays local at
`artifacts/paper22-final/model.joblib` and is intentionally excluded from Git.
Consult the bundle's `run.json` for exact source commit, versions, seed, dataset
hash, elapsed time, selected model and completion status. Generated figures were
visually inspected locally; no browser or cloud deployment is implied.

The selected tuned Random Forest is chosen by validation macro F1, not test
performance. Its held-out score is 329/330 correct (accuracy 0.996969696969697,
macro F1 approximately 0.9969663262). Other models may have better test results;
that is not permission to reselect after looking at the test set.

The sampled LIME explanation has surrogate R2 approximately 0.276. The stored
warning makes its low fidelity explicit: it should not be presented as a faithful
causal explanation or a validated nutrient prescription. The full CSV's original
provenance and measurement-unit conventions remain unverified.

## Not performed / not established

- No real GCS upload, bucket provisioning, scheduled backup or deployed inference
  service: user-owned cloud resources and credentials were not supplied.
- No historical-yield training, external/temporal/geographic field validation,
  probability calibration, causal experiment or agronomic safety validation.
- The optional legacy XGBoost/CatBoost/EBM/TabNet notebook was not rerun; its saved
  results are not substituted for the six paper baselines.
- GitHub pushes were attempted but rejected with HTTP 403: the configured
  account `AadiS27` lacks write permission to `SanchitMittalG/ai_project`.
  Work is committed locally; collaborator access or another authorized remote is
  required before those commits can be published. No push success is claimed.

## Re-run commands

From the repository root, using the installed environment:

```powershell
.venv\Scripts\python.exe -m compileall -q crop_xai
.venv\Scripts\python.exe -m crop_xai train --output artifacts/reproduction-new --jobs 2
.venv\Scripts\python.exe -m crop_xai predict --model artifacts/reproduction-new/model.joblib --input examples/soil_sample.json --threshold 0.5 --log artifacts/reproduction-predictions.jsonl --explain-output artifacts/reproduction-explanation
git diff --check
```

Use a fresh output directory for each execution. Five-fold training-only CV
scores after tuning are not unbiased nested-CV estimates; final generalization
claims must use the untouched held-out partition and acknowledge its small size.
