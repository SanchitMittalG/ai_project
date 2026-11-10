# Paper implementation and reproducibility contract

Reference: Surendra Kumar and Mohit Kumar, "Developing an XAI-Based Crop
Recommendation Framework Using Soil Nutrient Profiles and Historical Crop
Yields," IEEE Transactions on Consumer Electronics, 71(2), May 2025.
The supplied PDF is not redistributed in this repository.

## Original repository audit

The exploratory notebook includes engineered features, Random Forest, XGBoost,
CatBoost, TabNet, Logistic Regression, EBM, SHAP and LIME. It expects a CSV in
the working directory and selects its best model by test accuracy. It lacks the
paper's full tuning, baseline, cross-validation and ablation protocol. Its saved
outputs are not evidence for this reproduction.

## Implementation mapping

| Paper element | Reproduction |
| --- | --- |
| Seven inputs and crop labels | Schema, domain, finite-value and duplicate checks |
| Preprocessing | Fold-local median imputation and MinMax scaling |
| Data analysis | Pearson correlation, training-only z-score diagnostics |
| Three partitions | Stratified 70/15/15 split with saved row indices |
| Five-fold evaluation | Shuffled stratified CV using training rows only |
| LR, DT, RF, KNN, SVM, Gradient Boosting | Six common-split baseline pipelines |
| Hyperparameter optimization | Exhaustive GridSearchCV and saved candidate scores |
| Evaluation | Classification metrics, per-class reports, confusion matrices |
| RMSE, MAE, R2 | Segregated integer-label compatibility diagnostics |
| LIME and SHAP | Training-only references; local and global explanations |
| Ablation | Refit frozen selected configuration without each input |
| Suitability threshold | Ranked scores, abstention and range warnings |
| Remote storage | Optional authenticated Google Cloud Storage export |
| Report | Evidence, figures and provenance |

## Assumptions and limitations

- Repository CSV audit: 6,596 rows and 51 classes, not the README's former
  2,200 rows and 22 classes. The default paper22 cohort explicitly selects the
  22 canonical labels (100 rows each). The all cohort is a separate, imbalanced
  extension. This is a cohort reconstruction, not verified original dataset
  provenance. Both the source hash and original source row numbers are saved.

- The paper reserves 70% for training and 30% for testing/validation but does
  not subdivide the latter. We choose 15% validation and 15% test.
- Exact grids, source code and split indices are unavailable. Our recorded
  bounded grids are explicit reproduction choices, not author settings.
- Algorithm 1 uses min-max scaling while prose mentions z-scores. Z-scores are
  diagnostic only; no validation/test-based outlier removal occurs.
- Labels are classification targets, never inputs. One-hot target inputs would
  leak the answer. The balanced paper22 cohort does not require SMOTE. The all
  cohort retains its natural imbalance and reports macro and weighted metrics;
  it must not be described as the paper's balanced dataset.
- The CSV has no yield, date, field, location or rotation-history columns.
  Historical yield, temporal generalization, regional fairness and nutrient
  interventions cannot be trained or validated with this dataset.
- Numeric error/R2 on alphabetical class codes depend on arbitrary ordering.
  They are diagnostics, never selection criteria or comparable yield metrics.
- Probabilities are uncalibrated scores. XAI explains model behavior, not
  biological causation or safe fertilizer interventions.
- Cloud export requires a user-owned bucket and Application Default Credentials.
  No cloud upload/deployment is claimed unless actually performed.
- The original notebook is a legacy extension, not new report evidence.
