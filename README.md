# Crop Recommendation System using Explainable AI (XAI)

> A machine learning pipeline for intelligent crop recommendation backed by LIME (Local Interpretable Model-agnostic Explanations) to make model decisions transparent and interpretable.

---

## Team Members

| Name | Enrollment No. |
|------|----------------|
| Aadi Singhal | 202352301 |
| Piyush Jain | 202352324 |
| Sanchit Mittal | 202352328 |

---

## Project Overview

This project builds a **crop recommendation system** that suggests the most suitable crop to grow based on soil and climate parameters. We integrate **LIME (Local Interpretable Model-agnostic Explanations)** so farmers and agronomists can understand *why* a particular crop was recommended — not just what was recommended.

### Input Features

| Feature | Description | Unit |
|---------|-------------|------|
| N | Nitrogen content in soil | mg/kg |
| P | Phosphorus content in soil | mg/kg |
| K | Potassium content in soil | mg/kg |
| Temperature | Ambient temperature | C |
| Humidity | Relative humidity | % |
| pH | Soil pH level | — |
| Rainfall | Annual rainfall | mm |

### Output
- **Recommended Crop** (one of 22 crop classes)
- **Confidence Score** (class probability)
- **LIME Explanation** (feature-level contributions for each prediction)

---

## Dataset

| Property | Value |
|----------|-------|
| Source | `Crop_recommendation.csv` |
| Total Records | 2,200 |
| Number of Classes | 22 |
| Class Distribution | Perfectly balanced (100 records/class) |
| Split | 80% train / 20% test (stratified) |
| Test Set Size | 440 samples |

---

## Models Trained

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | 98.41% | Strong baseline, fast training |
| Decision Tree | 97.95% | Interpretable structure |
| Random Forest | **99.55%** | Best performer, used for final predictions |
| XGBoost | — | Gradient boosted ensemble |
| CatBoost | — | Handles categorical features natively |
| EBM (Explainable Boosting Machine) | — | Inherently interpretable |
| TabNet | — | Attention-based deep tabular model |

> **Note:** Evaluation was done on 80/20 stratified split (n=440 test set).

---

## Sample Prediction & LIME Explanation

### Input
```
N = 90, P = 40, K = 40
Temperature = 25 C, Humidity = 80%
pH = 6.5, Rainfall = 200 mm
```

### Prediction
> Predicted Crop: **Rice** (confidence: 51.5%)

### LIME Feature Contributions

| Feature | Contribution | Direction |
|---------|-------------|-----------|
| Rainfall | +0.044 | Pushes toward Rice |
| Potassium (K) | +0.031 | Pushes toward Rice |
| Nitrogen (N) | +0.027 | Pushes toward Rice |
| pH | -0.007 | Slightly against Rice |
| Humidity | -0.005 | Slightly against Rice |

---

## Exploratory Data Analysis (EDA)

- **Crop Distribution:** Perfectly balanced — 100 records per class across all 22 crops
- **Feature Histograms:** 7-panel grid showing distribution of N, P, K, Temperature, Humidity, pH, and Rainfall
- **Correlation Heatmap:** Reveals moderate correlation between temperature and humidity; K and P show low inter-feature correlation

---

## Project Structure

```
crop-recommendation-xai/
|
|-- data/
|   `-- crop_recommendation.csv             # Raw dataset (2,200 records, 22 classes)
|
|-- notebooks/
|   `-- crop_recommendation_pipeline.ipynb  # Full ML pipeline: EDA, training, LIME
|
|-- requirements.txt                        # Python dependencies
`-- README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/SanchitMittalG/ai_project.git
cd ai_project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook notebooks/crop_recommendation_pipeline.ipynb
```

---

## XAI Method Used

### LIME (Local Interpretable Model-agnostic Explanations)
- Explains individual predictions by perturbing the input and fitting a local linear model around it
- Shows which features pushed the model *toward* or *against* a specific crop prediction
- Model-agnostic — works with any classifier (Random Forest, XGBoost, etc.)
- Helps build trust in the system by providing human-readable justifications

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `scikit-learn` | Logistic Regression, Decision Tree, Random Forest |
| `xgboost` | Gradient Boosted Trees |
| `catboost` | CatBoost model |
| `pytorch-tabnet` | TabNet deep learning model |
| `interpret` | Explainable Boosting Machine (EBM) |
| `lime` | Local LIME explanations |
| `imbalanced-learn` | SMOTE for class imbalance |
| `seaborn` | Heatmap and distribution plots |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib` | Visualizations |

---

## Notes

- The train/test split used in this project is **80/20 stratified**; the notebook is the canonical reproducible source.
- LIME explanations are generated locally per prediction — results may vary slightly between runs due to LIME's perturbation sampling.

---
