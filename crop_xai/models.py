from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def model_definitions(seed=42, quick=False):
    trees = 40 if quick else 200
    models = {
        "logistic_regression": LogisticRegression(max_iter=3000, random_state=seed),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=1),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(probability=True, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=20 if quick else 100, random_state=seed
        ),
    }
    grids = {
        "logistic_regression": {"C": [0.1, 1, 10]},
        "decision_tree": {"max_depth": [5, 10, None], "min_samples_leaf": [1, 3]},
        "random_forest": {"n_estimators": [100, 200], "max_features": ["sqrt", None]},
        "knn": {"n_neighbors": [3, 5, 9], "weights": ["uniform", "distance"]},
        "svm": {"C": [1, 10], "kernel": ["rbf", "linear"]},
        "gradient_boosting": {"learning_rate": [0.05, 0.1], "n_estimators": [50, 100]},
    }
    if quick:
        grids = {name: {key: values[:1] for key, values in grid.items()} for name, grid in grids.items()}
        grids["random_forest"]["n_estimators"] = [40]
        grids["gradient_boosting"]["n_estimators"] = [20]
    return models, grids


def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", MinMaxScaler()),
        ("classifier", model),
    ])
