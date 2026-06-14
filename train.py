"""Train a Telco customer churn classifier.

Dependencies: pandas, numpy, scikit-learn, xgboost, joblib
"""

import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

BINARY_YES_NO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

NUMERIC_COLS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Monthly_to_Total_Ratio",
    *BINARY_YES_NO_COLS,
]

CATEGORICAL_COLS = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "Tenure_Group",
]

HYPERPARAM_GRID = {
    "n_estimators": [150, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
}


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add domain features and encode binary Yes/No columns."""
    engineered = data.copy()
    engineered["TotalCharges"] = pd.to_numeric(engineered["TotalCharges"], errors="coerce")
    engineered["Monthly_to_Total_Ratio"] = engineered["MonthlyCharges"] / (
        engineered["TotalCharges"] + 1e-5
    )
    engineered["Tenure_Group"] = pd.cut(
        engineered["tenure"],
        bins=[-1, 12, 24, 48, 60, np.inf],
        labels=["0-12", "12-24", "24-48", "48-60", "60+"],
    ).astype(str)

    for col in BINARY_YES_NO_COLS:
        engineered[col] = engineered[col].map({"Yes": 1, "No": 0})

    return engineered


def prepare_raw_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Drop IDs, extract target, and engineer features."""
    data = df.copy()
    data = data.drop(columns=["customerID"], errors="ignore")

    y = None
    if "Churn" in data.columns:
        y = data["Churn"].map({"Yes": 1, "No": 0})
        data = data.drop(columns=["Churn"])

    data = engineer_features(data)
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    return data[feature_cols], y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                NUMERIC_COLS,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                CATEGORICAL_COLS,
            ),
        ],
    )


def build_model(scale_pos_weight: float, **hyperparams) -> XGBClassifier:
    params = {
        "n_estimators": 150,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    params.update(hyperparams)
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )


def find_optimal_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    """Find the probability threshold that maximizes F1 on validation data."""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.0, 1.01, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(round(threshold, 2))

    return best_threshold, best_f1


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> tuple[XGBClassifier, dict, float, float]:
    """Select XGBoost hyperparameters by validation F1 after threshold tuning."""
    best_val_f1 = -1.0
    best_params: dict | None = None
    best_threshold = 0.5
    best_model: XGBClassifier | None = None

    grid_keys = list(HYPERPARAM_GRID.keys())
    for combo in product(*(HYPERPARAM_GRID[key] for key in grid_keys)):
        params = dict(zip(grid_keys, combo))
        model = build_model(scale_pos_weight, **params)
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = find_optimal_threshold(y_val, val_prob)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = params
            best_threshold = threshold
            best_model = model

    if best_model is None or best_params is None:
        raise RuntimeError("Hyperparameter tuning failed to find a valid model.")

    return best_model, best_params, best_threshold, best_val_f1


def prepare_inference_frame(customer: dict | pd.DataFrame) -> pd.DataFrame:
    """Build a single-row feature frame ready for the fitted preprocessor."""
    if isinstance(customer, dict):
        row = pd.DataFrame([customer])
    else:
        row = customer.copy()

    row = row.drop(columns=["customerID", "Churn"], errors="ignore")
    row = engineer_features(row)
    return row[NUMERIC_COLS + CATEGORICAL_COLS]


def predict_churn(
    customer: dict,
    preprocessor: ColumnTransformer,
    model: XGBClassifier,
    threshold: float,
) -> tuple[str, float]:
    """Transform customer data and return churn label + probability."""
    features = prepare_inference_frame(customer)
    transformed = preprocessor.transform(features)
    probability = float(model.predict_proba(transformed)[0, 1])
    churn_label = "Yes" if probability >= threshold else "No"
    return churn_label, probability


def train_and_evaluate(
    data_path: Path | str = DATA_PATH,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> dict:
    df = load_data(data_path)
    X, y = prepare_raw_data(df)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"scale_pos_weight (train): {scale_pos_weight:.4f}")
    print("Tuning hyperparameters on validation F1 (with threshold search)...")

    model, best_params, optimal_threshold, val_f1 = tune_hyperparameters(
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        scale_pos_weight,
    )

    test_prob = model.predict_proba(X_test_processed)[:, 1]
    y_pred = (test_prob >= optimal_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, test_prob)), 4),
        "f1_churn": round(float(f1_score(y_test, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "optimal_threshold": optimal_threshold,
        "val_f1_at_threshold": round(float(val_f1), 4),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "best_params": best_params,
        "confusion_matrix": cm.tolist(),
        "model": "XGBoost",
    }

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    with open(artifacts_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(preprocessor, artifacts_path / "preprocessor.joblib")
    joblib.dump(model, artifacts_path / "model.joblib")
    joblib.dump({"model": "XGBoost", "best_params": best_params}, artifacts_path / "metadata.joblib")

    print("\nBest hyperparameters (validation):")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    print("\nModel evaluation (holdout test set)")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1 (churn class): {metrics['f1_churn']:.4f}")
    print(f"Optimal threshold (tuned on validation): {metrics['optimal_threshold']:.2f}")
    print(f"Validation F1 at optimal threshold: {metrics['val_f1_at_threshold']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}  # misleading alone due to imbalance")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
