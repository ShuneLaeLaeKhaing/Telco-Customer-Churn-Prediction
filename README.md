# Telco Customer Churn Prediction

## Introduction

This project predicts customer churn for a telecommunications company using machine learning. It includes exploratory data analysis, a reproducible training pipeline, and a web API for real-time predictions.

The goal is to identify customers at risk of leaving so retention teams can act early.

## Dataset

Source: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains **7,043 customers** and **21 columns** — demographics, account details, services subscribed, and churn status. The CSV is included at:

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Project Objectives

1. **Data analysis** — Inspect the data and find factors linked to churn (e.g. fiber vs DSL, contract type, tenure).
2. **Model training** — Clean the data, train an XGBoost classifier, and handle class imbalance.
3. **Prediction service** — Expose a REST API and web form to score new customers.

## Project Structure

```
telco-churn-prediction/
├── analysis.ipynb      # Exploratory data analysis
├── train.py            # Training pipeline
├── app.py              # FastAPI service
├── static/index.html   # Prediction UI
├── data/               # Raw dataset
├── artifacts/          # Saved model, preprocessor, metrics
└── requirements.txt
```

## Data Preprocessing

Handled in `train.py`:

- Drop `customerID` (not used for prediction)
- Fix `TotalCharges` as numeric; impute missing values with median
- Encode Yes/No fields to 1/0
- Engineer features: `Tenure_Group`, `Monthly_to_Total_Ratio`
- One-hot encode categoricals via sklearn `ColumnTransformer`
- Fit preprocessor on training data only (no leakage)

## Exploratory Data Analysis

See `analysis.ipynb`. It covers:

- Data overview and missing values
- Churn distribution (~73% stay, ~27% leave)
- Bar charts of churn rate by category (contract, internet, payment method, etc.)
- Histograms for tenure and charges
- Summary of top churn drivers

Run with:

```bash
jupyter notebook analysis.ipynb
```

## Model Building

- **Algorithm:** XGBoost
- **Class imbalance:** `scale_pos_weight` from training set ratio
- **Tuning:** Hyperparameter search on validation set; probability threshold tuned for best F1 (0.0–1.0, step 0.01)
- **Split:** Stratified train / validation / test (64% / 16% / 20%)

Train the model:

```bash
python train.py
```

Saved artifacts:

| File | Description |
|------|-------------|
| `artifacts/preprocessor.joblib` | Fitted imputer + encoder |
| `artifacts/model.joblib` | Trained XGBoost model |
| `artifacts/metrics.json` | Evaluation metrics + optimal threshold |

## Model Evaluation

Primary metrics for this imbalanced dataset are **F1 (churn class)** and **ROC-AUC**.

**Holdout test results:**

| Metric | Value |
|--------|-------|
| Model | XGBoost |
| ROC-AUC | 0.8454 |
| F1 (churn class) | 0.6181 |
| Optimal threshold | 0.6 |
| Validation F1 at threshold | 0.6718 |
| scale_pos_weight | 2.7684 |
| Accuracy | 0.7729 *(misleading alone due to imbalance)* |

**Best hyperparameters:**

```python
{'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05,
 'subsample': 0.9, 'colsample_bytree': 0.8}
```

**Confusion matrix** `[[TN, FP], [FN, TP]]` (test set):

```
[[830, 205],
 [115, 259]]
```

## Web Application & API

Start the server:

```bash
uvicorn app:app --reload --port 8000
```

| Endpoint | Description |
|----------|-------------|
| http://localhost:8000 | Web form (19 customer fields) |
| http://localhost:8000/docs | Swagger API docs |
| `GET /health` | Service health check |
| `POST /predict` | Returns `churn` and `churn_probability` |

## How to Use

1. Clone the repository and enter the project folder.

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the EDA notebook (optional but recommended):

```bash
jupyter notebook analysis.ipynb
```

4. Train the model (optional — pre-trained artifacts are included):

```bash
python train.py
```

5. Start the API:

```bash
uvicorn app:app --reload --port 8000
```

## Dependencies

Install via `requirements.txt`:

- pandas, numpy, scikit-learn, xgboost, joblib
- fastapi, uvicorn, pydantic
- matplotlib, seaborn, jupyter, notebook

## Conclusion

The pipeline explores churn patterns in the Telco dataset, trains an XGBoost model with proper imbalance handling, and serves predictions through a FastAPI application. Key churn signals include short tenure, month-to-month contracts, fiber optic service, and electronic check payments.
