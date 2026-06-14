"""FastAPI service for Telco customer churn prediction.

Dependencies: fastapi, uvicorn, pydantic, joblib, pandas
Run: uvicorn app:app --reload --port 8000
"""

import json
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from train import predict_churn

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "artifacts" / "preprocessor.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

app = FastAPI(title="Telco Churn Prediction API")


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn: str
    churn_probability: float


model = None
preprocessor = None
optimal_threshold = 0.5


@app.on_event("startup")
def load_artifacts() -> None:
    global model, preprocessor, optimal_threshold
    if (
        not MODEL_PATH.exists()
        or not PREPROCESSOR_PATH.exists()
        or not METRICS_PATH.exists()
    ):
        return
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    optimal_threshold = metrics.get("optimal_threshold", 0.5)


@app.get("/")
def ui():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.get("/api")
def api_info():
    return {
        "message": "Telco Churn Prediction API is running",
        "ui": "/",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    if model is None or preprocessor is None or not METRICS_PATH.exists():
        return {"status": "missing_model", "detail": "Run train.py first"}
    return {"status": "ok", "optimal_threshold": optimal_threshold}


@app.get("/predict")
def predict_help():
    return {
        "message": "Use POST /predict with a CustomerInput JSON body to get a churn prediction.",
        "example": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run train.py first.",
        )

    churn_label, probability = predict_churn(
        customer.model_dump(),
        preprocessor,
        model,
        optimal_threshold,
    )

    return PredictionResponse(
        churn=churn_label,
        churn_probability=round(probability, 4),
    )
