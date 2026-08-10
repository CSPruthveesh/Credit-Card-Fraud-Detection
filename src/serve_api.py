import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Real-Time Credit Card Fraud Detection Service",
    version="1.0.0",
    description="Microservice for low-latency (<10ms) transaction fraud risk scoring."
)

# Global variables for model state
model = None
MODEL_PATH = "model.joblib"

@app.on_event("startup")
def load_model():
    """
    Pre-warms the API by loading the trained model state into memory at server startup.
    This eliminates load overhead latency on the first API request.
    """
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully from local storage.")
        # Pre-warm the C++ booster thread pool with a dummy prediction to eliminate first-request latency spikes
        try:
            model.predict_proba(np.zeros((1, 33)))
            print("C++ booster thread pool pre-warmed successfully.")
        except Exception as e:
            print(f"Booster pre-warming failed: {e}")
    else:
        # Create a mock classifier if model file is not found (for demonstration/health readiness)
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="stratified")
        # Dummy fit with both classes present to ensure predict_proba yields 2 columns
        model.fit(np.zeros((10, 33)), np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
        print("Warning: model.joblib not found. Initialized placeholder model for API start.")

class TransactionPayload(BaseModel):
    # Expecting 33 numerical features (V1 to V28, and scaled versions of: Amount, Time, Time_Delta, Last_5_Tx_Time_Span, Rolling_Mean_Amount_5)
    features: list[float] = Field(
        ...,
        min_items=33,
        max_items=33,
        description="List of 33 scaled features representing transaction attributes in chronological sequence."
    )

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    latency_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: TransactionPayload):
    """
    Scores an incoming transaction payload.
    Execution latency is designed for <10ms to meet strict banking SLAs.
    """
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model service is unavailable.")
        
    try:
        import time
        start_time = time.perf_counter()
        
        # Convert input list to numpy array shape (1, 33)
        input_data = np.array([payload.features], dtype=np.float32)
        
        # Execute prediction probabilities
        proba = model.predict_proba(input_data)[:, 1][0]
        
        # Binary classification based on risk threshold (set to 0.5)
        threshold = 0.5
        is_fraud = bool(proba >= threshold)
        
        latency = (time.perf_counter() - start_time) * 1000.0
        
        return PredictionResponse(
            fraud_probability=float(proba),
            is_fraud=is_fraud,
            threshold=threshold,
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health():
    """
    Liveness and readiness check probe.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """
    Serves the interactive Model Health & Playground HTML dashboard.
    """
    template_path = os.path.join("src", "templates", "dashboard.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Dashboard template not found.")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
