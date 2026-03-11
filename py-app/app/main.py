"""
main.py
-------
FastAPI application for the Shopping Channel Preference Predictor.

This file defines all the API routes (endpoints) that clients can call.
FastAPI automatically generates interactive documentation at:
  - http://localhost:8000/docs    (Swagger UI)
  - http://localhost:8000/redoc   (ReDoc)

To run the server:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from app import model_loader


# ─── Application Lifespan (startup / shutdown) ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs ONCE when the server starts.
    We pre-load the model so the first request isn't slow.
    """
    print("🚀 Starting Shopping Preference Predictor API...")
    model_loader._load_artifacts()
    print("✅ Model ready!")
    yield
    print("🛑 Shutting down API.")


# ─── FastAPI App Initialization ───────────────────────────────────────────────

app = FastAPI(
    title="Shopping Channel Preference Predictor API",
    description=(
        "Predicts whether a customer prefers **Online**, **Store**, or **Hybrid** shopping "
        "based on behavioral, demographic, and attitudinal features.\n\n"
        "**Model:** Logistic Regression (Tuned) — ~98% accuracy on test set.\n\n"
        "**Dataset:** Online vs In-Store Shopping Behaviour (11,789 records, 26 features)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS Middleware (allows Streamlit / web app to call this API) ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, restrict to your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    """
    Root endpoint — confirms the API is running.
    """
    return {
        "message": "Shopping Channel Preference Predictor API is running!",
        "docs"   : "/docs",
        "health" : "/health"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Returns model metadata so you can verify the right model is loaded.
    """
    try:
        info = model_loader.get_model_info()
        return {"status": "healthy", "model_info": info}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict shopping preference for one customer"
)
def predict_single(customer: CustomerFeatures):
    """
    **Single prediction endpoint.**

    Send a JSON body with all 24 customer features.
    Returns the predicted shopping channel preference (Online / Store / Hybrid)
    along with confidence score and class probabilities.

    **Example request body** — see the schema below for all fields.
    """
    try:
        # Convert Pydantic model → plain dict for model_loader
        raw_input = customer.dict()
        result    = model_loader.predict(raw_input)
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model files not found. Run training notebooks first. Error: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict shopping preferences for multiple customers"
)
def predict_batch(request: BatchPredictionRequest):
    """
    **Batch prediction endpoint.**

    Send up to 100 customer records in a single request.
    Returns a list of predictions in the same order as input.
    """
    try:
        raw_inputs  = [c.dict() for c in request.customers]
        predictions = model_loader.predict_batch(raw_inputs)
        return {
            "total"      : len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/features", tags=["Info"])
def get_feature_info():
    """
    Returns information about all input features:
    - Feature names grouped by type
    - Valid values for categorical features
    - Feature descriptions
    """
    import json
    from pathlib import Path

    feature_info_path = Path(__file__).resolve().parent.parent.parent / "models" / "feature_info.json"
    try:
        with open(feature_info_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="feature_info.json not found. Run preprocessing first.")