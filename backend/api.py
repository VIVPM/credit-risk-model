"""
FastAPI Backend for Credit Risk Prediction.
Serves the trained model via REST API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import uvicorn
import sys
import os
from pathlib import Path

# Add src to path for preprocessor pickle compatibility
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# Local imports (fixing path for predict module if needed, or importing directly if in same folder)
# In this structure, predict.py is in the same folder.
from predict import CreditRiskModel

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
crisk_model = None

# ---------------------------------------------------------------------------
# Lifespan (Startup/Shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global crisk_model
    try:
        # Load model on startup
        print("Loading Credit Risk Model...")
        # We assume models are in ../models relative to this file
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / "models" / "logistic_regression_model.joblib"
        preprocessor_path = base_dir / "models" / "preprocessor.joblib"
        
        crisk_model = CreditRiskModel(
            model_path=str(model_path)
        )
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        crisk_model = None
    
    yield
    print("🛑 Shutting down.")

# ---------------------------------------------------------------------------
# App Definition
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(
        ..., 
        example=[{
            "age": 30,
            "income": 50000,
            "loan_amount": 10000,
            "loan_purpose": "Personal",
            # Add other fields as necessary based on feature names
        }]
    )

class PredictionResponse(BaseModel):
    results: List[Dict[str, Any]]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Credit Risk Prediction API is running.",
        "status": "active" if crisk_model else "model_loading_failed"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(request: PredictionRequest):
    if not crisk_model:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(request.data)
        
        # Predict
        results = crisk_model.predict(df)
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
