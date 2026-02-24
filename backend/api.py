"""
FastAPI Backend for Credit Risk Prediction.
Serves the trained model via REST API.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import uvicorn
import shutil
import sys
import os
import threading
import traceback
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download

# Add backend to path for preprocessor pickle compatibility
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.predict import CreditRiskModel
from config import HF_TOKEN, HF_REPO_ID, HF_FILES, MODELS_DIR

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
crisk_model = None

# Track training status in memory
training_status = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "message": "No training has been run yet.",
    "error": None,
}
training_lock = threading.Lock()

# ---------------------------------------------------------------------------
# HuggingFace Helpers
# ---------------------------------------------------------------------------
def _hf_enabled() -> bool:
    """Returns True if HF Hub is configured."""
    return bool(HF_TOKEN) and HF_TOKEN != "your_hf_token_here"

def _upload_to_hf() -> bool:
    """Upload model artifacts from MODELS_DIR to HuggingFace Hub."""
    if not _hf_enabled():
        print("⚠️  HF Hub not configured — skipping upload.")
        return False
    try:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO_ID, exist_ok=True, private=False)
        for fname in HF_FILES:
            fpath = MODELS_DIR / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=HF_REPO_ID,
                    token=HF_TOKEN,
                )
                print(f"☁️  Uploaded {fname} → {HF_REPO_ID}")
        return True
    except Exception as e:
        print(f"❌ HF upload failed: {e}")
        return False

def _download_from_hf() -> bool:
    """Download model artifacts from HuggingFace Hub into MODELS_DIR."""
    if not _hf_enabled():
        return False
        
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Check if the repo exists first
    try:
        api = HfApi(token=HF_TOKEN)
        api.repo_info(repo_id=HF_REPO_ID)
    except Exception as e:
        if "404" in str(e):
            raise FileNotFoundError(f"HuggingFace repo '{HF_REPO_ID}' does not exist yet. Please train a model first.")
        raise
        
    for fname in HF_FILES:
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=fname,
                token=HF_TOKEN,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False, 
            )
            print(f"⬇️  Downloaded {fname} from {HF_REPO_ID}")
        except Exception as e:
            if "404" in str(e):
                raise FileNotFoundError(f"File '{fname}' missing in repo '{HF_REPO_ID}'. Please train a model.")
            raise e
    return True

def _load_model() -> None:
    """Download from HF (if enabled) and load into global state."""
    global crisk_model
    
    if _hf_enabled():
        print(f"🔄 Pulling model from HuggingFace Hub ({HF_REPO_ID})...")
        _download_from_hf()
    else:
        print("ℹ️  HF Hub not configured — loading from local files.")
        
    base_dir = Path(__file__).resolve().parent.parent
    model_data_path = base_dir / "models" / "model_data.joblib"
    
    if not model_data_path.exists():
         raise FileNotFoundError("Model file not found. Please train first.")
         
    crisk_model = CreditRiskModel(model_path=str(model_data_path))
    print("✅ Model loaded successfully.")

# ---------------------------------------------------------------------------
# Lifespan (Startup/Shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global crisk_model
    try:
        _load_model()
    except FileNotFoundError as e:
        print(f"⚠️ Failed to load model: {e}")
        crisk_model = None
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

class TrainingStatusResponse(BaseModel):
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: str
    error: Optional[str] = None
    model_name: Optional[str] = None
    num_features: Optional[int] = None
    accuracy: Optional[float] = None

# ---------------------------------------------------------------------------
# Background Training Worker
# ---------------------------------------------------------------------------
def _run_training_pipeline() -> None:
    global training_status
    
    with training_lock:
        training_status["status"] = "running"
        training_status["started_at"] = datetime.now().isoformat()
        training_status["completed_at"] = None
        training_status["error"] = None
        training_status["message"] = "Training started..."

    try:
        import sys
        base_dir = Path(__file__).resolve().parent.parent
        if str(base_dir) not in sys.path:
            sys.path.insert(0, str(base_dir))
            
        from backend.training.data_loader import DataLoader
        from backend.training.preprocessing import CreditRiskPreprocessor
        from backend.training.feature_engineering import apply_resampling, create_features
        from backend.training.train import train_model
        from backend.training.utils import save_joblib
        from sklearn.model_selection import train_test_split
        
        # 1. Loading
        with training_lock:
            training_status["message"] = "Step 1/3: Loading data..."
        
        data_dir_path = base_dir / "dataset"
        loader = DataLoader(data_dir=str(data_dir_path))
        df = loader.get_data()
        
        # 1.5 Create Features
        df = create_features(df)
        
        # 2. Preprocessing
        with training_lock:
            training_status["message"] = "Step 2/3: Preprocessing data..."
            
        from config import TARGET_COLUMN, ID_COLUMNS, TEST_SIZE, RANDOM_STATE
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in uploaded datasets.")
            
        X = df.drop(TARGET_COLUMN, axis=1)
        y = df[TARGET_COLUMN]
        X = X.drop([c for c in ID_COLUMNS if c in X.columns], axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        preprocessor = CreditRiskPreprocessor()
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Resampling
        X_train_res, y_train_res = apply_resampling(X_train_processed, y_train)
        
        # 3. Training
        with training_lock:
            training_status["message"] = "Step 3/3: Training Logistic Regression Model..."
        model = train_model(X_train_res, y_train_res)
        
        # 4. Evaluate to get accuracy score
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_test_processed)
        acc_score = accuracy_score(y_test, y_pred)
        
        # 5. Save locally
        model_data = {
            'model': model,
            'features': preprocessor.feature_names,
            'scaler': preprocessor.scaler,
            'cols_to_scale': preprocessor.numeric_cols
        }
        model_data_path = base_dir / "models" / "model_data.joblib"
        save_joblib(model_data, model_data_path)
        
        with training_lock:
            training_status["model_name"] = "Logistic Regression"
            training_status["num_features"] = len(preprocessor.feature_names)
            training_status["accuracy"] = acc_score
        
        # 5. Upload to HuggingFace
        with training_lock:
            training_status["message"] = "Uploading model to HuggingFace Hub..."
        hf_uploaded = _upload_to_hf()
        hf_note = f" | Uploaded to HF Hub ({HF_REPO_ID})" if hf_uploaded else ""
        
        # 6. Reload model in API
        _load_model()
        
        with training_lock:
            training_status["status"] = "completed"
            training_status["completed_at"] = datetime.now().isoformat()
            training_status["message"] = f"Training complete! Logistic Regression{hf_note}"
            
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"❌ Training failed:\n{err_msg}")
        with training_lock:
            training_status["status"] = "failed"
            training_status["completed_at"] = datetime.now().isoformat()
            training_status["message"] = f"Training failed: {str(e)}"
            training_status["error"] = err_msg

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Credit Risk Prediction API is running.",
        "status": "active" if crisk_model else "model_loading_failed"
    }

@app.get("/model/info")
def get_model_info():
    if not crisk_model:
        raise HTTPException(status_code=404, detail="Model not loaded")
    return {
        "model_name": "Logistic Regression",
        "num_features": len(crisk_model.features) if hasattr(crisk_model, 'features') and crisk_model.features is not None else "Unknown"
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

@app.post("/train")
async def trigger_training(files: List[UploadFile] = File(...)):
    with training_lock:
        if training_status["status"] == "running":
            raise HTTPException(status_code=409, detail="Training is already in progress.")

    # Save uploaded files into the dataset directory
    data_dir = Path(__file__).resolve().parent.parent / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        if file.filename:
            file_path = data_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

    thread = threading.Thread(
        target=_run_training_pipeline,
        daemon=True,
    )
    thread.start()
    
    return {
        "message": "Training started in background.",
        "status": "running",
        "poll_url": "/train/status",
    }

@app.get("/train/status", response_model=TrainingStatusResponse)
async def get_training_status():
    with training_lock:
        return TrainingStatusResponse(**training_status)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
