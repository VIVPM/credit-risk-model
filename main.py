"""
Main pipeline script for Credit Risk Prediction.
Orchestrates data loading, preprocessing, training, and evaluation.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    DATA_DIR, MODELS_DIR,
    TARGET_COLUMN, ID_COLUMNS, TEST_SIZE, RANDOM_STATE
)
from backend.training.data_loader import DataLoader
from backend.training.preprocessing import CreditRiskPreprocessor
from backend.training.feature_engineering import apply_resampling
from backend.training.train import train_model
from backend.training.evaluate import evaluate_model
from backend.training.utils import save_joblib

def main():
    print("=== Starting Credit Risk Pipeline ===")
    
    # 1. Load Data
    print("\n[1/6] Loading Data...")
    loader = DataLoader(data_dir=str(DATA_DIR))
    df = loader.get_data()
    print(f"Data loaded: {df.shape}")
    
    # 1.5 Create Derived Features
    print("\n[1.5/6] Creating Derived Features...")
    from backend.training.feature_engineering import create_features
    df = create_features(df)
    
    # 2. Prepare Data (Drop IDs, Split Target)
    print("\n[2/6] Preparing Data...")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")
        
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    # Drop IDs
    X = X.drop([c for c in ID_COLUMNS if c in X.columns], axis=1)
    
    # Split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 3. Preprocessing
    print("\n[3/6] Preprocessing...")
    preprocessor = CreditRiskPreprocessor()
    preprocessor.fit(X_train)
    
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Processed feature count: {X_train_processed.shape[1]}")
    
    # Save Preprocessor
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    # 4. Feature Engineering (Resampling)
    print("\n[4/6] Resampling...")
    # apply_resampling is currently RandomUnderSampler in the file
    X_train_res, y_train_res = apply_resampling(X_train_processed, y_train)
    
    # 5. Training
    print("\n[5/6] Training Model...")
    model = train_model(X_train_res, y_train_res)
    
    # Save Model
    model_path = MODELS_DIR / "logistic_regression_model.joblib"
    save_joblib(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save Model Data (Notebook Compatibility)
    model_data = {
        'model': model,
        'features': preprocessor.feature_names,
        'scaler': preprocessor.scaler,
        'cols_to_scale': preprocessor.numeric_cols
    }
    model_data_path = MODELS_DIR / "model_data.joblib"
    save_joblib(model_data, model_data_path)
    print(f"Model data saved to {model_data_path} (Notebook format compatibility)")
    
    # 6. Evaluation
    print("\n[6/6] Evaluation...")
    evaluate_model(model, X_test_processed, y_test)
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
