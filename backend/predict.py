import sys
import os
import pandas as pd
import joblib
import numpy as np

# Add project root to path to allow imports from backend.training
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.training.preprocessing import CreditRiskPreprocessor
from backend.training.feature_engineering import create_features

class CreditRiskModel:
    """
    Wrapper for loading the trained model and making predictions.
    Uses the custom CreditRiskPreprocessor and Feature Engineering pipeline.
    """
    def __init__(self, model_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Define paths to artifacts
        if model_path:
            self.model_path = model_path
        else:
             # Default to the consolidated model_data artifact
            self.model_path = os.path.join(base_dir, "models", "model_data.joblib")
            
        self.model = None
        self.features = None
        self.scaler = None
        self.cols_to_scale = None
        
        self.load()

    def load(self):
        """Loads model_data artifact containing model, features, and scaler."""
        try:
            if os.path.exists(self.model_path):
                 data = joblib.load(self.model_path)
                 self.model = data['model']
                 self.features = data['features']
                 self.scaler = data['scaler']
                 self.cols_to_scale = data['cols_to_scale']
            else:
                raise FileNotFoundError(f"Model artifact not found at {self.model_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load artifacts: {e}")

    def calculate_credit_score(self, probability):
        """
        Calculates credit score based on default probability.
        Score = 300 + (1 - probability) * 600
        """
        base_score = 300
        scale_length = 600
        non_default_probability = 1 - probability
        
        credit_score = int(base_score + (non_default_probability * scale_length))
        
        if 300 <= credit_score < 500:
            rating = 'Poor'
        elif 500 <= credit_score < 650:
            rating = 'Average'
        elif 650 <= credit_score < 750:
            rating = 'Good'
        elif 750 <= credit_score <= 900:
            rating = 'Excellent'
        else:
            rating = 'Undefined'
            
        return credit_score, rating

    def predict(self, data: pd.DataFrame):
        """
        Predicts default probability, credit score, and rating.
        Args:
            data (pd.DataFrame): Raw input data.
        Returns:
            list of dicts: Prediction results.
        """
        # 1. Feature Engineering (Raw -> Derived Features)
        df = create_features(data)
        
        # 2. Cleaning (Typo correction, filtering)
        # We reuse the lightweight clean_data logic (stateless)
        df = CreditRiskPreprocessor().clean_data(df)
        
        # 3. Preprocessing using model_data components
        # A. Scale Numeric
        if self.cols_to_scale:
             X_numeric = self.scaler.transform(df[self.cols_to_scale])
             df_numeric = pd.DataFrame(X_numeric, columns=self.cols_to_scale, index=df.index)
        else:
             df_numeric = pd.DataFrame(index=df.index)
             
        # B. Encode Categorical
        # Identify categorical columns (features not in cols_to_scale)
        # Note: We assume all other columns in cleaned df are categorical
        cat_cols = [c for c in df.columns if c not in self.cols_to_scale]
        
        if cat_cols:
            # Use get_dummies without drop_first=True to ensure we see all present categories
            # The reindex step will then filter to match training features (dropping reference vars)
            df_encoded = pd.get_dummies(df[cat_cols], drop_first=False)
        else:
            df_encoded = pd.DataFrame(index=df.index)
            
        # C. Concatenate and Align
        X_final = pd.concat([df_numeric, df_encoded], axis=1)
        
        # Align with model features (Add 0s for missing cols, drop unknown cols)
        X_final = X_final.reindex(columns=self.features, fill_value=0)
        
        # 4. Prediction
        predictions = self.model.predict(X_final)
        probabilities = self.model.predict_proba(X_final)[:, 1] # Prob of class 1 (Default)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            credit_score, rating = self.calculate_credit_score(prob)
            
            results.append({
                "default_prediction": int(pred),
                "default_probability": float(prob),
                "credit_score": credit_score,
                "rating": rating
            })
            
        return results

if __name__ == "__main__":
    try:
        model = CreditRiskModel()
        print("Model loaded successfully.")
        
        # Test Prediction with valid data matching Streamlit defaults
        sample_data = pd.DataFrame([{
            'age': 28,
            'income': 1200000,
            'loan_amount': 2560000,
            'loan_tenure_months': 36,
            'avg_dpd_per_delinquency': 20,
            'delinquency_ratio': 30,
            'credit_utilization_ratio': 30,
            'number_of_open_accounts': 2,
            'residence_type': 'Owned',
            'loan_purpose': 'Home',
            'loan_type': 'Unsecured'
        }])
        
        print("\nTesting Prediction...")
        result = model.predict(sample_data)
        print("Prediction Result:", result)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
