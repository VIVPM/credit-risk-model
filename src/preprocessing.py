import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os

# Add project root to path to find config/src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TYPO_CORRECTIONS, SELECTED_FEATURES
from src.utils import save_joblib, load_joblib

class CreditRiskPreprocessor:
    """
    Handles data cleaning, scaling, and encoding for the Credit Risk model.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        # Using sparse_output=False for easier DataFrame reconstruction
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.numeric_cols = []
        self.categorical_cols = []
        self.feature_names = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies basic data cleaning steps and selects features.
        """
        df = df.copy()
        
        # Apply typo corrections from config
        if 'loan_purpose' in df.columns:
            corrections = TYPO_CORRECTIONS.get('loan_purpose', {})
            df['loan_purpose'] = df['loan_purpose'].replace(corrections)
            
        # Filter for selected features only
        # Check which selected features are present in df
        available_features = [f for f in SELECTED_FEATURES if f in df.columns]
        if not available_features:
             # If none found (e.g. empty df), return as is or handle error
             pass
        else:
            df = df[available_features]
            
        return df

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the scaler and encoder to the training data.
        X should be the raw dataframe (minus target).
        """
        X = self.clean_data(X)
        
        # Identify columns by type
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fit scaler on numeric columns
        if self.numeric_cols:
            self.scaler.fit(X[self.numeric_cols])
            
        # Fit encoder on categorical columns
        if self.categorical_cols:
            self.encoder.fit(X[self.categorical_cols])
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using fitted scaler and encoder.
        Returns a processed DataFrame with all features.
        """
        X = self.clean_data(X)
        
        # 1. Scale Numeric Data
        if self.numeric_cols:
            X_numeric = self.scaler.transform(X[self.numeric_cols])
            df_numeric = pd.DataFrame(X_numeric, columns=self.numeric_cols, index=X.index)
        else:
            df_numeric = pd.DataFrame(index=X.index)

        # 2. Encode Categorical Data
        if self.categorical_cols:
            X_encoded = self.encoder.transform(X[self.categorical_cols])
            # Get feature names for encoded columns
            encoded_cols = self.encoder.get_feature_names_out(self.categorical_cols)
            df_encoded = pd.DataFrame(X_encoded, columns=encoded_cols, index=X.index)
        else:
            df_encoded = pd.DataFrame(index=X.index)
            
        # 3. Concatenate
        X_final = pd.concat([df_numeric, df_encoded], axis=1)
        self.feature_names = X_final.columns.tolist()
        
        return X_final

    def save(self, filepath):
        """Saves the preprocessor object."""
        save_joblib(self, filepath)

    @staticmethod
    def load(filepath):
        """Loads a saved preprocessor object."""
        return load_joblib(filepath)

if __name__ == "__main__":
    # Test stub
    # Assuming data_loader is available
    from data_loader import DataLoader
    try:
        loader = DataLoader(data_dir=os.path.join(os.path.dirname(__file__), "..", "dataset"))
        df = loader.get_data()
        
        X = df.drop('default', axis=1) # Assuming 'default' is target
        y = df['default']
        
        preprocessor = CreditRiskPreprocessor()
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)
        
        print("Preprocessing successful.")
        print(f"Original shape: {X.shape}")
        print(f"Processed shape: {X_processed.shape}")
        print(X_processed.head())
        
        # Test Save/Load
        preprocessor.save("models/preprocessor.joblib")
        print("Preprocessor saved.")
        
    except Exception as e:
        print(f"Test failed: {e}")
