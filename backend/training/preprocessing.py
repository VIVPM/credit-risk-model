# Cleans, scales, and encodes the merged DataFrame.
# OneHotEncoder with drop='first' matches what the notebook used (pd.get_dummies(drop_first=True)).
# MinMaxScaler on numerics keeps everything in [0,1] which helps logistic regression converge.
# Fit only on train data — transform is applied separately to avoid leakage.

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TYPO_CORRECTIONS, SELECTED_FEATURES
from sklearn.impute import SimpleImputer


class CreditRiskPreprocessor:
    def __init__(self):
        self.scaler  = MinMaxScaler()
        self.numeric_cols     = []
        self.categorical_cols = []
        self.feature_names    = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix known typos in loan_purpose and filter down to SELECTED_FEATURES.
        Both are safe to apply at inference time too.
        """
        df = df.copy()

        if 'loan_purpose' in df.columns:
            corrections = TYPO_CORRECTIONS.get('loan_purpose', {})
            df['loan_purpose'] = df['loan_purpose'].replace(corrections)

        available = [f for f in SELECTED_FEATURES if f in df.columns]
        if available:
            df = df[available]

        return df

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit scaler and encoder on training data.
        Call this once on train split, then use transform() for both train and test.
        """
        X = self.clean_data(X)

        self.numeric_cols     = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if self.numeric_cols:
            self.scaler.fit(X[self.numeric_cols])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numerics, one-hot encode categoricals, concatenate into a single DataFrame."""
        X = self.clean_data(X)

        if self.numeric_cols:
            X_numeric  = self.scaler.transform(X[self.numeric_cols])
            df_numeric = pd.DataFrame(X_numeric, columns=self.numeric_cols, index=X.index)
        else:
            df_numeric = pd.DataFrame(index=X.index)

        # Use get_dummies instead of OneHotEncoder for categoricals
        if self.categorical_cols:
             df_encoded = pd.get_dummies(X[self.categorical_cols], drop_first=True)
             
             # Align with training features if transform is called on test data
             if hasattr(self, 'feature_names') and self.feature_names:
                 # Only keep dummy columns that we saw during training
                 dummy_cols = [c for c in self.feature_names if c not in self.numeric_cols]
                 df_encoded = df_encoded.reindex(columns=dummy_cols, fill_value=0)
        else:
             df_encoded = pd.DataFrame(index=X.index)

        X_final = pd.concat([df_numeric, df_encoded], axis=1)
        self.feature_names = X_final.columns.tolist()
        return X_final




if __name__ == "__main__":
    from data_loader import DataLoader
    try:
        loader = DataLoader(data_dir=os.path.join(os.path.dirname(__file__), "..", "dataset"))
        df     = loader.get_data()
        X      = df.drop('default', axis=1)
        y      = df['default']

        preprocessor = CreditRiskPreprocessor()
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)

        print("Preprocessing successful.")
        print(f"Original shape: {X.shape}")
        print(f"Processed shape: {X_processed.shape}")
        print(X_processed.head())


    except Exception as e:
        print(f"Test failed: {e}")
