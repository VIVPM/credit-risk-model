# Cleans, scales, and encodes the merged DataFrame.
# OneHotEncoder with drop='first' matches what the notebook used (pd.get_dummies(drop_first=True)).
# MinMaxScaler on numerics keeps everything in [0,1] which helps logistic regression converge.
# Fit only on train data — transform is applied separately to avoid leakage.

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TYPO_CORRECTIONS, SELECTED_FEATURES
from backend.training.utils import save_joblib, load_joblib


class CreditRiskPreprocessor:
    def __init__(self):
        self.scaler  = MinMaxScaler()
        # drop='first' avoids the dummy variable trap, matching the notebook.
        # sparse_output=False so we can rebuild a proper DataFrame from the output.
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='error')
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
        if self.categorical_cols:
            self.encoder.fit(X[self.categorical_cols])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numerics, one-hot encode categoricals, concatenate into a single DataFrame."""
        X = self.clean_data(X)

        if self.numeric_cols:
            X_numeric  = self.scaler.transform(X[self.numeric_cols])
            df_numeric = pd.DataFrame(X_numeric, columns=self.numeric_cols, index=X.index)
        else:
            df_numeric = pd.DataFrame(index=X.index)

        if self.categorical_cols:
            X_encoded  = self.encoder.transform(X[self.categorical_cols])
            enc_cols   = self.encoder.get_feature_names_out(self.categorical_cols)
            df_encoded = pd.DataFrame(X_encoded, columns=enc_cols, index=X.index)
        else:
            df_encoded = pd.DataFrame(index=X.index)

        X_final = pd.concat([df_numeric, df_encoded], axis=1)
        self.feature_names = X_final.columns.tolist()
        return X_final

    def save(self, filepath):
        """Persist the fitted preprocessor so inference uses the same scalings."""
        save_joblib(self, filepath)

    @staticmethod
    def load(filepath):
        """Load a previously saved preprocessor."""
        return load_joblib(filepath)


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

        preprocessor.save("models/preprocessor.joblib")
        print("Preprocessor saved.")
    except Exception as e:
        print(f"Test failed: {e}")
