"""
Feature Engineering Module.
Handles feature creation, selection, and resampling (SMOTETomek).
"""

import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from config import RANDOM_STATE

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features from raw input data.
    
    Args:
        df: Input DataFrame with raw columns.
        
    Returns:
        DataFrame with added derived features.
    """
    df = df.copy()
    
    # 1. Loan to Income Ratio
    if 'loan_amount' in df.columns and 'income' in df.columns:
        # Handle potential zero division for income (though unlikely for valid loan apps)
        df['loan_to_income'] = np.where(
            df['income'] != 0,
            (df['loan_amount'] / df['income']).round(2),
            0
        )
        
    # 2. Delinquency Ratio
    if 'delinquent_months' in df.columns and 'total_loan_months' in df.columns:
        df['delinquency_ratio'] = np.where(
            df['total_loan_months'] != 0,
            (df['delinquent_months'] * 100 / df['total_loan_months']).round(1),
            0
        )

    # 3. Avg DPD per Delinquency
    if 'total_dpd' in df.columns and 'delinquent_months' in df.columns:
        df['avg_dpd_per_delinquency'] = np.where(
            df['delinquent_months'] != 0,
            (df['total_dpd'] / df['delinquent_months']).round(1),
            0
        )
        
    return df

def apply_resampling(X: pd.DataFrame, y: pd.Series):
    """
    Apply RandomUnderSampler to handle class imbalance.
    Undersamples the majority class to match the minority class size.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        X_resampled, y_resampled
    """
    print("Applying RandomUnderSampler to handle class imbalance...")
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    print(f"Resampled X shape: {X_resampled.shape}")
    print(f"Resampled y distribution:\n{y_resampled.value_counts()}")
    
    return X_resampled, y_resampled

