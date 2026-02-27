# Three derived ratios that showed up as the strongest predictors in the notebook:
#   loan_to_income         — how leveraged the borrower is
#   delinquency_ratio      — % of loan months in default
#   avg_dpd_per_delinquency — how severe each delinquency episode was
#
# All three guard against division by zero with np.where — the denominators
# can legitimately be 0 for some applicants.

import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from config import RANDOM_STATE


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three derived ratio columns if they're not already present."""
    df = df.copy()

    if 'loan_to_income' not in df.columns:
        if 'loan_amount' in df.columns and 'income' in df.columns:
            df['loan_to_income'] = np.where(
                df['income'] != 0,
                (df['loan_amount'] / df['income']).round(2),
                0
            )

    if 'delinquency_ratio' not in df.columns:
        if 'delinquent_months' in df.columns and 'total_loan_months' in df.columns:
            df['delinquency_ratio'] = np.where(
                df['total_loan_months'] != 0,
                (df['delinquent_months'] * 100 / df['total_loan_months']).round(1),
                0
            )

    if 'avg_dpd_per_delinquency' not in df.columns:
        if 'total_dpd' in df.columns and 'delinquent_months' in df.columns:
            df['avg_dpd_per_delinquency'] = np.where(
                df['delinquent_months'] != 0,
                (df['total_dpd'] / df['delinquent_months']).round(1),
                0
            )

    return df


def apply_resampling(X: pd.DataFrame, y: pd.Series):
    """
    Undersample the majority class (non-default) to match minority (default).
    We want high recall on defaults — undersampling is simpler and faster than SMOTE
    for this dataset size and gave better recall in the notebook experiments.
    """
    print("Applying RandomUnderSampler to handle class imbalance...")
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    print(f"Resampled X shape: {X_resampled.shape}")
    print(f"Resampled y distribution:\n{y_resampled.value_counts()}")
    return X_resampled, y_resampled
