"""
Training Module.
Handles model training using Logistic Regression.
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
from config import MODEL_PARAMS, RANDOM_STATE

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train the Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("Training Logistic Regression Model...")
    print(f"Parameters: {MODEL_PARAMS}")
    
    model = LogisticRegression(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    print("Training complete.")
    return model

