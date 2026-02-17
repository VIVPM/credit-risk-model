"""
Evaluation Module.
Handles model evaluation metrics and reporting.
"""

from sklearn.metrics import classification_report
import pandas as pd
from typing import Any

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    """
    Evaluate the model on test data and return classification report.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        
    Returns:
        Classification report string
    """
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred)
    
    print("\n=== Classification Report ===")
    print(report)
    print("=============================")
    
    return report
