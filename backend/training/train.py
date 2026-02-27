# Logistic Regression with Optuna-tuned hyperparameters from the notebook.
# Model params (C, tol, solver, etc.) live in config.py — don't hardcode them here.

from sklearn.linear_model import LogisticRegression
import pandas as pd
from config import MODEL_PARAMS, RANDOM_STATE


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train logistic regression with the params from config.
    After undersampling the training set is roughly balanced, so
    no class_weight adjustment is needed here.
    """
    print("Training Logistic Regression Model...")
    print(f"Parameters: {MODEL_PARAMS}")

    model = LogisticRegression(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    print("Training complete.")
    return model
