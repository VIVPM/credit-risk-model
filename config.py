"""
Configuration file for Credit Risk Prediction project.
Contains all paths, constants, and settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
SRC_DIR = ROOT_DIR / "src"
BACKEND_DIR = ROOT_DIR / "backend"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA FILE NAMES
# =============================================================================
FILE_CUSTOMERS = "customers.csv"
FILE_LOANS = "loans.csv"
FILE_BUREAU = "bureau_data.csv"

# =============================================================================
# FEATURE SETTINGS
# =============================================================================
TARGET_COLUMN = "default"
ID_COLUMNS = ["cust_id", "loan_id"]

RAW_INPUT_FEATURES = [
    "age",
    "income", 
    "loan_amount",
    "loan_tenure_months",
    "avg_dpd_per_delinquency",
    "delinquency_ratio",
    "credit_utilization_ratio",
    "number_of_open_accounts",
    "residence_type",
    "loan_purpose",
    "loan_type"
]

SELECTED_FEATURES = [
    "age",
    "loan_tenure_months",
    "number_of_open_accounts",
    "credit_utilization_ratio",
    "loan_to_income",
    "delinquency_ratio",
    "avg_dpd_per_delinquency",
    "residence_type",
    "loan_purpose",
    "loan_type"
]

# Typo correction map
TYPO_CORRECTIONS = {
    "loan_purpose": {"Personaal": "Personal"}
}

# =============================================================================
# MODEL SETTINGS
# =============================================================================
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Best parameters for Logistic Regression (from Optuna tuning in notebook cell 93)
MODEL_PARAMS = {
    "C": 3.9547714598670396,
    "solver": "liblinear",
    "tol": 0.00045289166921310325,
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "max_iter": 1000,
    "n_jobs": 1  # liblinear does not support n_jobs=-1
}

# =============================================================================
# HUGGINGFACE HUB INTEGRATION
# =============================================================================
load_dotenv(BACKEND_DIR / ".env")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "your_username/credit-risk-model")
HF_FILES = [
    "model_data.joblib",
    "model_comparison.csv"
]
