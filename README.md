# Credit Risk Modeling & MLOps Pipeline

A production-grade machine learning project for assessing credit risk, featuring a modular training pipeline, FastAPI backend, and Streamlit dashboard.

## 📊 Project Overview

This application predicts the **Probability of Default (PD)** for loan applicants and assigns a **Credit Score (300-900)** and **Risk Rating**. It is designed to be a complete MLOps solution, moving from a monolithic notebook to a structured, deployable package.

## 🏗️ Architecture

The project follows a modular 3-tier architecture:

1.  **Data & Training Pipeline (`src/`)**:
    *   **Data Injection**: Loads raw data from SQL/CSV sources.
    *   **Preprocessing**: Handles missing values, outliers, and feature scaling/encoding.
    *   **Feature Engineering**: Creates derived ratios like `Loan-to-Income`, `Delinquency Ratio`.
    *   **Model Training**: Trains Logistic Regression/XGBoost models with Class Imbalance handling (SMOTE/RandomUnderSampler).
2.  **Inference Engine (`backend/`)**:
    *   **Model Serving**: Validates inputs and runs predictions.
    *   **Scoring Logic**: Converts raw probabilities into industry-standard Credit Scores.
    *   **API**: FastAPI endpoints for real-time integration.
3.  **User Interface (`streamlit_app.py`)**:
    *   **Interactive Dashboard**: Allows loan officers to input applicant details.
    *   **Batch Processing**: Supports CSV uploads for bulk scoring.

## 📈 Model Performance & Metrics

The model evaluation focused on maximizing **Recall** for the "Default" class (Class 1) to minimize financial risk (missing a defaulter is costly).

### Classification Report (Logistic Regression)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Non-Default (0)** | 0.99 | 0.90 | 0.94 | 8553 |
| **Default (1)** | **0.34** | **0.90** | **0.49** | 1447 |

*   **Accuracy**: 90%
*   **Recall (Default)**: **90%** (Key Metric: We catch 90% of potential defaulters)
*   **ROC-AUC**: 0.94

*Note: The precision for defaults is lower (34%), implying some false alarms, which is acceptable in credit risk to prioritize safety.*

## 🚀 Automation Pipeline

The training process is fully automated via `src/train.py`:

1.  **`src/data_loader.py`**: Merges Customer, Loan, and Bureau datasets.
2.  **`src/feature_engineering.py`**:
    *   Calculates `loan_to_income = loan_amount / income`
    *   Calculates `delinquency_ratio = delinquent_months / total_loan_months`
3.  **`src/preprocessing.py`**:
    *   One-Hot Encoding for Categorical variables.
    *   MinMax Scaling for Numerical variables.
4.  **`src/train.py`**:
    *   Splits data (Stratified).
    *   Applies `SMOTETomek` for class balancing.
    *   Optimizes Hyperparameters (C, Solver).
    *   Saves artifacts to `models/`.

## 📂 Project Structure

```
credit-risk-model/
├── backend/                # Inference Engine
│   ├── api.py              # FastAPI Server
│   └── predict.py          # Prediction Logic & Credit Scoring
├── src/                    # Training Pipeline
│   ├── data_loader.py      # Data Ingestion
│   ├── preprocessing.py    # Cleaning & Scaling
│   ├── feature_engineering.py # Feature Creation
│   └── train.py            # Model Training Script
├── models/                 # Binary Artifacts
│   ├── logistic_regression_model.joblib
│   └── preprocessor.joblib
├── streamlit_app.py        # Frontend Dashboard
├── config.py               # Central Configuration
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## 💻 Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd credit-risk-model
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**:
    ```bash
    streamlit run streamlit_app.py
    ```
    Access at `http://localhost:8501`.

4.  **Run the API**:
    ```bash
    python backend/api.py
    ```
    API documentation available at `http://localhost:8000/docs`.

## ⚙️ Credit Scoring Logic

The probability of default ($P_{default}$) is converted to a score (300-900):

$$ Score = 300 + (1 - P_{default}) \times 600 $$

| Score Range | Rating | Risk Decision |
|-------------|--------|---------------|
| 300 - 499 | Poor | 🔴 High Risk |
| 500 - 649 | Average | 🟠 Medium Risk |
| 650 - 749 | Good | 🟡 Low Risk |
| 750 - 900 | Excellent | 🟢 Approved |

## 📜 License

MIT License