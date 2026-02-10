# Credit Risk Modeling Application

A machine learning-powered web application that predicts loan default probability and generates credit scores for borrowers.

## Overview

This application uses a logistic regression model to assess credit risk based on borrower characteristics and loan details. It outputs a default probability, credit score (300-900 scale), and risk rating.

## Features

- **Default Probability Prediction**: Estimates the likelihood of loan default
- **Credit Score Generation**: Converts probability to a 300-900 credit score
- **Risk Rating**: Categorizes borrowers as Poor, Average, Good, or Excellent
- **Interactive UI**: Streamlit-based interface for real-time predictions

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd credit-risk-modeling

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- joblib

## Usage

```bash
streamlit run main.py
```

Navigate to `http://localhost:8501` in your browser.

## Input Parameters

| Parameter | Description |
|-----------|-------------|
| Age | Borrower's age (18-100) |
| Income | Annual income |
| Loan Amount | Requested loan amount |
| Loan Tenure | Loan duration in months |
| Avg DPD | Average days past due per delinquency |
| Delinquency Ratio | Percentage of delinquent accounts |
| Credit Utilization Ratio | Percentage of available credit used |
| Open Loan Accounts | Number of active loan accounts (1-4) |
| Residence Type | Owned / Rented / Mortgage |
| Loan Purpose | Education / Home / Auto / Personal |
| Loan Type | Secured / Unsecured |

## Credit Score Mapping

| Score Range | Rating |
|-------------|--------|
| 300 - 499 | Poor |
| 500 - 649 | Average |
| 650 - 749 | Good |
| 750 - 900 | Excellent |

## Project Structure

```
├── main.py                 # Streamlit application
├── prediction_helper.py    # Model inference and scoring logic
├── artifacts/
│   └── model_data.joblib   # Trained model, scaler, and feature config
├── requirements.txt
└── README.md
```

## How It Works

1. User inputs borrower and loan details via the Streamlit interface
2. Input data is preprocessed and scaled using the fitted MinMaxScaler
3. Logistic regression model computes default probability
4. Probability is transformed to a credit score using: `score = 300 + (1 - default_prob) × 600`
5. Score is mapped to a risk rating category

## License

MIT