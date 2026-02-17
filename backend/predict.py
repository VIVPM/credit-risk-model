import pandas as pd
import joblib
import numpy as np
import os

class CreditRiskModel:
    """
    Wrapper for loading the trained model and making predictions.
    Uses the reference implementation logic to ensure consistent results.
    """
    def __init__(self, model_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Point to the reference artifact provided by the user
        self.model_path = os.path.join(base_dir, "credit-risk-model", "artifacts", "model_data.joblib")
        
        self.model = None
        self.scaler = None
        self.features = None
        self.cols_to_scale = None
        
        self.load()

    def load(self):
        """Loads model and preprocessor artifacts from the reference joblib."""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
            self.cols_to_scale = model_data['cols_to_scale']
        except FileNotFoundError:
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")

    def prepare_input(self, row):
        """
        Prepares input data matching the reference logic.
        """
        input_data = {
            'age': row['age'],
            'loan_tenure_months': row['loan_tenure_months'],
            'number_of_open_accounts': row['number_of_open_accounts'],
            'credit_utilization_ratio': row['credit_utilization_ratio'],
            'loan_to_income': row['loan_amount'] / row['income'] if row['income'] > 0 else 0,
            'delinquency_ratio': row['delinquency_ratio'],
            'avg_dpd_per_delinquency': row['avg_dpd_per_delinquency'],
            'residence_type_Owned': 1 if row['residence_type'] == 'Owned' else 0,
            'residence_type_Rented': 1 if row['residence_type'] == 'Rented' else 0,
            'loan_purpose_Education': 1 if row['loan_purpose'] == 'Education' else 0,
            'loan_purpose_Home': 1 if row['loan_purpose'] == 'Home' else 0,
            'loan_purpose_Personal': 1 if row['loan_purpose'] == 'Personal' else 0,
            'loan_type_Unsecured': 1 if row['loan_type'] == 'Unsecured' else 0,
            # Dummy fields required for scaling (as per reference implementation)
            'number_of_dependants': 1,
            'years_at_current_address': 1,
            'zipcode': 1,
            'sanction_amount': 1,
            'processing_fee': 1,
            'gst': 1,
            'net_disbursement': 1,
            'principal_outstanding': 1,
            'bank_balance_at_application': 1,
            'number_of_closed_accounts': 1,
            'enquiry_count': 1
        }
        
        df = pd.DataFrame([input_data])
        df[self.cols_to_scale] = self.scaler.transform(df[self.cols_to_scale])
        df = df[self.features]
        return df

    def calculate_credit_score(self, input_df):
        """
        Calculates credit score and rating based on model coefficients.
        """
        x = np.dot(input_df.values, self.model.coef_.T) + self.model.intercept_
        default_probability = 1 / (1 + np.exp(-x))
        non_default_probability = 1 - default_probability
        
        base_score = 300
        scale_length = 600
        credit_score = int(base_score + (non_default_probability.flatten()[0] * scale_length))
        
        if 300 <= credit_score < 500:
            rating = 'Poor'
        elif 500 <= credit_score < 650:
            rating = 'Average'
        elif 650 <= credit_score < 750:
            rating = 'Good'
        elif 750 <= credit_score <= 900:
            rating = 'Excellent'
        else:
            rating = 'Undefined'
            
        return default_probability.flatten()[0], credit_score, rating

    def predict(self, data: pd.DataFrame):
        """
        Predicts default probability, credit score, and rating.
        Args:
            data (pd.DataFrame): Raw input data.
        Returns:
            list of dicts: Prediction results.
        """
        results = []
        for _, row in data.iterrows():
            input_df = self.prepare_input(row)
            probability, credit_score, rating = self.calculate_credit_score(input_df)
            
            # Prediction threshold standard is 0.5
            prediction = 1 if probability > 0.5 else 0
            
            results.append({
                "default_prediction": prediction,
                "default_probability": float(probability),
                "credit_score": credit_score,
                "rating": rating
            })
            
        return results

if __name__ == "__main__":
    try:
        model = CreditRiskModel()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error: {e}")
