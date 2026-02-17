import streamlit as st
import pandas as pd
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from predict import CreditRiskModel

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

st.title("🏦 Credit Risk Assessment Dashboard")

# Initialize Model
@st.cache_resource
def load_model():
    return CreditRiskModel()

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ====================== TABS ======================
tab1, tab2 = st.tabs(["📋 Single Prediction", "📁 Batch Prediction"])

# ====================== SINGLE PREDICTION ======================
with tab1:
    st.subheader("Enter Customer Details")
    
    with st.form("single_prediction_form"):
        # Match inputs from credit-risk-model/main.py
        
        # Row 1: Age, Income, Loan Amount
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, step=1, value=28)
        with col2:
            income = st.number_input("Income", min_value=0, value=1200000)
        with col3:
            loan_amount = st.number_input("Loan Amount", min_value=0, value=2560000)
            
        # Display calculated Loan to Income for reference (like in main.py)
        loan_to_income_val = loan_amount / income if income > 0 else 0
        st.write(f"**Loan to Income Ratio:** {loan_to_income_val:.2f}")

        # Row 2: Tenure, Avg DPD
        col4, col5 = st.columns(2)
        with col4:
            loan_tenure_months = st.number_input("Loan Tenure (months)", min_value=0, step=1, value=36)
        with col5:
            avg_dpd_per_delinquency = st.number_input("Avg DPD", min_value=0, value=20)
            
        # Row 3: Delinquency Ratio, Credit Util, Open Accounts
        col6, col7, col8 = st.columns(3)
        with col6:
            delinquency_ratio = st.number_input("Delinquency Ratio", min_value=0, max_value=100, step=1, value=30)
        with col7:
             credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0, max_value=100, step=1, value=30)
        with col8:
             number_of_open_accounts = st.number_input("Open Loan Accounts", min_value=1, max_value=4, value=2)

        # Row 4: Residence, Purpose, Type
        col9, col10, col11 = st.columns(3)
        with col9:
            residence_type = st.selectbox("Residence Type", ["Owned", "Rented", "Mortgage"])
        with col10:
            loan_purpose = st.selectbox("Loan Purpose", ["Education", "Home", "Auto", "Personal"])
        with col11:
            loan_type = st.selectbox("Loan Type", ["Unsecured", "Secured"])

        submitted = st.form_submit_button("🔍 Predict Credit Risk", use_container_width=True)

    if submitted:
        # Build input DataFrame
        input_data = {
            "age": [age],
            "income": [income],
            "loan_amount": [loan_amount],
            "loan_tenure_months": [loan_tenure_months],
            "avg_dpd_per_delinquency": [avg_dpd_per_delinquency],
            "delinquency_ratio": [delinquency_ratio],
            "credit_utilization_ratio": [credit_utilization_ratio],
            "number_of_open_accounts": [number_of_open_accounts],
            "residence_type": [residence_type],
            "loan_purpose": [loan_purpose],
            "loan_type": [loan_type]
        }
        df_input = pd.DataFrame(input_data)

        with st.spinner("Analyzing credit risk..."):
            try:
                result = model.predict(df_input)[0]
                prediction = result["default_prediction"]
                probability = result["default_probability"]
                credit_score = result.get("credit_score", "N/A")
                rating = result.get("rating", "N/A")

                st.markdown("---")
                st.subheader("Prediction Result")

                col_r1, col_r2, col_r3 = st.columns(3)
                
                with col_r1:
                    if prediction == 1:
                        st.error(f"⚠️ **HIGH RISK**")
                    else:
                        st.success(f"✅ **LOW RISK**")
                    
                    if prediction == 1:
                        st.caption("Likely to Default")
                    else:
                        st.caption("Unlikely to Default")
                        
                with col_r2:
                    st.metric("Default Probability", f"{probability:.2%}")
                
                with col_r3:
                    st.metric("Credit Score", f"{credit_score}")
                    st.caption(f"Rating: {rating}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ====================== BATCH PREDICTION ======================
with tab2:
    st.subheader("Upload Customer Data for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Predict Credit Risk", key="batch_predict"):
            with st.spinner("Analyzing..."):
                try:
                    predictions = model.predict(df)

                    results_df = df.copy()
                    results_df["Default Prediction"] = [p['default_prediction'] for p in predictions]
                    results_df["Default Probability"] = [p['default_probability'] for p in predictions]
                    results_df["Credit Score"] = [p.get('credit_score', 'N/A') for p in predictions]
                    results_df["Rating"] = [p.get('rating', 'N/A') for p in predictions]
                    results_df["Risk Status"] = results_df["Default Prediction"].map({0: "Low Risk", 1: "High Risk"})

                    st.write("### Prediction Results")

                    def highlight_risk(val):
                        color = 'red' if val == 'High Risk' else 'green'
                        return f'background-color: {color}; color: white'

                    st.dataframe(
                        results_df.style.applymap(highlight_risk, subset=['Risk Status']),
                        use_container_width=True
                    )

                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results",
                        csv,
                        "credit_risk_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
