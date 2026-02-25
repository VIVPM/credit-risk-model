import streamlit as st
import pandas as pd
import sys
import os

# Add backend to path (optional if running from root, but good for local)
# sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from backend.predict import CreditRiskModel

import requests

# Set API URL from:
# 1. secrets.toml (for local secure dev) or Streamlit Cloud Secrets
# 2. Environment Variable (for Render/Docker)
# 3. None (Local Standalone Mode)

if "API_URL" in st.secrets:
    API_URL = st.secrets["API_URL"]
else:
    API_URL = None

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

# Inject Custom CSS for Status Cards
st.markdown("""
<style>
.status-running {
    background-color: #0d47a1;
    color: white;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 10px;
    font-size: 1.1em;
}
.status-completed {
    background-color: #1b5e20;
    color: white;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 10px;
    font-size: 1.1em;
}
.status-failed {
    background-color: #b71c1c;
    color: white;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 10px;
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Credit Risk Assessment Dashboard")

# ---------------------------------------------------------------------------
# Helper: Check API connection
# ---------------------------------------------------------------------------
def check_api():
    if not API_URL:
        return False, {}
    try:
        r = requests.get(f"{API_URL}/", timeout=15)
        return r.status_code == 200, r.json()
    except (requests.ConnectionError, requests.Timeout):
        return False, {}

def get_model_versions():
    if not API_URL:
        return []
    try:
        r = requests.get(f"{API_URL}/model/versions", timeout=10)
        if r.status_code == 200:
            return r.json().get("versions", [])
    except Exception:
        pass
    return []

def get_model_info(version="main"):
    if not API_URL:
        return None
    try:
        r = requests.get(f"{API_URL}/model/info?version={version}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏦 Credit Risk Predictor")
    st.markdown("---")

    api_ok, root_data = check_api()
    model_loaded = root_data.get("status") == "active"

    if not API_URL:
        st.warning("⚠️ Running in Standalone Mode. Remote API is disconnected.")
        st.session_state["selected_version"] = "local"
    elif api_ok:
        st.success("✅ API Connected")
        
        # Version Selection
        versions = get_model_versions()
        
        if not versions:
             st.warning("⚠️ No model versions found on HF Hub. Train a model first.")
             st.session_state["selected_version"] = "local"
        else:
             selected_version = st.selectbox(
                 "📂 Select Model Version",
                 options=reversed(versions), # Show newest first
                 index=0
             )
             st.session_state["selected_version"] = selected_version

             info = get_model_info(selected_version)
             if info:
                 st.markdown(f"**Model:** `{info.get('model_name', 'Loaded')}`")
                 st.markdown(f"**Features:** `{info.get('num_features', 'Unknown')}`")
                 st.markdown(f"**Version:** `{selected_version}`")
             else:
                 st.warning("⚠️ Model not loaded yet — train first")
    else:
        st.error("❌ API Offline")
        st.code("cd backend && uvicorn api:app --reload", language="bash")

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. **Train Model** — upload 3 CSV datasets
    2. **Single Prediction** — fill form
    3. **Batch Prediction** — upload combined CSV
    """)

# Initialize Model (Local Mode only)
# Only load local model if API_URL is NOT set
@st.cache_resource
def load_model():
    return CreditRiskModel()

if not API_URL:
    try:
        model = load_model()
        st.success("Loaded Local Model (Standalone Mode)")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.info("Connected to Remote API Server")

# ====================== TABS ======================
tab_train, tab_single, tab_batch = st.tabs(["⚙️ Train Model", "📋 Single Prediction", "📁 Batch Prediction"])

# ====================== SINGLE PREDICTION ======================
with tab_single:
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
                if API_URL:
                    # Remote API Call
                    # Convert to list of dicts (records) which API expects
                    payload = df_input.to_dict(orient="records")
                    version = st.session_state.get("selected_version", "main")
                    response = requests.post(f"{API_URL}/predict?version={version}", json={"data": payload})
                    if response.status_code == 200:
                        result = response.json()["results"][0]
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                        st.stop()
                else:
                    # Local Function Call
                    df_input = pd.DataFrame(input_data)
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
with tab_batch:
    st.subheader("Upload Customer Data for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Predict Credit Risk", key="batch_predict"):
            with st.spinner("Analyzing..."):
                try:
                    if API_URL:
                        # Convert Dataframe to list of dicts for API
                        payload = {"data": df.to_dict(orient="records")}
                        version = st.session_state.get("selected_version", "main")
                        response = requests.post(f"{API_URL}/predict?version={version}", json=payload)
                        if response.status_code == 200:
                            predictions = response.json()["results"]
                        else:
                            st.error(f"API Error: {response.text}")
                            st.stop()
                    else:
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

# ====================== TRAIN MODEL ======================
with tab_train:
    st.subheader("⚙️ Train Credit Risk Model")
    st.write("Upload the three required datasets (`customers.csv`, `loans.csv`, and `bureau_data.csv`) to trigger background training and upload the newly trained model to HuggingFace Hub.")

    if not API_URL:
        st.warning("⚠️ Training is only available when connected to the remote API Server. You are currently in Standalone Mode.")
    else:
        uploaded_files = st.file_uploader(
            "Upload customers.csv, loans.csv, and bureau_data.csv", 
            type=["csv"], 
            accept_multiple_files=True
        )
        
        # Initialize state
        if "training_triggered" not in st.session_state:
            st.session_state["training_triggered"] = False
            
        train_btn = st.button("🚀 Start Training", type="primary", use_container_width=True)
        
        if train_btn:
            if not uploaded_files or len(uploaded_files) < 3:
                st.error("Please upload all 3 CSV files (customers.csv, loans.csv, bureau_data.csv) before starting training.")
            else:
                with st.spinner("Triggering training..."):
                    try:
                        files_payload = [
                            ("files", (f.name, f.getvalue(), "text/csv")) 
                            for f in uploaded_files
                        ]
                        resp = requests.post(f"{API_URL}/train", files=files_payload, timeout=30)
                        if resp.status_code == 200:
                            st.session_state["training_triggered"] = True
                        elif resp.status_code == 409:
                            st.warning("⚠️ Training already in progress.")
                            st.session_state["training_triggered"] = True
                        else:
                            st.error(f"❌ Failed to start training: {resp.text}")
                    except Exception as e:
                        st.error(f"❌ API connection failed: {e}")

        # Status Tracking
        st.markdown("---")
        
        # Placeholders for dynamic updates
        status_ph = st.empty()
        msg_ph = st.empty()
        metrics_ph = st.empty()
        refresh_ph = st.empty()
        
        s = {}
        status_val = "idle"
        
        # First, check if there's an active or recently completed training in this session
        if st.session_state.get("training_triggered", False):
            try:
                r = requests.get(f"{API_URL}/train/status", timeout=10)
                if r.status_code == 200:
                    s = r.json()
                    status_val = s.get("status", "idle")
            except Exception:
                pass

        # If no active training is going on, check if a model version is loaded/selected
        if status_val == "idle":
            selected = st.session_state.get("selected_version")
            if selected and selected != "local":
                info = get_model_info(selected)
                if info:
                    status_val = "loaded_from_hf"
                    s = {
                        "message": f"Viewing model details for {selected} from Hugging Face Hub.",
                        "model_name": info.get("model_name", "Logistic Regression"),
                        "accuracy": info.get("accuracy_score"),
                        "num_features": info.get("num_features")
                    }
        
        if s and status_val != "idle":
            msg_val = s.get("message", "")
            err_val = s.get("error", "")
            
            if status_val == "running":
                status_ph.markdown(
                    f'<div class="status-running">🔄 <strong>Training in Progress</strong><br>{msg_val}</div>',
                    unsafe_allow_html=True
                )
                
                # Step indicators logic
                msg_lower = msg_val.lower()
                step1 = "✅" if "step 2" in msg_lower or "step 3" in msg_lower or "complete" in msg_lower else ("🔄" if "step 1" in msg_lower else "⏳")
                step2 = "✅" if "step 3" in msg_lower or "complete" in msg_lower else ("🔄" if "step 2" in msg_lower else "⏳")
                step3 = "✅" if "complete" in msg_lower else ("🔄" if "step 3" in msg_lower else "⏳")

                msg_ph.markdown(f"""
                | Step | Task | Status |
                |------|------|--------|
                | 1 | Loading Data | {step1} |
                | 2 | Preprocessing | {step2} |
                | 3 | Training Logistic Regression | {step3} |
                """)
                
            elif status_val in ("completed", "loaded_from_hf"):
                title = "Training Completed!" if status_val == "completed" else "Model Loaded"
                status_ph.markdown(
                    f'<div class="status-completed">✅ <strong>{title}</strong><br>{msg_val}</div>',
                    unsafe_allow_html=True
                )
                msg_ph.markdown("""
                #### 📊 Training Status
                | Step | Task | Status |
                |------|------|--------|
                | 1 | Loading Data | ✅ |
                | 2 | Preprocessing | ✅ |
                | 3 | Training Logistic Regression | ✅ |
                """)
                m1, m2, m3 = metrics_ph.columns(3)
                
                # 1. Best Model
                model_name = s.get("model_name")
                if not model_name or model_name == "-":
                    model_name = "Logistic Regression"
                m1.metric("🏆 Best Model", model_name)
                
                # 2. Score
                acc = s.get("accuracy")
                m2.metric("📈 Accuracy Score", f"{acc:.4f}" if acc else "Not calculated")
                
                # 3. Features Used
                feats = s.get("num_features")
                if not feats or feats == "-":
                    feats = "Wait..."
                m3.metric("🔢 Features Used", feats)
            elif status_val == "failed":
                status_ph.markdown(
                    f'<div class="status-failed">❌ <strong>Training Failed</strong><br>{msg_val}</div>',
                    unsafe_allow_html=True
                )
                if err_val:
                    with st.expander("View Error Details"):
                        st.code(err_val)
                        
            # Manual refresh button
            if status_val == "running":
                with refresh_ph.container():
                    st.write("⏳ Training running in background — click Refresh to check progress.")
                    if st.button("🔄 Refresh Status"):
                        st.rerun()
            elif status_val in ("completed", "failed", "loaded_from_hf"):
                 with refresh_ph.container():
                    if st.button("🔄 Refresh Status"):
                        st.rerun()
                        
        else:
            status_ph.info("💤 **Idle** - No models trained yet. Upload 3 datasets and hit Start Training.")
