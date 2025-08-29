import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load trained pipeline
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\91902\OneDrive\Desktop\smart_premium\xgb_pipeline_model.pkl")

pipeline = load_model()

# ----------------------------
# App Layout
# ----------------------------
st.set_page_config(page_title="Insurance Premium Predictor", page_icon="💰", layout="wide")
st.title("💰 Insurance Premium Prediction App")
st.markdown("This app predicts **insurance premiums** based on customer details. "
            "It uses a trained **XGBoost model with preprocessing pipeline**.")

st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
- Built with **Streamlit + XGBoost**
- Features: Demographics, Financials, Health, Policy
- Model trained with log-transformed target
""")

# ----------------------------
# Input Form
# ----------------------------
with st.form("prediction_form"):
    st.subheader("Enter Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)

    with col2:
        income = st.number_input("Annual Income (₹)", min_value=10000, max_value=5000000, step=1000)
        education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
        occupation = st.selectbox("Occupation", ["Salaried", "Self-Employed", "Business", "Retired", "Unemployed"])
        location = st.selectbox("Location", ["Urban", "Semi-Urban", "Rural"])

    with col3:
        health_score = st.slider("Health Score", min_value=0, max_value=100, value=70)
        policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
        previous_claims = st.number_input("Previous Claims", min_value=0, max_value=20, value=0)
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=650)
        duration = st.number_input("Insurance Duration (years)", min_value=1, max_value=30, value=5)

    # Submit Button
    submitted = st.form_submit_button("🔮 Predict Premium")

# ----------------------------
# Prediction
# ----------------------------
if submitted:
    # Prepare input as DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Annual Income": [income],
        "Marital Status": [marital_status],
        "Number of Dependents": [dependents],
        "Education Level": [education],
        "Occupation": [occupation],
        "Health Score": [health_score],
        "Location": [location],
        "Policy Type": [policy_type],
        "Previous Claims": [previous_claims],
        "Vehicle Age": [vehicle_age],
        "Credit Score": [credit_score],
        "Insurance Duration": [duration]
    })

    # Predict
    raw_pred = pipeline.predict(input_data)[0]
    predicted_premium = np.expm1(raw_pred)  # inverse of log1p

    st.success(f"✅ Predicted Annual Insurance Premium: **₹{predicted_premium:,.2f}**")

    # Show input summary
    with st.expander("📋 Input Summary"):
        st.write(input_data)
