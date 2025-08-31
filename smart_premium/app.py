import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# ----------------------------
# Load Local Model (Pickle)
# ----------------------------
MODEL_PATH = "XGBoost_pipeline_model.pkl"  # saved from train.py

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("✅ Local model loaded successfully")
else:
    st.error("❌ Model file not found! Please run train.py to generate the model first.")
    st.stop()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("💡 Smart Insurance Premium Prediction")
st.write("Enter customer details below to predict the insurance premium.")

# Collect user inputs (same as train.py features)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
annual_income = st.number_input("Annual Income", min_value=10000, max_value=10000000, value=500000)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
education = st.selectbox("Education Level", ["High School", "Graduate", "Postgraduate", "PhD"])
occupation = st.selectbox("Occupation", ["Salaried", "Self-Employed", "Business", "Retired", "Student"])
health_score = st.slider("Health Score", min_value=0, max_value=100, value=50)
location = st.text_input("Location (City)", "Mumbai")
policy_type = st.selectbox("Policy Type", ["Comprehensive", "Third-Party", "Critical Illness"])
smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
exercise_freq = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
property_type = st.selectbox("Property Type", ["Owned", "Rented", "Mortgaged"])
customer_feedback = st.selectbox("Customer Feedback", ["Positive", "Neutral", "Negative"])
previous_claims = st.number_input("Number of Previous Claims", min_value=0, max_value=50, value=0)
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=700)
insurance_duration = st.number_input("Insurance Duration (years)", min_value=1, max_value=30, value=5)
policy_year = st.number_input("Policy Start Year", min_value=2000, max_value=2030, value=2022)
policy_month = st.number_input("Policy Start Month", min_value=1, max_value=12, value=1)
policy_day = st.number_input("Policy Start Day", min_value=1, max_value=31, value=1)

# Convert inputs into DataFrame (columns must match train.py features)
input_data = pd.DataFrame([{
    "Age": age,
    "Annual Income": annual_income,
    "Number of Dependents": dependents,
    "Health Score": health_score,
    "Previous Claims": previous_claims,
    "Vehicle Age": vehicle_age,
    "Credit Score": credit_score,
    "Insurance Duration": insurance_duration,
    "Policy_Start_Year": policy_year,
    "Policy_Start_Month": policy_month,
    "Policy_Start_Day": policy_day,
    "Gender": gender,
    "Marital Status": marital_status,
    "Education Level": education,
    "Occupation": occupation,
    "Location": location,
    "Policy Type": policy_type,
    "Smoking Status": smoking_status,
    "Exercise Frequency": exercise_freq,
    "Property Type": property_type,
    "Customer Feedback": customer_feedback
}])

# Prediction
if st.button("Predict Premium"):
    try:
        log_pred = model.predict(input_data)[0]
        prediction = np.expm1(log_pred)  # reverse log1p used in training
        st.success(f"💰 Estimated Insurance Premium: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
