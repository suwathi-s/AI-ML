#  Smart Insurance Premium Prediction

This project predicts insurance premiums for customers based on their demographic, financial, and lifestyle details.  
It uses **Machine Learning (XGBoost, Random Forest, Decision Tree, Linear Regression)** and is deployed as a **Streamlit web app**.

---

##  Features
- Data preprocessing & feature engineering (handling missing values, encoding categorical features, date extraction).
- Multiple ML models trained & evaluated with metrics (RMSE, MAE, R²).
- Automatic best-model selection (XGBoost by default).
- Model tracking using **MLflow**.
- Interactive **Streamlit UI** for real-time predictions.
- Deployment on **Streamlit Cloud**.

---

## 📂 Project Structure
smart_premium/
│
├── app.py # Streamlit app for UI & predictions
├── train.py # Model training, evaluation, MLflow logging
├── insurance_premium_prediction.ipynb # EDA & visualization
│
├── train.csv # Training dataset
├── test.csv # Test dataset
├── sample_submission.csv # Submission format (and predictions saved here)
├── train_cleaned.csv # Cleaned training dataset
├── test_cleaned.csv # Cleaned test dataset
│
├── XGBoost_pipeline_model.pkl # Saved best ML model (used by Streamlit)
│
├── mlruns/ # MLflow experiment tracking
│ └── models/insurance_premium_model/
│ └── version-1/
│ └── meta.yaml
│
├── requirements.txt # Dependencies for running project
└── README.md # Project documentation

## Usage
1️⃣ Train Models

Run:

python train.py

This will:

Clean & preprocess the datasets

Train multiple ML models

Select the best model (XGBoost usually)

Save the trained pipeline as XGBoost_pipeline_model.pkl

Log results & model in MLflow

2️⃣ Run Streamlit App

streamlit run app.py

The app will open in your browser and allow you to enter customer details for premium prediction.

3️⃣ Deploy on Streamlit Cloud

Push repo to GitHub

Go to Streamlit Cloud

Select repo & point to app.py

Done! 

 ## Live Demo

👉 https://vutyjqhyhasg8i8lrqp3wg.streamlit.app/

## Example Input (via Streamlit UI)

Age : 30

Gender : Male

Annual Income : 500000

Marital Status : Single

Number of Dependents : 0

Education Level : High School

Occupation : Salaried

Health Score : 75

Location (City) : Mumbai

Policy Type : Comprehensive

Smoking Status : Yes

Exercise Frequency : Daily

Property Type : Owned

Customer Feedback : Positive

Number of Previous Claims : 0

Vehicle Age (years) : 5

Credit Score : 400

Insurance Duration (years) : 10

Policy Start Year : 2022

Policy Start Month : 1

Policy Start Day : 1

💰 Estimated Insurance Premium: ₹ 618.69 its my output

## Requirements

Key libraries:

pandas

numpy

scikit-learn

xgboost

matplotlib

seaborn

mlflow

streamlit

joblib

## 👨‍💻 Author
Suwathi S


