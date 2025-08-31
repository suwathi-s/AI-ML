# train.py
# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import mlflow
import mlflow.sklearn
import os

# -----------------------------
# 1. Load Raw Data
# -----------------------------
train = pd.read_csv(r"C:\Users\91902\OneDrive\Desktop\smart_premium\train.csv")
test  = pd.read_csv(r"C:\Users\91902\OneDrive\Desktop\smart_premium\test.csv")

# -----------------------------
# 2. Data Cleaning & Feature Engineering
# -----------------------------
def clean_and_engineer(df):
    # Fill missing numeric values with median
    num_cols = ["Age", "Annual Income", "Number of Dependents", 
                "Health Score", "Previous Claims", "Vehicle Age", 
                "Credit Score", "Insurance Duration"]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill missing categorical values with mode
    cat_cols = ["Gender", "Marital Status", "Education Level", "Occupation", 
                "Location", "Policy Type", "Smoking Status", 
                "Exercise Frequency", "Property Type", "Customer Feedback"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Policy Start Date → Year, Month, Day
    df['Policy_Start_Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
    df['Policy_Start_Year'] = df['Policy_Start_Date'].dt.year
    df['Policy_Start_Month'] = df['Policy_Start_Date'].dt.month
    df['Policy_Start_Day'] = df['Policy_Start_Date'].dt.day
    
    return df

train = clean_and_engineer(train)
test = clean_and_engineer(test)

# Save cleaned CSVs
train.to_csv("train_cleaned.csv", index=False)
test.to_csv("test_cleaned.csv", index=False)

# -----------------------------
# 3. Features & Target
# -----------------------------
num_cols = ["Age", "Annual Income", "Number of Dependents", 
            "Health Score", "Previous Claims", "Vehicle Age", 
            "Credit Score", "Insurance Duration",
            "Policy_Start_Year", "Policy_Start_Month", "Policy_Start_Day"]

cat_cols = ["Gender", "Marital Status", "Education Level", "Occupation", 
            "Location", "Policy Type", "Smoking Status", 
            "Exercise Frequency", "Property Type", "Customer Feedback"]

features = num_cols + cat_cols
target = "Premium Amount"

X = train[features]
y = np.log1p(train[target])  # Log transform target

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Evaluation Function
# -----------------------------
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsle = np.sqrt(mean_squared_error(y_true, y_pred))  # Already log-transformed
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "RMSLE": rmsle, "MAE": mae, "R2": r2}

# -----------------------------
# 5. Preprocessing Pipeline
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# -----------------------------
# 6. Train & Evaluate Models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, tree_method='hist')
}

results = {}
pipelines = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    metrics = evaluate_model(y_val, y_pred)
    results[name] = metrics
    pipelines[name] = pipeline
    print(f"{name} metrics: {metrics}")

# -----------------------------
# 7. Select Best Model Automatically
# -----------------------------
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_pipeline = pipelines[best_model_name]
print(f"Best model based on R²: {best_model_name}")

# Retrain on full dataset
best_pipeline.fit(X, y)

# -----------------------------
# 8. Predict on Test Set
# -----------------------------
X_test = test[features]
test_preds = np.expm1(best_pipeline.predict(X_test))  # Convert back from log
submission = pd.DataFrame({"id": test["id"], "Premium Amount": test_preds})
submission.to_csv("sample_submission.csv", index=False)

# -----------------------------
# 9. Save Trained Model
# -----------------------------
joblib.dump(best_pipeline, f"{best_model_name}_pipeline_model.pkl")

# -----------------------------
# 10. MLflow Logging (No Git Warnings)
# -----------------------------
os.environ["GIT_PYTHON_REFRESH"] = "quiet"  # Suppress Git warnings
mlflow.set_experiment("Insurance_Premium_Prediction")

with mlflow.start_run(run_name=f"{best_model_name}_Insurance"):
    mlflow.sklearn.log_model(best_pipeline, f"{best_model_name}_model", input_example=X.head(5))
    mlflow.log_metrics(evaluate_model(y, best_pipeline.predict(X)))
