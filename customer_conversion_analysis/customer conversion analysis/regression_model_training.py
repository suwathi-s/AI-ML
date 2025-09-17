import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------
train_path = r"C:\Users\91902\OneDrive\Desktop\customer conversion analysis\train_features.csv"
test_path  = r"C:\Users\91902\OneDrive\Desktop\customer conversion analysis\test_features.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# Target is session-level revenue proxy: price
y = train["price"]
X = train.drop(columns=["price"])

# ---------------------------------------------------------------------
# 2. Train/Validation Split
# ---------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# 3. Candidate Models
# ---------------------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
    )
}

results = {}

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# ---------------------------------------------------------------------
# 4. Train & Validate
# ---------------------------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    results[name] = evaluate(y_val, preds)

# Show validation results
print("\n=== Validation Metrics ===")
for name, metrics in results.items():
    print(f"\n{name}:")
    for m, v in metrics.items():
        print(f"  {m}: {v:.4f}")

# ---------------------------------------------------------------------
# 5. Pick Best (lowest RMSE)
# ---------------------------------------------------------------------
best_model_name = min(results, key=lambda k: results[k]["RMSE"])
best_model = models[best_model_name]
print(f"\nBest model based on RMSE: {best_model_name}")

# Save it
joblib.dump(best_model, "best_regressor.pkl")
print("Saved best model -> best_regressor.pkl")

# ---------------------------------------------------------------------
# 6. Final Test Evaluation
# ---------------------------------------------------------------------
X_test = test.drop(columns=["price"])
y_test = test["price"]

y_pred = best_model.predict(X_test)
final_metrics = evaluate(y_test, y_pred)

print("\n=== Final Test Metrics ===")
for m, v in final_metrics.items():
    print(f"{m}: {v:.4f}")
