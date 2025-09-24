
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -----------------------------
# 1. Load Data
# -----------------------------
data = pd.read_csv(r"C:\Users\91902\OneDrive\Desktop\customer conversion analysis\train_data.csv")

# Quick look
print("Initial shape:", data.shape)
print("Columns:", data.columns.tolist())
print(data.head())

# -----------------------------
# 2. Basic Cleaning
# -----------------------------
# Remove duplicate rows if any
data.drop_duplicates(inplace=True)

# Optional: Rename columns to remove spaces or special chars
data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]

X = data.copy()

# Identify column types
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)

# -----------------------------
# 4. Build Preprocessing Pipelines
# -----------------------------
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ]
)

# -----------------------------
# 5. Apply Transform
# -----------------------------
X_preprocessed = preprocessor.fit_transform(X)
print("Transformed shape:", X_preprocessed.shape)

# Save processed data for later modeling
import joblib
joblib.dump(preprocessor, "preprocessor.pkl")
print("Preprocessing complete. Preprocessor saved to preprocessor.pkl")
