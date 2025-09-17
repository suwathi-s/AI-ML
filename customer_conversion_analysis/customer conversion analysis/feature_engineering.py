import pandas as pd
import numpy as np

# Load raw data
data = pd.read_csv(r"C:\Users\91902\OneDrive\Desktop\customer conversion analysis\train_data.csv")

# --- 1. Drop columns that add no predictive value ---
data = data.drop(columns=["year"])  # constant
# keep session_id for now if you want session-based features

# --- 2. Session-level features ---
# Example: clicks per session
session_features = (
    data.groupby("session_id")
        .agg(
            total_clicks=("order", "max"),
            avg_price=("price", "mean"),
            unique_pages=("page", "nunique")
        )
        .reset_index()
)

# Merge back to each row if needed
data = data.merge(session_features, on="session_id", how="left")

# --- 3. Encode categorical variables ---
# page2_clothing_model is high-cardinality; start with label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["page2_clothing_model"] = le.fit_transform(data["page2_clothing_model"])

# Save processed data
data.to_csv("train_features.csv", index=False)
print("Feature engineering complete. Output: train_features.csv")
