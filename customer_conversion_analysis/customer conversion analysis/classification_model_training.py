import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
import joblib


def evaluate(y_true, y_pred, y_proba):
    """Return key classification metrics."""
    # Only probability of positive class for binary ROC-AUC
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_proba = y_proba[:, 1]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1": f1_score(y_true, y_pred, average="weighted"),
        "ROC-AUC": roc_auc_score(y_true, y_proba)
    }


def main():
    # === Load data ===
    data = pd.read_csv(
        r"C:\Users\91902\OneDrive\Desktop\customer conversion analysis\train_features.csv"
    )

    # Shift labels from {1,2} to {0,1}
    y = data.pop("price_2") - 1
    X = data

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}

    # Logistic Regression
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])
    log_reg.fit(X_train, y_train)
    results["LogisticRegression"] = evaluate(
        y_val, log_reg.predict(X_val), log_reg.predict_proba(X_val)
    )

    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=15, class_weight="balanced", random_state=42
    )
    dt.fit(X_train, y_train)
    results["DecisionTree"] = evaluate(
        y_val, dt.predict(X_val), dt.predict_proba(X_val)
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results["RandomForest"] = evaluate(
        y_val, rf.predict(X_val), rf.predict_proba(X_val)
    )

    # XGBoost  
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    results["XGBoost"] = evaluate(
        y_val, xgb.predict(X_val), xgb.predict_proba(X_val)
    )

    # Simple Neural Net
    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = len(y.unique())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.long)

    for _ in range(20):
        optimizer.zero_grad()
        outputs = net(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        outputs = net(X_val_t)
        pred_nn = outputs.argmax(1).numpy()
        proba_nn = torch.softmax(outputs, dim=1).numpy()

    results["NeuralNet"] = evaluate(y_val, pred_nn, proba_nn)

    # === Show comparison ===
    print("\n=== Model Comparison ===")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Save the best by F1
    best_model_name = max(results, key=lambda m: results[m]["F1"])
    print(f"\nBest model based on F1: {best_model_name}")

    if best_model_name == "RandomForest":
        joblib.dump(rf, "best_classifier.pkl")
    elif best_model_name == "LogisticRegression":
        joblib.dump(log_reg, "best_classifier.pkl")
    elif best_model_name == "DecisionTree":
        joblib.dump(dt, "best_classifier.pkl")
    elif best_model_name == "XGBoost":
        joblib.dump(xgb, "best_classifier.pkl")
    elif best_model_name == "NeuralNet":
        torch.save(net.state_dict(), "best_classifier_nn.pth")


if __name__ == "__main__":
    main()
