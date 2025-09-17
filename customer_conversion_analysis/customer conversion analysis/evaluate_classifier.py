import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    # 1️ Load test data
    test_df = pd.read_csv(r"C:\Users\91902\OneDrive\Desktop\customer conversion analysis\test_features.csv")

    # Replace 'target_column' with the real target column name from your project
    # In your previous code you mapped labels from {1,2} -> {0,1}
    y_test = test_df["price_2"] - 1
    X_test = test_df.drop(columns=["price_2"])

    # 2️ Load the saved best classifier
    model = joblib.load("best_classifier.pkl")

    # 3️ Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 4️ Metrics
    print("\n=== Final Test Metrics ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    main()
