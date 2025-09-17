# final_analysis.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# ----------------------------
# 1. Load test features and cluster labels
# ----------------------------
test_features = pd.read_csv("test_features.csv")
clusters = pd.read_csv("test_clusters.csv")[["session_id", "cluster"]]

# Merge cluster labels
test_features = test_features.merge(clusters, on="session_id", how="left")

# ----------------------------
# 2. Load saved models
# ----------------------------
classifier = joblib.load("best_classifier.pkl")   # Classification model
regressor = joblib.load("best_regressor.pkl")    # Regression model

# ----------------------------
# 3. Prepare features for prediction
# ----------------------------
cols_to_drop = ["price", "price_2", "cluster"]
X_test_model = test_features.drop(columns=cols_to_drop)

# ----------------------------
# 4. Predict conversion & revenue
# ----------------------------
test_features["pred_conversion"] = classifier.predict(X_test_model)
test_features["pred_conversion_prob"] = classifier.predict_proba(X_test_model)[:, 1]
test_features["pred_revenue"] = regressor.predict(X_test_model)

# ----------------------------
# 5. Cluster-level summary
# ----------------------------
cluster_summary = test_features.groupby("cluster").agg({
    "pred_revenue": "mean",
    "pred_conversion": "mean",
    "total_clicks": "mean",
    "avg_price": "mean",
    "unique_pages": "mean"
}).reset_index()

print("\n=== Cluster Summary ===")
print(cluster_summary)

# ----------------------------
# 6. PCA 2D Projection for Cluster Visualization
# ----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_model)

plt.figure(figsize=(8,6))
for c in test_features["cluster"].unique():
    mask = test_features["cluster"] == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {c}', alpha=0.6)
plt.title("PCA: 2D Cluster Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cluster_pca_plot.png")
plt.show()

# ----------------------------
# 7. Bar Plots: Predicted Revenue & Conversion by Cluster
# ----------------------------
fig, ax = plt.subplots(1, 2, figsize=(12,5))

# Revenue
cluster_summary.plot(kind="bar", x="cluster", y="pred_revenue", ax=ax[0], color="skyblue")
ax[0].set_title("Average Predicted Revenue by Cluster")
ax[0].set_ylabel("Revenue")

# Conversion Probability
cluster_summary.plot(kind="bar", x="cluster", y="pred_conversion", ax=ax[1], color="orange")
ax[1].set_title("Average Predicted Conversion by Cluster")
ax[1].set_ylabel("Conversion Probability")

plt.tight_layout()
plt.savefig("cluster_revenue_conversion_bar.png")
plt.show()

# ----------------------------
# 8. Feature Importance Plots
# ----------------------------
def plot_feature_importance(model, X, title, filename):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        print(f"Model type {type(model)} does not have feature_importances_ or coef_")
        return
    
    feature_names = X.columns
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    
    plt.figure(figsize=(8,6))
    plt.barh(importance_df["feature"], importance_df["importance"], color="teal")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Conversion feature importance
plot_feature_importance(classifier, X_test_model, "Top Features Driving Conversion", "feature_importance_conversion.png")

# Revenue feature importance
plot_feature_importance(regressor, X_test_model, "Top Features Driving Revenue", "feature_importance_revenue.png")

# ----------------------------
# 9. Save final combined predictions
# ----------------------------
test_features.to_csv("final_test_predictions.csv", index=False)
print("\nSaved final combined predictions -> final_test_predictions.csv")
print("Saved PCA plot -> cluster_pca_plot.png")
print("Saved bar plots -> cluster_revenue_conversion_bar.png")
print("Saved feature importance plots -> feature_importance_conversion.png & feature_importance_revenue.png")
