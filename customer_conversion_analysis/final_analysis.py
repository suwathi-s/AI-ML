# final_analysis.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ----------------------------
# 1. Load test features and cluster labels
# ----------------------------
test_features = pd.read_csv("test_features.csv")
clusters = pd.read_csv("test_clusters.csv")[["session_id", "cluster"]]
test_features = test_features.merge(clusters, on="session_id", how="left")

# ----------------------------
# 2. Load saved models
# ----------------------------
classifier = joblib.load("best_classifier.pkl")
regressor  = joblib.load("best_regressor.pkl")

# ----------------------------
# 3. Ensure we use EXACT training columns
# ----------------------------
if hasattr(classifier, "feature_names_in_"):
    clf_cols = list(classifier.feature_names_in_)
else:
    raise ValueError(
        "Classifier does not store feature_names_in_. Provide exact training columns manually."
    )

if hasattr(regressor, "feature_names_in_"):
    reg_cols = list(regressor.feature_names_in_)
else:
    raise ValueError(
        "Regressor does not store feature_names_in_. Provide exact training columns manually."
    )

# Subset test features to exactly match model's training columns
X_classifier = test_features[clf_cols]
X_regressor  = test_features[reg_cols]

# ----------------------------
# 4. Predictions
# ----------------------------
# Classification: convert 0/1 to 1/2
test_features["pred_conversion"] = classifier.predict(X_classifier) + 1
test_features["pred_conversion_prob"] = classifier.predict_proba(X_classifier)[:,1]
# Regression: revenue prediction
test_features["pred_revenue"] = regressor.predict(X_regressor)

# ----------------------------
# 5. Cluster-level summary
# ----------------------------
cluster_summary = (
    test_features.groupby("cluster")
    .agg({
        "pred_revenue":"mean",
        "pred_conversion_prob":"mean",
        "pred_conversion":"mean",
        "total_clicks":"mean",
        "avg_price":"mean",
        "unique_pages":"mean"
    })
    .reset_index()
)
print("\n=== Cluster Summary ===")
print(cluster_summary)

# ----------------------------
# 6. PCA Visualization
# ----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_regressor)
plt.figure(figsize=(8,6))
for c in test_features["cluster"].unique():
    mask = test_features["cluster"] == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {c}', alpha=0.6)
plt.title("PCA: 2D Cluster Visualization")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("cluster_pca_plot.png")
plt.show()

# ----------------------------
# 7. Cluster-wise Bar Plots
# ----------------------------
# Revenue by cluster
plt.figure(figsize=(6,4))
plt.bar(cluster_summary["cluster"].astype(str), cluster_summary["pred_revenue"], color='skyblue')
plt.title("Average Predicted Revenue by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Revenue")
plt.tight_layout()
plt.savefig("cluster_avg_revenue.png")
plt.show()

# Conversion probability by cluster
plt.figure(figsize=(6,4))
plt.bar(cluster_summary["cluster"].astype(str), cluster_summary["pred_conversion_prob"], color='lightgreen')
plt.title("Average Predicted Conversion Probability by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Conversion Probability")
plt.tight_layout()
plt.savefig("cluster_avg_conversion_prob.png")
plt.show()

# ----------------------------
# 8. Save Final Predictions
# ----------------------------
test_features.to_csv("final_test_predictions.csv", index=False)
print("\nSaved final predictions to final_test_predictions.csv")
