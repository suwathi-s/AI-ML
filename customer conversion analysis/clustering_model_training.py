# clustering_model_training.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
train_df = pd.read_csv(r"C:\Users\91902\OneDrive\Desktop\customer conversion analysis\train_features.csv")
test_df  = pd.read_csv(r"C:\Users\91902\OneDrive\Desktop\customer conversion analysis\test_features.csv")

# Drop obvious ID-like columns if they exist
id_cols = ["session_id", "order", "country"]
X_train = train_df.drop(columns=[c for c in id_cols if c in train_df.columns])
X_test  = test_df.drop(columns=[c for c in id_cols if c in test_df.columns])

# ---------------------------------------------------------
# 2. Scale Data
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

def safe_metrics(labels, X):
    """Return (Silhouette, Davies-Bouldin), ignoring DBSCAN noise (-1)."""
    mask = labels != -1
    if len(set(labels[mask])) <= 1:
        return -1.0, np.inf
    return (
        silhouette_score(X[mask], labels[mask]),
        davies_bouldin_score(X[mask], labels[mask])
    )

results = []

# ---------------------------------------------------------
# 3. K-Means: search k = 2..10
# ---------------------------------------------------------
print("\n=== K-Means ===")
best_kmeans, best_k, best_sil = None, None, -1
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X_train_s)
    test_labels = km.predict(X_test_s)
    sil, dbi = safe_metrics(test_labels, X_test_s)
    wcss = km.inertia_
    print(f"k={k:2d} | Silhouette={sil:.4f} | Davies-Bouldin={dbi:.4f} | WCSS={wcss:.2f}")
    if sil > best_sil:
        best_sil, best_k = sil, k
        best_kmeans = km

results.append({
    "name": f"KMeans(k={best_k})",
    "silhouette": best_sil,
    "davies_bouldin": davies_bouldin_score(X_test_s,
                                           best_kmeans.predict(X_test_s)),
    "wcss": best_kmeans.inertia_,
    "model": best_kmeans
})

# ---------------------------------------------------------
# 4. DBSCAN
# ---------------------------------------------------------
print("\n=== DBSCAN ===")
db = DBSCAN(eps=0.8, min_samples=5)
db_labels = db.fit_predict(X_test_s)   # evaluate on test
sil_db, dbi_db = safe_metrics(db_labels, X_test_s)
print(f"Silhouette={sil_db:.4f} | Davies-Bouldin={dbi_db:.4f} | WCSS=N/A")
results.append({
    "name": "DBSCAN",
    "silhouette": sil_db,
    "davies_bouldin": dbi_db,
    "wcss": None,
    "model": db
})

# ---------------------------------------------------------
# 5. Agglomerative (Hierarchical)
# ---------------------------------------------------------
print("\n=== Agglomerative ===")
agg = AgglomerativeClustering(n_clusters=best_k)
agg_labels = agg.fit_predict(X_test_s)
sil_agg, dbi_agg = safe_metrics(agg_labels, X_test_s)
print(f"Silhouette={sil_agg:.4f} | Davies-Bouldin={dbi_agg:.4f} | WCSS=N/A")
results.append({
    "name": f"Agglomerative(k={best_k})",
    "silhouette": sil_agg,
    "davies_bouldin": dbi_agg,
    "wcss": None,
    "model": agg
})

# ---------------------------------------------------------
# 6. Choose Best Model (highest Silhouette)
# ---------------------------------------------------------
best_model_info = max(results, key=lambda r: r["silhouette"])
print("\n=== BEST CLUSTERING MODEL ===")
print(f"Model: {best_model_info['name']}")
print(f"Silhouette Score: {best_model_info['silhouette']:.4f}")
print(f"Davies-Bouldin Index: {best_model_info['davies_bouldin']:.4f}")
if best_model_info["wcss"] is not None:
    print(f"Within-Cluster Sum of Squares: {best_model_info['wcss']:.2f}")

# ---------------------------------------------------------
# 7. Save Scaler + Best Model
# ---------------------------------------------------------
joblib.dump(
    {"scaler": scaler, "model": best_model_info["model"]},
    "best_cluster_model.pkl"
)
print("\nSaved best cluster model -> best_cluster_model.pkl")

# ---------------------------------------------------------
# 8. OPTIONAL: Save Test Set with Cluster Labels
# ---------------------------------------------------------
# Generate cluster labels for the test set for analysis/visualization
best_model = best_model_info["model"]

if hasattr(best_model, "predict"):
    labels = best_model.predict(X_test_s)
else:
    # Agglomerative / DBSCAN require fit_predict
    labels = best_model.fit_predict(X_test_s)

test_with_clusters = test_df.copy()
test_with_clusters["cluster"] = labels
test_with_clusters.to_csv("test_clusters.csv", index=False)
print("Saved cluster assignments -> test_clusters.csv")
