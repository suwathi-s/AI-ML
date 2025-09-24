# 🛍️ Customer Conversion Analysis for Online Shopping

An end-to-end **Machine Learning and Analytics project** that predicts **customer conversion**, estimates **session revenue**, and segments customers into **behavioral clusters** using e-commerce clickstream data.

---

## Dataset

* Source: UCI Machine Learning Repository – Clickstream Data.
* Variables: Country, session_id, page categories, product details, price, etc.
* Train/Test CSVs provided in the repository.


##  Features

* **Classification** – Predict whether a browsing session ends in a purchase (1 = Purchase, 2 = Not Purchase).
* **Regression** – Estimate the potential revenue for each session.
* **Clustering** – Segment sessions into distinct behavioral groups for marketing.
* **Interactive Streamlit App**
  * **Insights Dashboard** with filters (cluster, country, probability, revenue).
  * **Batch Prediction** with downloadable CSV of predictions.
  * **Single Session Prediction** with sliders and dropdowns.
  * **Model Information** page describing all ML models and pipeline.

---

## 🗂️ Project Structure

customer-conversion-analysis/
│
├─ app.py # Streamlit multi-page app
├─ data_preprocessing.py # Missing value handling & scaling
├─ feature_engineering.py # Session-level aggregates & encoding
├─ classification_model_training.py
├─ regression_model_training.py
├─ clustering_model_training.py
├─ evaluate_classifier.py
├─ final_analysis.py
├─ requirements.txt
│
├─ best_classifier.pkl # Trained Gradient Boosting classifier
├─ best_regressor.pkl # Trained regressor
├─ best_cluster_model.pkl # Trained clustering model (offline)
├─ preprocessor.pkl # Preprocessing pipeline
├─ test_clusters.csv # Precomputed cluster labels
├─ train_data.csv / test_data.csv
└─ train_features.csv / test_features.csv



---

##  Approach

1. **Data Preprocessing & Feature Engineering**
   * Imputation of missing values (median/mode).
   * Label/one-hot encoding of categorical features (Country, Colour, etc.).
   * Session metrics: total clicks, unique pages, average price.

2. **Modeling**
   * **Classification**: Gradient Boosting Classifier (XGBoost/LightGBM).
   * **Regression**: Gradient Boosting Regressor.
   * **Clustering**: K-Means with 4–6 clusters.

3. **Evaluation**
   * Classification: Accuracy, F1, ROC-AUC.
   * Regression: RMSE, MAE, R².
   * Clustering: Silhouette Score, Davies–Bouldin Index.


##  Running Locally

streamlit run app.py


## Author: suwathi s

