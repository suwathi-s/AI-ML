# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Conversion Analysis", layout="wide")
st.title("🛍️ Customer Conversion Analysis Dashboard")

# ---------------------------
# Helpers
# ---------------------------
def load_model_safe(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Failed to load {path}: {e}")
        return None

def map_conversion_labels(preds):
    return np.where(np.asarray(preds) == 1, 1, 2)

# ---------------------------
# Load models
# ---------------------------
classifier = load_model_safe("best_classifier.pkl")
regressor  = load_model_safe("best_regressor.pkl")

clf_cols = list(classifier.feature_names_in_) if classifier else []
reg_cols = list(regressor.feature_names_in_)  if regressor  else []

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Insights", "Batch Prediction", "Single Prediction", "Model Info"])

# ============================================================
# Insights Page
# ============================================================
if page == "Insights":
    st.header("Insights Dashboard")
    try:
        final_df = pd.read_csv("final_test_predictions.csv")
    except FileNotFoundError:
        st.info("Run **Batch Prediction** first.")
        st.stop()

    st.sidebar.subheader("Filters")
    clusters = sorted(final_df["cluster"].dropna().unique()) if "cluster" in final_df else []
    countries = sorted(final_df["country"].dropna().unique()) if "country" in final_df else []
    clust_f = st.sidebar.multiselect("Cluster", clusters, default=clusters)
    country_f = st.sidebar.multiselect("Country", countries, default=countries)
    prob_range = st.sidebar.slider("Conversion Probability", 0.0, 1.0, (0.0, 1.0))
    rev_min, rev_max = final_df["pred_revenue"].min(), final_df["pred_revenue"].max()
    rev_range = st.sidebar.slider("Predicted Revenue", float(rev_min), float(rev_max),
                                  (float(rev_min), float(rev_max)))

    filt = pd.Series(True, index=final_df.index)
    if clust_f:   filt &= final_df["cluster"].isin(clust_f)
    if country_f: filt &= final_df["country"].isin(country_f)
    filt &= final_df["pred_conversion_prob"].between(*prob_range)
    filt &= final_df["pred_revenue"].between(*rev_range)
    view = final_df[filt]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Sessions", len(view))
    c2.metric("Avg Pred Revenue", f"${view['pred_revenue'].mean():.2f}")
    c3.metric("Avg Conv Prob", f"{view['pred_conversion_prob'].mean():.2%}")
    c4.metric("Purchase Rate", f"{(view['pred_conversion']==1).mean():.2%}")

    if "cluster" in view.columns:
        st.subheader("Avg Revenue by Cluster")
        grp = view.groupby("cluster")["pred_revenue"].mean().reset_index()
        fig, ax = plt.subplots()
        ax.bar(grp["cluster"].astype(str), grp["pred_revenue"], color="skyblue")
        st.pyplot(fig)

    st.subheader("Conversion Prob vs Revenue")
    fig2, ax2 = plt.subplots()
    ax2.scatter(view["pred_conversion_prob"], view["pred_revenue"],
                c=view.get("cluster", pd.Series([-1]*len(view))), alpha=0.6, s=12)
    ax2.set_xlabel("Conversion Probability"); ax2.set_ylabel("Predicted Revenue")
    st.pyplot(fig2)
    st.subheader("Sample Records")
    st.dataframe(view.head(200))

# ============================================================
# Batch Prediction
# ============================================================
elif page == "Batch Prediction":
    st.header("Batch Prediction")
    file = st.file_uploader("Upload feature-engineered CSV", type="csv")
    if file is None:
        st.info("Upload a CSV to run predictions.")
        st.stop()

    df_new = pd.read_csv(file)
    st.subheader("Uploaded Preview"); st.dataframe(df_new.head())

    # Precomputed clusters
    try:
        clusters = pd.read_csv("test_clusters.csv")[["session_id", "cluster"]]
        df_new = df_new.merge(clusters, on="session_id", how="left")
    except Exception:
        st.warning("No precomputed clusters found."); df_new["cluster"] = -1

    for c in clf_cols:
        if c not in df_new: df_new[c] = 0
    for c in reg_cols:
        if c not in df_new: df_new[c] = 0

    Xc, Xr = df_new[clf_cols], df_new[reg_cols]
    df_new["pred_conversion"] = map_conversion_labels(classifier.predict(Xc))
    df_new["pred_conversion_prob"] = classifier.predict_proba(Xc)[:,1]
    df_new["pred_revenue"] = regressor.predict(Xr)

    st.subheader("Prediction Preview")
    st.dataframe(df_new.head(200))
    df_new.to_csv("final_test_predictions.csv", index=False)
    st.success("Saved to final_test_predictions.csv")
    st.download_button("Download CSV", df_new.to_csv(index=False).encode(),
                       "final_test_predictions.csv")

# ============================================================
# Single Prediction  (with full dataset categories)
# ============================================================
elif page == "Single Prediction":
    st.header("Single Session Prediction")

    # --- Category dictionaries
    country_map = {
        1:"Australia",2:"Austria",3:"Belgium",4:"British Virgin Islands",5:"Cayman Islands",
        6:"Christmas Island",7:"Croatia",8:"Cyprus",9:"Czech Republic",10:"Denmark",
        11:"Estonia",12:"Unidentified",13:"Faroe Islands",14:"Finland",15:"France",
        16:"Germany",17:"Greece",18:"Hungary",19:"Iceland",20:"India",21:"Ireland",
        22:"Italy",23:"Latvia",24:"Lithuania",25:"Luxembourg",26:"Mexico",
        27:"Netherlands",28:"Norway",29:"Poland",30:"Portugal",31:"Romania",
        32:"Russia",33:"San Marino",34:"Slovakia",35:"Slovenia",36:"Spain",
        37:"Sweden",38:"Switzerland",39:"Ukraine",40:"UAE",41:"United Kingdom",
        42:"USA",43:"biz (.biz)",44:"com (.com)",45:"int (.int)",46:"net (.net)",
        47:"org (*.org)"
    }
    page1_map = {1:"Trousers",2:"Skirts",3:"Blouses",4:"Sale"}
    colour_map = {1:"Beige",2:"Black",3:"Blue",4:"Brown",5:"Burgundy",6:"Gray",
                  7:"Green",8:"Navy Blue",9:"Many colors",10:"Olive",11:"Pink",
                  12:"Red",13:"Violet",14:"White"}
    location_map = {1:"Top Left",2:"Top Middle",3:"Top Right",
                    4:"Bottom Left",5:"Bottom Middle",6:"Bottom Right"}
    model_photo_map = {1:"En Face",2:"Profile"}
    

    with st.form("single_form"):
        year = st.selectbox("Year", [2008])
        month = st.selectbox("Month", list(range(4,9)))
        day = st.slider("Day of Month", 1, 31, 1)
        order = st.number_input("Order (click sequence)", 1, 100, 1)
        country = st.selectbox("Country", options=list(country_map.keys()),
                               format_func=lambda x: f"{x} - {country_map[x]}")
        session_id = st.text_input("Session ID", "108001")
        page1 = st.selectbox("Page1 Main Category", options=list(page1_map.keys()),
                             format_func=lambda x: f"{x} - {page1_map[x]}")
        page2 = st.number_input("Page2 Clothing Model (1-217)", 1, 217, 1)
        colour = st.selectbox("Colour", options=list(colour_map.keys()),
                              format_func=lambda x: f"{x} - {colour_map[x]}")
        location = st.selectbox("Photo Location", options=list(location_map.keys()),
                                format_func=lambda x: f"{x} - {location_map[x]}")
        model_photo = st.selectbox("Model Photography", options=list(model_photo_map.keys()),
                                   format_func=lambda x: f"{x} - {model_photo_map[x]}")
        price = st.number_input("Price (USD)", 0.0, 500.0, 50.0)
        page = st.selectbox("Page Number", list(range(1,6)))
        total_clicks = st.number_input("Total Clicks", 0, 500, 10)
        avg_price = st.number_input("Average Price", 0.0, 500.0, 50.0)
        unique_pages = st.number_input("Unique Pages", 1, 10, 3)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Construct input row
        input_df = pd.DataFrame([{
            "YEAR":year,"MONTH":month,"DAY":day,"ORDER":order,
            "COUNTRY":country,"session_id":session_id,
            "page1_main_category":page1,"page2_clothing_model":page2,
            "colour":colour,"location":location,"model_photography":model_photo,
            "price":price,"page":page,
            "total_clicks":total_clicks,"avg_price":avg_price,"unique_pages":unique_pages
        }])

        for c in clf_cols:
            if c not in input_df: input_df[c] = 0
        for c in reg_cols:
            if c not in input_df: input_df[c] = 0

        Xc, Xr = input_df[clf_cols], input_df[reg_cols]
        pred_conv = map_conversion_labels(classifier.predict(Xc))[0]
        pred_prob = classifier.predict_proba(Xc)[:,1][0]
        pred_rev = regressor.predict(Xr)[0]

        status = "1 - Purchase" if pred_conv == 1 else "2 - Not Purchase"
        st.success(f"Predicted Conversion: {status}")

        st.info(f"Conversion Probability: {pred_prob:.2%}")
        st.info(f"Predicted Revenue: ${pred_rev:,.2f}")

# ============================================================
# Model Info
# ============================================================
elif page == "Model Info":
    st.header("Model Information & Documentation")

    # --- General project description
    st.markdown("""
    ###  Project Goal
    The app predicts **customer conversion**, estimates **expected revenue**,  
    and segments sessions into **customer clusters** using click-stream data.
    """)

    # --------------------------------------------------------
    # 1️ Classification Model
    # --------------------------------------------------------
    st.subheader(" Classification Model – Purchase Prediction")
    st.write("""
    * **Purpose:** Predicts whether a browsing session will end in a purchase.  
    * **Algorithm:** Gradient Boosting Classifier (e.g., XGBoost or LightGBM – adjust if different).  
    * **Target:** `conversion` label → `1 = Purchase`, `2 = No Purchase`.  
    * **Key Features:**  
        - Time features: **Month**, **Day**, **Order**  
        - Product/category info: **Page1**, **Page2**, **Colour**, **Location**, **Price/Price2**  
        - Behaviour features: **Total Clicks**, **Unique Pages**, **Average Price**, etc.
    * **Outputs:**  
        - **Prediction**: 1 or 2 (Purchase / No Purchase)  
        - **Probability**: Likelihood of purchase
    * **Metrics:** Accuracy, F1-score, ROC-AUC.
    """)

    # --------------------------------------------------------
    # 2️ Regression Model
    # --------------------------------------------------------
    st.subheader(" Regression Model – Revenue Estimation")
    st.write("""
    * **Purpose:** Estimates **expected revenue in USD** for each session.  
    * **Algorithm:** Gradient Boosting Regressor (e.g., XGBoostRegressor).  
    * **Target:** Continuous numeric `revenue` value.
    * **Input Features:** Same engineered set as classifier.
    * **Output:** Predicted revenue amount.
    * **Metrics:** RMSE, MAE, R².
    """)

    # --------------------------------------------------------
    # 3️ Clustering Model
    # --------------------------------------------------------
    st.subheader(" Clustering Model – Customer Segmentation")
    st.write("""
    * **Purpose:** Groups sessions into behavioural **clusters** for marketing and analysis.  
    * **Algorithm:** K-Means (trained offline).  
    * **Training Data:** The same feature set used for prediction, after scaling.  
    * **Number of Clusters:** Typically 4–6 (check your `test_clusters.csv` for exact labels).  
    * **Outputs:**  
        - `cluster` ID for each session (integer label).  
        - Allows analysis of **average revenue**, **conversion probability**, and **click behaviour** per cluster.
    * **Use in App:**  
        - Cluster labels are precomputed and stored in `test_clusters.csv`.  
        - They are merged with uploaded session data to provide the **Cluster Summary & Visualization** tab.
    """)

    # --------------------------------------------------------
    # Data Pipeline
    # --------------------------------------------------------
    st.subheader(" Data Pre-processing & Feature Engineering")
    st.write("""
    * Missing value imputation (median for numeric, constant for categorical).  
    * One-hot encoding for categorical variables such as **Country**, **Colour**, **Page1**, etc.  
    * Scaling/standardization where required for clustering and regression.  
    * Behavioural aggregates: total clicks, unique pages, average price.
    """)

    # --------------------------------------------------------
    # Loaded Artifacts Overview
    # --------------------------------------------------------
    st.subheader(" Loaded Artifacts")
    def show_model(name, model, cols):
        st.markdown(f"**{name}:** {' Loaded' if model else ' Not Loaded'}")
        if model:
            st.write(f"Feature count: {len(cols)}")
            if cols:
                st.write("Sample features:", ", ".join(cols[:15]) + (" ..." if len(cols) > 15 else ""))

    show_model("Classifier", classifier, clf_cols)
    show_model("Regressor",  regressor,  reg_cols)

    # Cluster file check
    try:
        clusters_preview = pd.read_csv("test_clusters.csv").head()
        st.markdown("**Cluster Labels File:**  `test_clusters.csv` loaded")
        st.dataframe(clusters_preview)
    except Exception as e:
        st.markdown(f"**Cluster Labels File:**  Not loaded ({e})")

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------
    st.subheader(" Interpretation Tips")
    st.markdown("""
    * **Conversion Probability** ranks sessions for targeted marketing.  
    * **Predicted Revenue** helps forecast potential earnings per session.  
    * **Clusters** reveal groups with similar shopping behaviour (e.g., high-value vs. low-value).
    """)

