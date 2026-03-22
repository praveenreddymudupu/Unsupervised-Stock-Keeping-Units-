# app.py

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.title(" Stock Keeping Units Clustering Dashboard")

# Load model
model_bundle = pickle.load(open("advanced_sku_model.pkl", "rb"))

pipeline = model_bundle["pipeline"]
kmeans = model_bundle["kmeans"]
dbscan = model_bundle["dbscan"]
hierarchical = model_bundle["hierarchical"]
best_model = model_bundle["best_model"]
scores = model_bundle["scores"]

# ---------------------------
# Show Scores
# ---------------------------
st.subheader("📊 Model Scores")
st.write(scores)

uploaded_file = st.file_uploader("Upload SKU File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Transform
    X = pipeline.transform(df)

    # Predictions
    df["KMeans"] = kmeans.predict(X)
    df["DBSCAN"] = dbscan.fit_predict(X)
    df["Hierarchical"] = hierarchical.fit_predict(X)
    df["Best_Model"] = best_model.fit_predict(X)

    st.write("### 🔍 Results")
    st.dataframe(df)

    # =====================================================
    # 🔥 PCA SCATTER VISUALIZATION
    # =====================================================
    st.subheader("📍 Cluster Visualization (PCA)")

    pca_df = pd.DataFrame(X, columns=["PC1", "PC2"])
    pca_df["Cluster"] = df["Best_Model"]

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="Clusters Visualization (Best Model)",
    )

    st.plotly_chart(fig)

    # =====================================================
    # 🔥 CLUSTER DISTRIBUTION
    # =====================================================
    st.subheader("📊 Cluster Distribution")

    cluster_counts = df["Best_Model"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]

    fig2 = px.bar(
        cluster_counts,
        x="Cluster",
        y="Count",
        title="Cluster Distribution"
    )

    st.plotly_chart(fig2)

    st.success("✅ Visualization Completed!")