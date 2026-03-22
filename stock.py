# train_advanced_sku.py

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_excel('sku_data.xlsx')
df = df.dropna(how='all')

# ---------------------------
# Column Separation
# ---------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# ---------------------------
# Preprocessing
# ---------------------------
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# ---------------------------
# PCA
# ---------------------------
pca = PCA(n_components=2)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', pca)
])

X = pipeline.fit_transform(df)

# =====================================================
# 🔥 1. ELBOW METHOD
# =====================================================
inertia = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.savefig("elbow.png")
plt.close()

# Choose optimal k (simple logic)
optimal_k = 3

# =====================================================
# 🔥 2. MODELS
# =====================================================
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
hierarchical_labels = hierarchical.fit_predict(X)

# =====================================================
# 🔥 3. SILHOUETTE SCORES
# =====================================================
def get_score(X, labels):
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    else:
        return -1  # invalid clustering

scores = {
    "KMeans": get_score(X, kmeans_labels),
    "DBSCAN": get_score(X, dbscan_labels),
    "Hierarchical": get_score(X, hierarchical_labels)
}

print("Silhouette Scores:", scores)

# =====================================================
# 🔥 4. BEST MODEL SELECTION
# =====================================================
best_model_name = max(scores, key=scores.get)

if best_model_name == "KMeans":
    best_model = kmeans
elif best_model_name == "DBSCAN":
    best_model = dbscan
else:
    best_model = hierarchical

print("Best Model:", best_model_name)

# =====================================================
# 🔥 5. VISUALIZATION
# =====================================================
plt.figure()

plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
plt.title("KMeans Clusters")
plt.savefig("kmeans_clusters.png")
plt.close()

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels)
plt.title("DBSCAN Clusters")
plt.savefig("dbscan_clusters.png")
plt.close()

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=hierarchical_labels)
plt.title("Hierarchical Clusters")
plt.savefig("hierarchical_clusters.png")
plt.close()

# =====================================================
# 🔥 SAVE EVERYTHING
# =====================================================
model_bundle = {
    "pipeline": pipeline,
    "kmeans": kmeans,
    "dbscan": dbscan,
    "hierarchical": hierarchical,
    "best_model": best_model,
    "scores": scores
}

with open("advanced_sku_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("✅ Saved as advanced_sku_model.pkl")