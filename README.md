# Unsupervised-Stock-Keeping-Units

# 📦 Advanced SKU Clustering Dashboard

## 🚀 Overview

This project implements an **end-to-end Unsupervised Machine Learning pipeline** for **Stock Keeping Unit (SKU) segmentation** using multiple clustering techniques.

The system processes raw SKU data, performs preprocessing, reduces dimensionality using PCA, and applies multiple clustering algorithms to group similar products.

An interactive **Streamlit dashboard** is built to visualize clustering results and enable real-time predictions.

---

## 🎯 Objectives

* Segment SKUs based on patterns in data
* Compare multiple clustering algorithms
* Automatically select the best model
* Visualize clusters for better business insights

---

## 🧠 Algorithms Used

* **KMeans Clustering** – Partition-based clustering
* **DBSCAN** – Density-based clustering (detects noise/outliers)
* **Hierarchical Clustering** – Tree-based clustering
* **PCA (Principal Component Analysis)** – Dimensionality reduction

---

## ⚙️ Machine Learning Pipeline

The pipeline includes:

* Missing value handling (**SimpleImputer**)
* Feature scaling (**StandardScaler**)
* Categorical encoding (**OneHotEncoder**)
* Dimensionality reduction (**PCA**)
* Clustering models (KMeans, DBSCAN, Hierarchical)

---

## 📊 Model Evaluation

* **Elbow Method** → Optimal cluster selection
* **Silhouette Score** → Model performance comparison
* **Automatic Best Model Selection** based on highest score

---

## 📈 Visualizations

* 📍 PCA-based Cluster Scatter Plot
* 📊 Cluster Distribution Chart
* 📉 Elbow Method Graph

---

## 🖥️ Streamlit Dashboard Features

* Upload SKU dataset (.xlsx)
* View clustering results instantly
* Compare model performance
* Interactive visualizations using Plotly

---

## 📁 Project Structure

```
├── train_advanced_sku.py        # Training pipeline + model saving
├── app.py                      # Streamlit dashboard
├── advanced_sku_model.pkl      # Saved model (pickle file)
├── sku_data.xlsx               # Dataset
├── elbow.png                   # Elbow method graph
├── kmeans_clusters.png
├── dbscan_clusters.png
├── hierarchical_clusters.png
└── README.md
```

---

## ⚡ Installation

```bash
pip install pandas scikit-learn openpyxl streamlit plotly
```

---

## ▶️ How to Run

### Step 1: Train Model

```bash
python train_advanced_sku.py
```

### Step 2: Run Dashboard

```bash
streamlit run app.py
```

---

## 📌 Output

* Cluster labels assigned to each SKU
* Visual insights into SKU segmentation
* Best clustering model selected automatically

---

## 💡 Use Cases

* Inventory segmentation
* Demand pattern grouping
* Warehouse optimization
* Product categorization

---

## 🔥 Key Highlights

* End-to-end ML pipeline
* Multiple clustering algorithms
* Automated model selection
* Interactive dashboard deployment
* Industry-level project structure

---

## 👨‍💻 Author

**Praveen Reddy**
Aspiring Data Scientist | Machine Learning Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!
