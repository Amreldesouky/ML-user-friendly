# 📊 Streamlit AutoML Mini App

A simple yet powerful **AutoML** web application built with [Streamlit](https://streamlit.io/) that allows you to upload your dataset, select a machine learning algorithm, and instantly visualize results — all without writing code.

---

## 🚀 Features
- **Upload CSV data** directly from your device
- Choose learning type:
  - **Supervised**: Classification / Regression
  - **Unsupervised**: Clustering
- Models supported:
  - Linear Regression
  - Logistic Regression
  - SVM
  - KNN
  - ANN (MLP)
  - KMeans
- Automatic preprocessing:
  - Handle missing values
  - Encode categorical features
  - Scale numerical features
- Visualizations:
  - Model fit plots
  - Scatter plots for regression
  - Cluster visualizations

---

## 🖥️ Live Demo
🔗 **[View the Project Here][(YOUR_PROJECT_LINK_HERE)](https://ml-user-friendly-ryz9crwlrb2crm6mcsbi6d.streamlit.app/)]**  
*(Replace `YOUR_PROJECT_LINK_HERE` with your deployed Streamlit Cloud link)*

---

## 📦 Dependencies

This app requires the following Python libraries:

```txt
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow  # Optional, only if using ANN (MLP)
