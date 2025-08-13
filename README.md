#  Streamlit AutoML Mini App

A user-friendly web application built with [Streamlit](https://streamlit.io/) that enables anyone to perform automated machine learning tasks — no coding required. Upload your CSV data, choose a model, get real-time metrics and plots, and even make custom predictions.

---

##  Live Demo
Check out the app in action here:  
🔗 [https://ml-user-friendly-ryz9crwlrb2crm6mcsbi6d.streamlit.app/](https://ml-user-friendly-ryz9crwlrb2crm6mcsbi6d.streamlit.app/)

---

##  Features

- Upload CSV file from your local machine.
- Choose between Supervised (Classification / Regression) or Unsupervised (Clustering) tasks.
- Supports a variety of commonly used models:
  - Linear Regression – Predict continuous outcomes using a linear combination of input features.
  - Logistic Regression – Classify data into categories using a logistic (sigmoid) function.
  - Support Vector Machines (SVM) – Powerful classification and regression using customizable kernels.
  - K-Nearest Neighbors (KNN) – Simple yet effective method predicting outputs based on similarity to nearby samples.
  - Artificial Neural Network (ANN) via MLP – Flexible deep learning approach with customizable hidden layers.
  - KMeans Clustering (Unsupervised) – Partition data into groups based on similarity via iterative centroid optimization.

- Automatic Preprocessing:
  - Numeric imputation (mean / median / most frequent).
  - Categorical imputation (most frequent / constant).
  - One-hot encoding of categories.
  - Optional scaling of numeric features for numeric stability.

- Train/Test Split & Metrics:
  - Classification: Accuracy, Weighted F1, Confusion Matrix, ROC Curve & AUC (binary classification), Training Loss Curve (for ANN).
  - Regression: MSE, MAE, R², Actual vs Predicted and Residual plots, Training Loss Curve (for ANN).
  - Clustering: Cluster label counts, Silhouette Score (if applicable), PCA 2D visualization.

- Custom Input Predictions: After training, input feature values manually to get single-sample predictions and probabilities (for classification).

---

##  Dependencies

```txt
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow  # Optional, only needed if using the ANN (MLP) models
