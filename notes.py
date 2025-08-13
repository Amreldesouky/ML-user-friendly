import streamlit as st

def show_notes():
    with st.expander("Notes & Tips"):
        st.markdown("""
        **Supervised**
        - The **last column** in your CSV is treated as the **target**.
        - Choose the right metric: 
          - Classification → Accuracy & F1 (weighted) + Confusion Matrix + ROC (binary)
          - Regression → MSE / MAE / R² + Residuals plot
        - For ANN (MLP), a training **loss curve** is plotted.

        **Unsupervised**
        - No target column is used.
        - For KMeans, you get cluster counts, **silhouette score** (when valid), and a 2D PCA visualization.

        **Preprocessing**
        - Numeric: imputation (mean/median/most_frequent) + optional scaling.
        - Categorical: imputation + one-hot encoding (unknowns ignored safely).

        **Predictions**
        - After training a supervised model, enter custom inputs to get a single prediction.

        **Reproducibility & Deployment**
        - Run locally: streamlit run app.py
        - Suggested requirements: streamlit pandas numpy scikit-learn matplotlib
        """)
