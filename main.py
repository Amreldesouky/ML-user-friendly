# Streamlit AutoML Mini App
# -------------------------------------------------------------
# Features
# - Upload CSV
# - Choose Learning Type: Supervised (Classification/Regression) or Unsupervised (Clustering)
# - Models: Linear Regression, Logistic Regression, SVM, KNN, ANN (MLP), KMeans
# - Assumes LAST column is target for Supervised (as requested)
# - Robust preprocessing: missing values, categorical encoding, scaling
# - Train/Test split and metrics (Accuracy/F1; MSE/MAE/R2; inertia/silhouette)
# - Visualizations: Confusion Matrix, ROC (binary), Residuals, Training loss (MLP), PCA scatter for clusters
# - Optional: manual input for single prediction after training
# -------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)
from sklearn.decomposition import PCA

# -------------------------------------------------------------
# Page Config
# -------------------------------------------------------------
st.set_page_config(page_title="AutoML Mini App", layout="wide")
st.title("⚙️ AutoML Mini App — Supervised, Unsupervised & ANN")
st.caption("Upload a CSV, pick a learning type and model, and get metrics & visuals. **Supervised assumes the last column is the target.**")

# -------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------
with st.sidebar:
    st.header("1) Upload & Setup")
    file = st.file_uploader("Upload CSV file", type=["csv"])
    test_size = st.slider("Test size (for supervised)", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)
    st.divider()

    st.header("2) Learning Type")
    learning_type = st.radio("Choose learning type", ["Supervised", "Unsupervised"], index=0)

    problem_kind = None
    model_choice = None

    if learning_type == "Supervised":
        problem_kind = st.radio("Task", ["Classification", "Regression"], index=0)
        if problem_kind == "Classification":
            model_choice = st.selectbox(
                "Model",
                ["Logistic Regression", "SVM (SVC)", "KNN (Classifier)", "ANN (MLPClassifier)"]
            )
        else:
            model_choice = st.selectbox(
                "Model",
                ["Linear Regression", "SVM (SVR)", "KNN (Regressor)", "ANN (MLPRegressor)"]
            )
    else:
        model_choice = st.selectbox("Clustering Model", ["KMeans"])  # extendable

    st.divider()
    st.header("3) Preprocessing")
    scale_numeric = st.checkbox("Scale numeric features (StandardScaler)", value=True)
    impute_strategy_num = st.selectbox("Numeric imputation", ["mean", "median", "most_frequent"], index=0)
    impute_strategy_cat = st.selectbox("Categorical imputation", ["most_frequent", "constant"], index=0)

    st.divider()
    st.header("4) Model Hyperparameters")
    if learning_type == "Supervised":
        if model_choice == "Logistic Regression":
            C = st.number_input("C (inverse regularization)", min_value=1e-4, value=1.0, step=0.1, format="%f")
            max_iter = st.number_input("max_iter", min_value=100, value=500, step=50)
        elif model_choice == "SVM (SVC)":
            C = st.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f")
            kernel = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
            probability = st.checkbox("Enable probability estimates (slower)", value=True)
        elif model_choice == "KNN (Classifier)":
            n_neighbors = st.slider("n_neighbors", 1, 25, 5)
        elif model_choice == "ANN (MLPClassifier)":
            hidden = st.text_input("Hidden layers (e.g., 100 or 64,32)", value="100")
            activation = st.selectbox("activation", ["relu", "tanh", "logistic"], index=0)
            alpha = st.number_input("alpha (L2)", min_value=0.0, value=0.0001, step=0.0001, format="%f")
            max_iter = st.number_input("max_iter", min_value=100, value=300, step=50)
        elif model_choice == "Linear Regression":
            pass
        elif model_choice == "SVM (SVR)":
            C = st.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f")
            kernel = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
        elif model_choice == "KNN (Regressor)":
            n_neighbors = st.slider("n_neighbors", 1, 25, 5)
        elif model_choice == "ANN (MLPRegressor)":
            hidden = st.text_input("Hidden layers (e.g., 100 or 64,32)", value="64,32")
            activation = st.selectbox("activation", ["relu", "tanh", "logistic"], index=0)
            alpha = st.number_input("alpha (L2)", min_value=0.0, value=0.0001, step=0.0001, format="%f")
            max_iter = st.number_input("max_iter", min_value=100, value=300, step=50)
    else:
        if model_choice == "KMeans":
            n_clusters = st.slider("n_clusters", 2, 10, 3)
            init = st.selectbox("init", ["k-means++", "random"], index=0)
            n_init = st.number_input("n_init", min_value=1, value=10, step=1)
            max_iter = st.number_input("max_iter", min_value=10, value=300, step=10)

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def parse_hidden_layers(s: str) -> Tuple[int, ...]:
    try:
        parts = [int(p.strip()) for p in s.split(",") if p.strip()]
        if not parts:
            return (100,)
        return tuple(parts)
    except Exception:
        return (100,)


def build_preprocessor(df: pd.DataFrame, target_col: str | None) -> Tuple[ColumnTransformer, List[str], List[str]]:
    if target_col is not None and target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy_num)),
        ("scaler", StandardScaler() if scale_numeric else "passthrough")
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy_cat, fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return preprocessor, num_cols, cat_cols


# -------------------------------------------------------------
# Main Logic
# -------------------------------------------------------------
if file is not None:
    df = pd.read_csv(file)

    st.subheader("Preview")
    st.dataframe(df.head(20))

    if learning_type == "Supervised":
        df = df.dropna()

        target_col = df.columns[-1]
        st.info(f"Supervised mode: using **last column** as target → `{target_col}`")

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        preprocessor, num_cols, cat_cols = build_preprocessor(df, target_col)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() < 20 else None
        )

        # Model selection
        model = None
        is_classification = (problem_kind == "Classification")

        if is_classification:
            if model_choice == "Logistic Regression":
                model = LogisticRegression(C=C, max_iter=max_iter, n_jobs=None)
            elif model_choice == "SVM (SVC)":
                model = SVC(C=C, kernel=kernel, probability=probability)
            elif model_choice == "KNN (Classifier)":
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
            elif model_choice == "ANN (MLPClassifier)":
                model = MLPClassifier(hidden_layer_sizes=parse_hidden_layers(hidden), activation=activation,
                                      alpha=alpha, max_iter=max_iter, random_state=random_state)
        else:
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "SVM (SVR)":
                model = SVR(C=C, kernel=kernel)
            elif model_choice == "KNN (Regressor)":
                model = KNeighborsRegressor(n_neighbors=n_neighbors)
            elif model_choice == "ANN (MLPRegressor)":
                model = MLPRegressor(hidden_layer_sizes=parse_hidden_layers(hidden), activation=activation,
                                      alpha=alpha, max_iter=max_iter, random_state=random_state)

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        with st.spinner("Training..."):
            pipe.fit(X_train, y_train)

        st.success("Model trained ✅")

        # Metrics & Visuals
        y_pred = pipe.predict(X_test)

        if is_classification:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            st.metric("Accuracy", f"{acc:.4f}")
            st.metric("F1 (weighted)", f"{f1:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            im = ax.imshow(cm)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(len(np.unique(y_test))))
            ax.set_yticks(range(len(np.unique(y_test))))
            ax.figure.colorbar(im, ax=ax)
            st.pyplot(fig)

            # ROC (binary only)
            if len(np.unique(y_test)) == 2:
                try:
                    # get proba or decision function
                    if hasattr(pipe["model"], "predict_proba"):
                        y_score = pipe.predict_proba(X_test)[:, 1]
                    else:
                        y_score = pipe.decision_function(X_test)
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    auc_val = roc_auc_score(y_test, y_score)
                    fig2, ax2 = plt.subplots()
                    ax2.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
                    ax2.plot([0, 1], [0, 1], linestyle="--")
                    ax2.set_xlabel("FPR")
                    ax2.set_ylabel("TPR")
                    ax2.set_title("ROC Curve")
                    ax2.legend()
                    st.pyplot(fig2)
                except Exception as e:
                    st.info("ROC not available for this model.")

            # Loss curve for MLPClassifier
            if isinstance(model, MLPClassifier) and hasattr(model, "loss_curve_"):
                fig3, ax3 = plt.subplots()
                ax3.plot(model.loss_curve_)
                ax3.set_title("Training Loss (MLP)")
                ax3.set_xlabel("Iteration")
                ax3.set_ylabel("Loss")
                st.pyplot(fig3)

            st.text("Classification report:")
            st.code(classification_report(y_test, y_pred))

        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.metric("MSE", f"{mse:.4f}")
            st.metric("MAE", f"{mae:.4f}")
            st.metric("R2", f"{r2:.4f}")

            # Actual vs Predicted with perfect fit line
            fig_fit, ax_fit = plt.subplots()
            ax_fit.scatter(y_test, y_pred, alpha=0.7, label="Data points")
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax_fit.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect fit (y = x)")
            ax_fit.set_xlabel("Actual")
            ax_fit.set_ylabel("Predicted")
            ax_fit.set_title("Actual vs Predicted")
            ax_fit.legend()
            st.pyplot(fig_fit)

            # Residual plot
            resid = y_test - y_pred
            fig, ax = plt.subplots()
            ax.scatter(y_pred, resid)
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Predicted")
            st.pyplot(fig)


            # Loss curve for MLPRegressor
            if isinstance(model, MLPRegressor) and hasattr(model, "loss_curve_"):
                fig2, ax2 = plt.subplots()
                ax2.plot(model.loss_curve_)
                ax2.set_title("Training Loss (MLP)")
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Loss")
                st.pyplot(fig2)

        st.divider()
        st.subheader("🔮 Predict on Custom Input")
        st.caption("Provide feature values and get a single prediction based on the trained model.")

        with st.form("predict_form"):
            input_data = {}
            # Build inputs from TRAIN columns (pre-encoding)
            for col in X.columns:
                if col in num_cols:
                    # infer range
                    col_min = pd.to_numeric(X[col], errors='coerce').min()
                    col_max = pd.to_numeric(X[col], errors='coerce').max()
                    default_val = float(X[col].median()) if pd.notnull(X[col].median()) else 0.0
                    val = st.number_input(f"{col}", value=float(default_val))
                else:
                    # categorical options
                    opts = sorted([str(x) for x in pd.Series(X[col].astype(str).unique()).dropna().tolist()])
                    if len(opts) == 0:
                        opts = ["missing"]
                    val = st.selectbox(f"{col}", opts)
                input_data[col] = val
            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([input_data])
            try:
                pred = pipe.predict(input_df)[0]
                st.success(f"Prediction: {pred}")
                if is_classification and hasattr(pipe["model"], "predict_proba"):
                    proba = pipe.predict_proba(input_df)
                    st.write("Class probabilities:")
                    st.write(pd.DataFrame(proba, columns=[f"class_{c}" for c in pipe["model"].classes_]))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        # Unsupervised: no target column
        st.info("Unsupervised mode: **no target** is used. Preprocessing applies to all columns.")
        preprocessor, num_cols, cat_cols = build_preprocessor(df, target_col=None)

        if model_choice == "KMeans":
            model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state)

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        with st.spinner("Fitting clustering model..."):
            labels = pipe.fit_predict(df)

        st.success("Clustering done ✅")
        st.write("Cluster label counts:")
        st.write(pd.Series(labels).value_counts().sort_index())

        # Metrics (unsupervised)
        try:
            # silhouette requires > 1 cluster and > n_clusters samples
            transformed = pipe["prep"].transform(df)
            if len(np.unique(labels)) > 1 and transformed.shape[0] > len(np.unique(labels)):
                sil = silhouette_score(transformed, labels)
                st.metric("Silhouette Score", f"{sil:.4f}")
        except Exception:
            pass

        # PCA visualization
        try:
            pca = PCA(n_components=2, random_state=random_state)
            reduced = pca.fit_transform(transformed)
            fig, ax = plt.subplots()
            scatter = ax.scatter(reduced[:,0], reduced[:,1], c=labels)
            ax.set_title("PCA (2D) of Clusters")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            st.pyplot(fig)
        except Exception as e:
            st.info("PCA plot not available.")

else:
    st.info("⬅️ Upload a CSV from the sidebar to begin.")

# -------------------------------------------------------------
# Notes / Tips (displayed in an expander)
# -------------------------------------------------------------
with st.expander("Notes & Tips"):
    st.markdown(
        """
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
        - Run locally: `streamlit run app.py`
        - Suggested requirements: `streamlit pandas numpy scikit-learn matplotlib`
        """
    )










