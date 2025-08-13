import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from helpers import build_preprocessor, parse_hidden_layers

def run_supervised(df, problem_kind, model_choice, test_size, random_state,
                   scale_numeric, impute_strategy_num, impute_strategy_cat):

    df = df.dropna()
    target_col = df.columns[-1]
    st.info(f"Supervised mode: using **last column** as target → {target_col}")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    preprocessor, num_cols, cat_cols = build_preprocessor(df, target_col,
                                                          impute_strategy_num, impute_strategy_cat, scale_numeric)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() < 20 else None
    )

    is_classification = (problem_kind == "Classification")
    model = None

    # -----------------------
    # Model selection
    # -----------------------
    if is_classification:
        if model_choice == "Logistic Regression":
            C = st.sidebar.number_input("C (inverse regularization)", min_value=1e-4, value=1.0, step=0.1, format="%f")
            max_iter = st.sidebar.number_input("max_iter", min_value=100, value=500, step=50)
            model = LogisticRegression(C=C, max_iter=max_iter)
        elif model_choice == "SVM (SVC)":
            C = st.sidebar.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f")
            kernel = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
            probability = st.sidebar.checkbox("Enable probability estimates (slower)", value=True)
            model = SVC(C=C, kernel=kernel, probability=probability)
        elif model_choice == "KNN (Classifier)":
            n_neighbors = st.sidebar.slider("n_neighbors", 1, 25, 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif model_choice == "ANN (MLPClassifier)":
            hidden = st.sidebar.text_input("Hidden layers (e.g., 100 or 64,32)", value="100")
            activation = st.sidebar.selectbox("activation", ["relu", "tanh", "logistic"], index=0)
            alpha = st.sidebar.number_input("alpha (L2)", min_value=0.0, value=0.0001, step=0.0001, format="%f")
            max_iter = st.sidebar.number_input("max_iter", min_value=100, value=300, step=50)
            model = MLPClassifier(hidden_layer_sizes=parse_hidden_layers(hidden), activation=activation,
                                  alpha=alpha, max_iter=max_iter, random_state=random_state)
    else:
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "SVM (SVR)":
            C = st.sidebar.number_input("C", min_value=1e-4, value=1.0, step=0.1, format="%f")
            kernel = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
            model = SVR(C=C, kernel=kernel)
        elif model_choice == "KNN (Regressor)":
            n_neighbors = st.sidebar.slider("n_neighbors", 1, 25, 5)
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        elif model_choice == "ANN (MLPRegressor)":
            hidden = st.sidebar.text_input("Hidden layers (e.g., 100 or 64,32)", value="64,32")
            activation = st.sidebar.selectbox("activation", ["relu", "tanh", "logistic"], index=0)
            alpha = st.sidebar.number_input("alpha (L2)", min_value=0.0, value=0.0001, step=0.0001, format="%f")
            max_iter = st.sidebar.number_input("max_iter", min_value=100, value=300, step=50)
            model = MLPRegressor(hidden_layer_sizes=parse_hidden_layers(hidden), activation=activation,
                                 alpha=alpha, max_iter=max_iter, random_state=random_state)

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    with st.spinner("Training..."):
        pipe.fit(X_train, y_train)
    st.success("Model trained ✅")

    y_pred = pipe.predict(X_test)

    # -----------------------
    # Classification metrics
    # -----------------------
    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        st.metric("Accuracy", f"{acc:.4f}")
        st.metric("F1 (weighted)", f"{f1:.4f}")

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

        if len(np.unique(y_test)) == 2:
            try:
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
            except Exception:
                st.info("ROC not available for this model.")

        if isinstance(model, MLPClassifier) and hasattr(model, "loss_curve_"):
            fig3, ax3 = plt.subplots()
            ax3.plot(model.loss_curve_)
            ax3.set_title("Training Loss (MLP)")
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Loss")
            st.pyplot(fig3)

        st.text("Classification report:")
        st.code(classification_report(y_test, y_pred))

    # -----------------------
    # Regression metrics
    # -----------------------
    else:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.metric("MSE", f"{mse:.4f}")
        st.metric("MAE", f"{mae:.4f}")
        st.metric("R2", f"{r2:.4f}")

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

        resid = y_test - y_pred
        fig, ax = plt.subplots()
        ax.scatter(y_pred, resid)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Predicted")
        st.pyplot(fig)

        if isinstance(model, MLPRegressor) and hasattr(model, "loss_curve_"):
            fig2, ax2 = plt.subplots()
            ax2.plot(model.loss_curve_)
            ax2.set_title("Training Loss (MLP)")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Loss")
            st.pyplot(fig2)

    # -----------------------
    # Predict custom input
    # -----------------------
    st.divider()
    st.subheader("🔮 Predict on Custom Input")
    st.caption("Provide feature values and get a single prediction based on the trained model.")

    with st.form("predict_form"):
        input_data = {}
        for col in X.columns:
            if col in num_cols:
                default_val = float(X[col].median()) if pd.notnull(X[col].median()) else 0.0
                val = st.number_input(f"{col}", value=default_val)
            else:
                opts = sorted([str(x) for x in pd.Series(X[col].astype(str).unique()).dropna().tolist()])
                opts = opts if opts else ["missing"]
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
