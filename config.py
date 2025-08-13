import streamlit as st

def setup_page():
    st.set_page_config(page_title="AutoML Mini App", layout="wide")
    st.title("⚙️ AutoML Mini App — Supervised, Unsupervised & ANN")
    st.caption("Upload a CSV, pick a learning type and model, and get metrics & visuals. **Supervised assumes the last column is the target.**")

def sidebar_controls():
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
            model_choice = st.selectbox("Clustering Model", ["KMeans"])

        st.divider()
        st.header("3) Preprocessing")
        scale_numeric = st.checkbox("Scale numeric features (StandardScaler)", value=True)
        impute_strategy_num = st.selectbox("Numeric imputation", ["mean", "median", "most_frequent"], index=0)
        impute_strategy_cat = st.selectbox("Categorical imputation", ["most_frequent", "constant"], index=0)

        st.divider()
        st.header("4) Model Hyperparameters")
        return (file, test_size, random_state, learning_type, problem_kind, model_choice,
                scale_numeric, impute_strategy_num, impute_strategy_cat)
