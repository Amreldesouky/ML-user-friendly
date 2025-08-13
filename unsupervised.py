import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from helpers import build_preprocessor

def run_unsupervised(df, model_choice, random_state,
                     scale_numeric, impute_strategy_num, impute_strategy_cat):

    st.info("Unsupervised mode: **no target** is used. Preprocessing applies to all columns.")
    preprocessor, num_cols, cat_cols = build_preprocessor(df, target_col=None,
                                                          impute_strategy_num=impute_strategy_num,
                                                          impute_strategy_cat=impute_strategy_cat,
                                                          scale_numeric=scale_numeric)

    if model_choice == "KMeans":
        n_clusters = st.sidebar.slider("n_clusters", 2, 10, 3)
        init = st.sidebar.selectbox("init", ["k-means++", "random"], index=0)
        n_init = st.sidebar.number_input("n_init", min_value=1, value=10, step=1)
        max_iter = st.sidebar.number_input("max_iter", min_value=10, value=300, step=10)
        model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init,
                       max_iter=max_iter, random_state=random_state)

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    with st.spinner("Fitting clustering model..."):
        labels = pipe.fit_predict(df)

    st.success("Clustering done ✅")
    st.write("Cluster label counts:")
    st.write(pd.Series(labels).value_counts().sort_index())

    try:
        transformed = pipe["prep"].transform(df)
        if len(np.unique(labels)) > 1 and transformed.shape[0] > len(np.unique(labels)):
            sil = silhouette_score(transformed, labels)
            st.metric("Silhouette Score", f"{sil:.4f}")
    except Exception:
        pass

    try:
        pca = PCA(n_components=2, random_state=random_state)
        reduced = pca.fit_transform(transformed)
        fig, ax = plt.subplots()
        scatter = ax.scatter(reduced[:,0], reduced[:,1], c=labels)
        ax.set_title("PCA (2D) of Clusters")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)
    except Exception:
        st.info("PCA plot not available.")
