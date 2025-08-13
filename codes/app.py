import streamlit as st
import pandas as pd
from config import setup_page, sidebar_controls
from helpers import build_preprocessor, parse_hidden_layers
from supervised import run_supervised
from unsupervised import run_unsupervised
from notes import show_notes

setup_page()

(file, test_size, random_state, learning_type, problem_kind, model_choice,
 scale_numeric, impute_strategy_num, impute_strategy_cat) = sidebar_controls()

if file is not None:
    df = pd.read_csv(file)
    st.subheader("Preview")
    st.dataframe(df.head(20))

    if learning_type == "Supervised":
        run_supervised(df, problem_kind, model_choice, test_size, random_state,
                       scale_numeric, impute_strategy_num, impute_strategy_cat)
    else:
        run_unsupervised(df, model_choice, random_state,
                         scale_numeric, impute_strategy_num, impute_strategy_cat)
else:
    st.info("⬅️ Upload a CSV from the sidebar to begin.")

show_notes()
