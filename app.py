import streamlit as st
import pandas as pd
from config import setup_page, sidebar_controls
from helpers import build_preprocessor, parse_hidden_layers
from supervised import run_supervised
from unsupervised import run_unsupervised
from notes import show_notes








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





import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from helpers import build_preprocessor








from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline




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
