from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def parse_hidden_layers(s: str) -> Tuple[int, ...]:
    try:
        parts = [int(p.strip()) for p in s.split(",") if p.strip()]
        return tuple(parts) if parts else (100,)
    except Exception:
        return (100,)

def build_preprocessor(df: pd.DataFrame, target_col: str | None,
                       impute_strategy_num: str, impute_strategy_cat: str,
                       scale_numeric: bool) -> Tuple[ColumnTransformer, List[str], List[str]]:
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
