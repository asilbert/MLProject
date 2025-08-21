# src/encoders.py
import numpy as np
import pandas as pd
from typing import Iterable, Optional, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .config import ORDINAL_ORDER, CAP_COLS


# ---------- fold-safe quantile capper ----------
class QuantileCapper(BaseEstimator, TransformerMixin):
    """
    Per-column quantile capping (winsorization), fold-safe.
    Fits quantiles on training data inside CV folds and applies to validation data.
    Intended to be used inside a ColumnTransformer on a (sub)set of numeric columns.
    """
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        # X is the slice of columns this transformer is applied to by ColumnTransformer
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            df = X
        else:
            # X is 2D array; make names generic to keep get_feature_names_out stable
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=self.feature_names_in_)

        self.lower_ = df.quantile(self.lower_q)
        self.upper_ = df.quantile(self.upper_q)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            df = df.clip(self.lower_, self.upper_, axis=1)
            return df.to_numpy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_in_)
            df = df.clip(self.lower_, self.upper_, axis=1)
            return df.to_numpy()

    def get_feature_names_out(self, input_features: Optional[List[str]] = None):
        # propagate names through ColumnTransformer.get_feature_names_out
        return np.array(self.feature_names_in_, dtype=object)


def _fill_none(X: pd.DataFrame) -> pd.DataFrame:
    return X.infer_objects(copy=False).fillna("None")


def build_preprocessor(
    df: pd.DataFrame,
    *,
    numeric_cap: Optional[Iterable[str]] = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99
):
    """
    Builds your original preprocessor, with optional quantile capping on a subset of numerics.

    Args:
        df: engineered dataframe
        numeric_cap: subset of numeric columns to cap (winsorize). If None or empty, no capping is applied.
        lower_q, upper_q: quantiles for capping (default 1% and 99%)
    """
    # detect types after feature engineering
    numeric_features_all = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # define zero-fill numeric columns
    zero_fill_numeric = [
        "BsmtFullBath","BsmtHalfBath","FullBath","HalfBath",
        "TotalBsmtSF","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF",
        "GarageCars","GarageArea",
        "Fireplaces","TotRmsAbvGrd","KitchenAbvGr"
    ]
    zero_fill_numeric = [c for c in zero_fill_numeric if c in numeric_features_all]

    
    categorical_all = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # separate ordinal vs nominal
    ordinal_features = [c for c in ORDINAL_ORDER.keys() if c in df.columns and c != "Electrical"]
    nominal_features = [c for c in categorical_all if c not in ORDINAL_ORDER.keys()]

    # split rest into capped vs noncapped
    numeric_cap = list(numeric_cap) if numeric_cap else []
    numeric_cap = [c for c in numeric_cap if c in numeric_features_all]
    numeric_nocap = [c for c in numeric_features_all if c not in numeric_cap + zero_fill_numeric]

    # specific Electrical pipeline (ordinal with impute-most-frequent)
    electrical_pipe = Pipeline(steps=[
        ("impute_electrical", SimpleImputer(strategy="most_frequent")),
        ("ordinal_electrical", OrdinalEncoder(categories=[ORDINAL_ORDER["Electrical"]]))
    ])

    # numeric pipelines
    num_plain_pipe = Pipeline([("impute_mean", SimpleImputer(strategy="mean"))])

    num_cap_pipe = Pipeline([
        ("cap", QuantileCapper(lower_q=lower_q, upper_q=upper_q)),
        ("impute_mean", SimpleImputer(strategy="mean")),
    ])


    num_zero_pipe = Pipeline([
        ("impute_zero", SimpleImputer(strategy="constant", fill_value=0))
    ])


    # ordinal pipe
    ordinal_pipe = Pipeline(steps=[
        ("impute_none", SimpleImputer(strategy="constant", fill_value="None")),
        ("ordinal", OrdinalEncoder(categories=[ORDINAL_ORDER[f] for f in ordinal_features]))
    ])

    # nominal pipe
    nominal_pipe = Pipeline(steps=[
        ("impute_none", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    transformers = []
    if "Electrical" in df.columns:
        transformers.append(("electrical", electrical_pipe, ["Electrical"]))
        nominal_features = [c for c in nominal_features if c != "Electrical"]

    if numeric_cap:
        transformers.append(("num_cap", num_cap_pipe, numeric_cap))
    if numeric_nocap:
        transformers.append(("num", num_plain_pipe, numeric_nocap))
    if zero_fill_numeric:
        transformers.append(("num_zero", num_zero_pipe, zero_fill_numeric))
    if ordinal_features:
        transformers.append(("ordinal", ordinal_pipe, ordinal_features))
    if nominal_features:
        transformers.append(("nominal", nominal_pipe, nominal_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,         # your current setting
        verbose_feature_names_out=True
    )
    return preprocessor