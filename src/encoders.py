# src/encoders.py
import numpy as np
import pandas as pd
from typing import Iterable, Optional, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from .config import ORDINAL_ORDER

# ---------- fold-safe quantile capper (kept as-is) ----------
class QuantileCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.feature_names_in_ = list(df.columns)
        self.lower_ = df.quantile(self.lower_q)
        self.upper_ = df.quantile(self.upper_q)
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None)) if not isinstance(X, pd.DataFrame) else X.copy()
        df = df.clip(self.lower_, self.upper_, axis=1)
        return df.to_numpy()

    def get_feature_names_out(self, input_features: Optional[List[str]] = None):
        return np.array(self.feature_names_in_, dtype=object)

# ---------- main preprocessor ----------
def build_preprocessor(
    df: pd.DataFrame,
    *,
    numeric_cap: Optional[Iterable[str]] = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    log_cols: Optional[Iterable[str]] = None,
    scale_features: Optional[Iterable[str]] = None,
    zero_impute_cols: Optional[Iterable[str]] = None,
):
    """
    Build a fold-safe ColumnTransformer:
      - Categorical: impute 'None' then encode (no manual string-casting needed)
      - Ordinal: impute 'None', encode with fixed category order, unknown-> -1
      - Numeric: optional quantile capping, optional log1p, optional scaling
      - Specific numeric zero-impute cols (e.g., BsmtFullBath/HalfBath) handled explicitly
    """

    # ------- column discovery -------
    numeric_all = df.select_dtypes(include=["number"]).columns.tolist()
    cat_all     = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Ordinal features you defined in ORDINAL_ORDER (except special-cased Electrical)
    ordinal_features = [c for c in ORDINAL_ORDER.keys() if c in df.columns and c != "Electrical"]
    nominal_features = [c for c in cat_all if c not in ORDINAL_ORDER.keys() and c != "Electrical"]

    # Controls
    numeric_cap = set(numeric_cap or [])
    log_cols    = set(log_cols or [])
    scale_features = set(scale_features or [])
    zero_impute_cols = set(zero_impute_cols or ["BsmtFullBath", "BsmtHalfBath"])

    # Keep only columns that actually exist
    numeric_cap = [c for c in numeric_cap if c in numeric_all]
    zero_impute_cols = [c for c in zero_impute_cols if c in numeric_all]

    # Remove zero-impute columns from other numeric groups so they’re handled once
    numeric_all_minus_zero = [c for c in numeric_all if c not in zero_impute_cols]

    # ------- split numeric into groups (cap vs no-cap) x (log+scale / scale / passthrough) -------
    def _intersect(cols, *sets):
        s = set(cols)
        for z in sets: s &= set(z)
        return [c for c in s]

    def _diff(cols, *sets):
        s = set(cols)
        for z in sets: s -= set(z)
        return [c for c in s]

    # CAP groups
    cap_and_log_scale = _intersect(numeric_cap, log_cols, scale_features)
    cap_scale_only    = _intersect(numeric_cap, scale_features)  # includes log cols; will remove next
    cap_scale_only    = [c for c in cap_scale_only if c not in cap_and_log_scale]
    cap_passthrough   = _diff(numeric_cap, scale_features)

    # NO-CAP groups
    nocap = [c for c in numeric_all_minus_zero if c not in numeric_cap]
    nocap_and_log_scale = _intersect(nocap, log_cols, scale_features)
    nocap_scale_only    = _intersect(nocap, scale_features)
    nocap_scale_only    = [c for c in nocap_scale_only if c not in nocap_and_log_scale]
    nocap_passthrough   = _diff(nocap, scale_features)

    # ------- pipes -------
    # numeric base steps
    impute_mean = SimpleImputer(strategy="mean")

    cap_step = QuantileCapper(lower_q=lower_q, upper_q=upper_q)
    log_step = FunctionTransformer(np.log1p, validate=False)
    scale_step = StandardScaler()

    # With cap
    num_cap_log_scale_pipe = Pipeline([
        ("cap", cap_step),
        ("impute", impute_mean),
        ("log1p", log_step),
        ("scale", scale_step),
    ])
    num_cap_scale_pipe = Pipeline([
        ("cap", cap_step),
        ("impute", impute_mean),
        ("scale", scale_step),
    ])
    num_cap_plain_pipe = Pipeline([
        ("cap", cap_step),
        ("impute", impute_mean),
    ])

    # Without cap
    num_log_scale_pipe = Pipeline([
        ("impute", impute_mean),
        ("log1p", log_step),
        ("scale", scale_step),
    ])
    num_scale_pipe = Pipeline([
        ("impute", impute_mean),
        ("scale", scale_step),
    ])
    num_plain_pipe = Pipeline([
        ("impute", impute_mean),
    ])

    # Zero-impute explicit numeric cols
    num_zero_pipe = Pipeline([
        ("impute_zero", SimpleImputer(strategy="constant", fill_value=0.0))
    ])

    # Categorical: impute -> encode
    nominal_pipe = Pipeline([
        ("impute_none", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    ordinal_pipe = Pipeline([
        ("impute_none", SimpleImputer(strategy="constant", fill_value="None")),
        ("ordinal", OrdinalEncoder(
            categories=[ORDINAL_ORDER[c] for c in ordinal_features],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    electrical_pipe = None
    if "Electrical" in df.columns:
        electrical_pipe = Pipeline([
            ("impute_mf", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                categories=[ORDINAL_ORDER["Electrical"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])

    # ------- assemble ColumnTransformer -------
    transformers = []

    if electrical_pipe is not None:
        transformers.append(("elect", electrical_pipe, ["Electrical"]))

    if cap_and_log_scale: transformers.append(("num_cap_log_scale", num_cap_log_scale_pipe, cap_and_log_scale))
    if cap_scale_only:    transformers.append(("num_cap_scale",     num_cap_scale_pipe,    cap_scale_only))
    if cap_passthrough:   transformers.append(("num_cap_plain",     num_cap_plain_pipe,    cap_passthrough))

    if nocap_and_log_scale: transformers.append(("num_log_scale", num_log_scale_pipe, nocap_and_log_scale))
    if nocap_scale_only:    transformers.append(("num_scale",     num_scale_pipe,     nocap_scale_only))
    if nocap_passthrough:   transformers.append(("num_plain",     num_plain_pipe,     nocap_passthrough))

    if zero_impute_cols:
        transformers.append(("num_zero", num_zero_pipe, zero_impute_cols))

    if ordinal_features:
        transformers.append(("ordinal", ordinal_pipe, ordinal_features))

    if nominal_features:
        transformers.append(("nominal", nominal_pipe, nominal_features))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False  # keep clean names, no "step__" prefixes
    )
    return pre