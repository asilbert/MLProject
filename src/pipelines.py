# src/pipelines.py
from typing import Union, Optional, Dict
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator

from .encoders import build_preprocessor

def make_linear_pipe(
    df_schema,
    *,
    model: Union[str, BaseEstimator] = "ridge",
    model_kwargs: Optional[Dict] = None,
    numeric_cap_cols=None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    log_cols=None,
    scale_features=None,
    zero_impute_cols=None,
    log_target: bool = True,
    densify: bool = True,
):
    """
    Build a full pipeline:
      preprocess (cap → zero-impute → ordinal/onehot → log+scale numerics) → linear model.
    `model` can be "ridge" | "lasso" | "enet" OR a ready sklearn estimator (e.g., RidgeCV()).
    """
    model_kwargs = model_kwargs or {}

    # --- preprocessor (fold-safe) ---
    pre = build_preprocessor(
        df_schema,
        numeric_cap=numeric_cap_cols, lower_q=lower_q, upper_q=upper_q,
        log_cols=log_cols, scale_features=scale_features,
        zero_impute_cols=zero_impute_cols,
    )

    # --- pick/regressors ---
    if isinstance(model, str):
        name = model.lower()
        if name == "ridge":
            reg = Ridge(max_iter=200_000, **model_kwargs)
        elif name == "lasso":
            reg = Lasso(max_iter=200_000, tol=1e-3, **model_kwargs)
        elif name == "enet":
            reg = ElasticNet(max_iter=200_000, tol=1e-3, **model_kwargs)
        else:
            raise ValueError("model must be 'ridge'|'lasso'|'enet' or a sklearn estimator")
    else:
        # already an estimator (e.g., RidgeCV / LassoCV / ElasticNetCV)
        reg = model

    # --- assemble steps ---
    core = Pipeline([("prep", pre), ("reg", reg)])

    # optional densify for some linear solvers if your ColumnTransformer outputs sparse
    if densify:
        core = Pipeline([("prep", pre), ("reg", reg)])  # keep as-is; scikit handles sparse OK for linear

    # --- target transform wrapper (log1p on y) ---
    if log_target:
        core = TransformedTargetRegressor(
            regressor=core,
            func=np.log1p,
            inverse_func=np.expm1
        )

    return core


def make_tree_pipe(df_schema, *, model="histgb", model_kwargs=None,
                   numeric_cap_cols=None, lower_q=0.01, upper_q=0.99,
                   # trees don’t need scaling; log of features optional—set log_cols=[] to skip
                   log_cols=None, scale_features=None,
                   zero_impute_cols=("BsmtFullBath","BsmtHalfBath"),
                   log_target=True, random_state=42):
    model_kwargs = (model_kwargs or {}) | {"random_state": random_state}

    if model == "histgb":
        reg = HistGradientBoostingRegressor(**model_kwargs)
    elif model == "rf":
        reg = RandomForestRegressor(n_jobs=-1, **model_kwargs)
    else:
        raise ValueError("model must be histgb/rf")

    pre = build_preprocessor(
        df_schema,
        numeric_cap=numeric_cap_cols, lower_q=lower_q, upper_q=upper_q,
        log_cols=log_cols, scale_features=scale_features,
        zero_impute_cols=zero_impute_cols
    )

    pipe = Pipeline([("prep", pre), ("reg", reg)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1) if log_target else pipe

def _maybe_memory(memory_cache_dir):
    return Memory(location=memory_cache_dir, verbose=0) if memory_cache_dir else None

def _linear_base_pipe(df_schema, memory_cache_dir=None, **pre_kwargs):
    pre = build_preprocessor(df_schema, **pre_kwargs)
    return Pipeline(
        steps=[("prep", pre), ("reg", Ridge())],  # placeholder reg; overwritten below
        memory=_maybe_memory(memory_cache_dir)
    )

def build_ridge_pipe(df_schema=None, *, alpha=12.0, memory_cache_dir=None, **pre_kwargs):
    pipe = _linear_base_pipe(df_schema, memory_cache_dir, **pre_kwargs)
    pipe.set_params(reg=Ridge(alpha=alpha, max_iter=200_000, tol=1e-3))
    return pipe

def build_lasso_pipe(df_schema=None, *, alpha=0.001, memory_cache_dir=None, **pre_kwargs):
    pipe = _linear_base_pipe(df_schema, memory_cache_dir, **pre_kwargs)
    pipe.set_params(reg=Lasso(alpha=alpha, max_iter=200_000, tol=1e-3))
    return pipe

def build_enet_pipe(df_schema=None, *, alpha=0.001, l1_ratio=0.5, memory_cache_dir=None, **pre_kwargs):
    pipe = _linear_base_pipe(df_schema, memory_cache_dir, **pre_kwargs)
    pipe.set_params(reg=ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=200_000, tol=1e-3))
    return pipe