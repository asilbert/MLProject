import numpy as np
import pandas as pd
from .encoders import build_preprocessor
from .io_utils import save_pickle

def _strip_prefixes(names):
    out = []
    for n in names:
        if ":" in n:
            out.append(n.split(":", 1)[1])
        else:
            out.append(n)
    return out

def build_design_matrices(
    ames_df: pd.DataFrame,
    *,
    numeric_cap=None,
    lower_q=0.01,
    upper_q=0.99,
    return_preprocessor=True
):
    # 1) build + fit preprocessor
    pre = build_preprocessor(ames_df, numeric_cap=numeric_cap, lower_q=lower_q, upper_q=upper_q)
    Xt = pre.fit_transform(ames_df)

    # 2) final feature names in ColumnTransformer order
    #    (works because encoders.build_preprocessor sets verbose_feature_names_out=False)
    names = pre.get_feature_names_out()
    names = _strip_prefixes(names)

    # 3) dense DataFrame if sparse
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    transformed_df = pd.DataFrame(Xt, columns=names, index=ames_df.index)

    # 4) for linear models, you already do scaling & log separately in notebook → X_final = transformed_df there if you wish
    X_final = transformed_df

    if return_preprocessor:
        return transformed_df, X_final, pre
    return transformed_df, X_final