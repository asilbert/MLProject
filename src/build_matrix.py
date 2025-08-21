# src/build_matrix.py
import numpy as np
import pandas as pd
from collections import Counter
from .encoders import build_preprocessor  # your ColumnTransformer builder

def _short_tag(left_name: str) -> str:
    """Map transformer branch to a short tag for rare collision cases."""
    return {
        "num_cap": "cap",
        "num": "num",
        "ordinal": "ord",
        "nominal": "oh",
        "electrical": "elect",
    }.get(left_name, left_name)

def build_design_matrices(df: pd.DataFrame, **pre_kwargs):
    """
    Build design matrices with readable, mostly prefix-free names.
    - Strips ColumnTransformer prefixes (e.g., 'nominal__', 'num__').
    - Keeps plain OHE names like 'Neighborhood_OldTown', 'SaleType_WD'.
    - Re-attaches a short tag ONLY if needed to resolve duplicate names.
    Returns: transformed_df, X_final, scaler (None here)
    """
    pre = build_preprocessor(df, **pre_kwargs)
    Xt = pre.fit_transform(df)

    # Raw names from the fitted ColumnTransformer/Pipelines
    raw_names = pre.get_feature_names_out()

    # 1) Strip branch prefixes like 'nominal__', 'num__', etc.
    #    (OHE already emits 'Feature_Category' if we passed input features)
    base_names = [n.split("__", 1)[1] if "__" in n else n for n in raw_names]

    # 2) Detect duplicates after stripping
    counts = Counter(base_names)

    # 3) For duplicates only, re-attach a short tag (e.g., 'cap:LotArea')
    clean_names = []
    for raw, base in zip(raw_names, base_names):
        if counts[base] == 1:
            clean_names.append(base)  # unique → no tag
        else:
            # collision: add a minimal tag from the left side of 'left__right'
            left = raw.split("__", 1)[0] if "__" in raw else "x"
            clean_names.append(f"{_short_tag(left)}:{base}")

    # 4) Final uniqueness guard (very rare): append counters if still duped
    if len(set(clean_names)) != len(clean_names):
        seen = {}
        uniq = []
        for n in clean_names:
            if n not in seen:
                seen[n] = 0
                uniq.append(n)
            else:
                seen[n] += 1
                uniq.append(f"{n}#{seen[n]}")
        clean_names = uniq

    # 5) Dense dataframe if Xt is sparse
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    transformed_df = pd.DataFrame(Xt, columns=clean_names, index=df.index)

    # If you also build a scaled numeric matrix elsewhere, you can return that here instead.
    X_final = transformed_df
    scaler = None
    return transformed_df, X_final, scaler