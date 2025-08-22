from pathlib import Path
import json
import pandas as pd
import joblib

def save_parquet(df: pd.DataFrame, path: str, index: bool = False):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=index)

def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def save_json(obj, path: str):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def save_pickle(obj, path: str):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)

def load_pickle(path: str):
    return joblib.load(path)