# src/features.py
import pandas as pd
import numpy as np

RAW_YEAR_COLS = ["YearBuilt","YearRemodAdd","GarageYrBlt"]

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    # Derived areas
    X["GrLivArea_per_LotArea"] = X["GrLivArea"] / (X["LotArea"] + 1)
    X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
    
    # Baths aggregate (uses 0.5 weight as your baseline)
    bath_cols = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
    X[bath_cols] = X[bath_cols].fillna(0)
    X["TotalBaths"] = X["FullBath"] + 0.5*X["HalfBath"] + X["BsmtFullBath"] + 0.5*X["BsmtHalfBath"]

    # Year-derived ages (ensure ints)
    X["HouseAge"] = X["YrSold"].astype(int) - X["YearBuilt"].astype(int)
    X["HouseAgeSinceRemod"] = X["YrSold"].astype(int) - X["YearRemodAdd"].astype(int)
    X["GarageAge"] = X["YrSold"].astype(int) - X["GarageYrBlt"]
    X["WasRemodeled"] = (X["YearRemodAdd"].astype(int) > X["YearBuilt"].astype(int))

    # Cast some to categorical (as you did)
    X["MSSubClass"] = X["MSSubClass"].astype("object")
    X["YrSold"] = X["YrSold"].astype("object")
    X["MoSold"] = X["MoSold"].astype("object")
    X["WasRemodeled"] = X["WasRemodeled"].astype("object")

    # Drop raw year columns if present
    X = X.drop(columns=[c for c in RAW_YEAR_COLS if c in X.columns], errors="ignore")
    X = X.drop(columns=["PID", "SalePrice"])
    return X

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    if all(c in X.columns for c in ["GrLivArea","OverallQual"]):
        X["GrLivArea_x_Qual"] = X["GrLivArea"] * X["OverallQual"]
    if all(c in X.columns for c in ["TotalSF","OverallQual"]):
        X["TotalSF_x_Qual"] = X["TotalSF"] * X["OverallQual"]
    if all(c in X.columns for c in ["TotalBaths","GarageCars"]):
        X["Baths_x_Garage"] = X["TotalBaths"] * X["GarageCars"]
    if all(c in X.columns for c in ["HouseAge","OverallQual"]):
        X["Age_x_Qual"] = X["HouseAge"] * X["OverallQual"]
    if all(c in X.columns for c in ["HouseAgeSinceRemod","OverallQual"]):
        X["AgeRemod_x_Qual"] = X["HouseAgeSinceRemod"] * X["OverallQual"]
    return X

def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    if all(c in X.columns for c in ["TotalBaths","BedroomAbvGr"]):
        X["Baths_per_Bed"] = X["TotalBaths"] / (X["BedroomAbvGr"] + 1)
    if all(c in X.columns for c in ["GrLivArea","TotRmsAbvGrd"]):
        X["GrLiv_per_Room"] = X["GrLivArea"] / (X["TotRmsAbvGrd"] + 1)
    return X