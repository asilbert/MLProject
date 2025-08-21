# src/config.py
RANDOM_STATE = 42

# Ordinal category order (same as your dict)
ORDINAL_ORDER = {
    'Electrical': ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'],
    'LotShape': ['IR3', 'IR2', 'IR1', 'Reg'],
    'Utilities': ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],
    'LandSlope': ['Sev', 'Mod', 'Gtl'],
    'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    'FireplaceQu': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageFinish': ['None', 'Unf', 'RFn', 'Fin'],
    'GarageQual': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageCond': ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'PavedDrive': ['N', 'P', 'Y'],
    'PoolQC': ['None', 'Fa', 'TA', 'Gd', 'Ex'],
    'Fence': ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
}

# Numeric columns you scale (as per your chunk; presence-checked later)
SCALE_FEATURES = [
    # Continuous
    "GrLivArea","LotFrontage","LotArea","MasVnrArea",
    "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
    "1stFlrSF","2ndFlrSF","LowQualFinSF","GarageArea",
    "WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch",
    "ScreenPorch","PoolArea","MiscVal","TotalSF","GrLivArea_per_LotArea",
    # Counts
    "BsmtFullBath","BsmtHalfBath","FullBath","HalfBath",
    "BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd",
    "Fireplaces","GarageCars","TotalBaths",
    # Ordinal-as-numeric
    "OverallQual","OverallCond",
    # Year/ages
    "HouseAge", "HouseAgeSinceRemod", "GarageAge"
]

# Numeric columns to log1p before scaling (subset of above)
LOG_COLS = [
    "LotArea","LotFrontage","MasVnrArea",
    "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
    "1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea",
    "GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch",
    "3SsnPorch","ScreenPorch","PoolArea","MiscVal","TotalSF","GrLivArea_per_LotArea"
]



# Numeric columns to be capped
CAP_COLS = [
    "LotArea","LotFrontage","MasVnrArea","TotalBsmtSF","BsmtFinSF1","BsmtFinSF2",
    "BsmtUnfSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","GarageArea",
    "WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch",
    "PoolArea","MiscVal","TotalSF","GrLivArea_per_LotArea"
]