"""
Target binarisation, categorical encoding, and scaling for the admission
neural network pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    TARGET_COL, DROP_COLS, ADMIT_THRESHOLD, CATEGORICAL_COLS,
)
from src.logger import get_logger

logger = get_logger(__name__)


def binarize_target(df: pd.DataFrame, threshold: float = ADMIT_THRESHOLD) -> pd.DataFrame:

    df = df.copy()
    df[TARGET_COL] = (df[TARGET_COL] >= threshold).astype(int)
    rate = df[TARGET_COL].mean()
    logger.info(
        "Target binarized at threshold=%.2f — admit rate=%.1f%%",
        threshold, rate * 100,
    )
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns not used for modelling."""
    df = df.copy()
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.debug("Dropped column: %s", col)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("object")

    ohe_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, dtype="int")
    logger.info("Shape after one-hot encoding: %s", df.shape)
    return df


def split_features_target(df: pd.DataFrame):

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.info("Features: %d  |  Target: %s", X.shape[1], TARGET_COL)
    return X, y


def fit_scaler(X_train: pd.DataFrame):

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    logger.info("Scaler fitted on %d training samples.", X_train.shape[0])
    return scaler, X_scaled


def transform_scaler(scaler: MinMaxScaler, X: pd.DataFrame) -> np.ndarray:

    return scaler.transform(X)


def preprocess_single_input(raw_dict: dict, feature_columns: list,
                             scaler: MinMaxScaler) -> np.ndarray:

    df_input = pd.DataFrame([raw_dict])

    for col in CATEGORICAL_COLS:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype("object")

    ohe_cols = [c for c in CATEGORICAL_COLS if c in df_input.columns]
    df_input = pd.get_dummies(df_input, columns=ohe_cols, dtype="int")

    # Align to training feature set (missing dummy columns filled with 0)
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    return scaler.transform(df_input)
