"""
tests/test_pipeline.py
Unit tests for the UCLA Admission Neural Network pipeline.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_data
from src.preprocessing import (
    binarize_target, drop_unused_columns, encode_features,
    split_features_target, fit_scaler,
)
from src.config import DATA_PATH, TARGET_COL, ADMIT_THRESHOLD


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def processed_df(raw_df):
    df = binarize_target(raw_df)
    df = drop_unused_columns(df)
    df = encode_features(df)
    return df


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_shape(self, raw_df):
        assert raw_df.shape == (500, 9)

    def test_no_missing(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_target_column_exists(self, raw_df):
        assert TARGET_COL in raw_df.columns

    def test_target_is_continuous(self, raw_df):
        assert raw_df[TARGET_COL].between(0, 1).all()


class TestPreprocessing:
    def test_binarize_produces_binary(self, raw_df):
        df = binarize_target(raw_df)
        assert set(df[TARGET_COL].unique()).issubset({0, 1})

    def test_binarize_threshold_correct(self, raw_df):
        df = binarize_target(raw_df, threshold=ADMIT_THRESHOLD)
        expected = (raw_df[TARGET_COL] >= ADMIT_THRESHOLD).astype(int)
        assert (df[TARGET_COL] == expected).all()

    def test_drop_serial_no(self, raw_df):
        df = binarize_target(raw_df)
        df = drop_unused_columns(df)
        assert "Serial_No" not in df.columns

    def test_encode_creates_dummies(self, processed_df):
        dummy_cols = [c for c in processed_df.columns if c.startswith("University_Rating_")]
        assert len(dummy_cols) > 0
        research_cols = [c for c in processed_df.columns if c.startswith("Research_")]
        assert len(research_cols) > 0

    def test_feature_split_shapes(self, processed_df):
        X, y = split_features_target(processed_df)
        assert X.shape[0] == y.shape[0] == 500
        assert TARGET_COL not in X.columns

    def test_scaler_output_range(self, processed_df):
        X, y = split_features_target(processed_df)
        scaler, X_scaled = fit_scaler(X)
        assert X_scaled.min() >= -1e-9
        assert X_scaled.max() <= 1 + 1e-9

    def test_admit_rate_reasonable(self, raw_df):
        df = binarize_target(raw_df)
        rate = df[TARGET_COL].mean()
        # Expect roughly 30% admit rate per notebook observations
        assert 0.2 <= rate <= 0.5
