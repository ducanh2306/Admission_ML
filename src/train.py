"""
Neural network (MLPClassifier) training, cross-validation, evaluation,
and persistence.
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.config import (
    TEST_SIZE, RANDOM_STATE, MODEL_PATH, SCALER_PATH, CV_FOLDS,
    DEFAULT_HIDDEN_LAYER_SIZES, DEFAULT_ACTIVATION,
    DEFAULT_BATCH_SIZE, DEFAULT_MAX_ITER,
)
from src.preprocessing import fit_scaler, transform_scaler
from src.logger import get_logger

logger = get_logger(__name__)


# ── Train / test split ───────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info(
        "Split: train=%d  test=%d  (stratified, test_size=%.0f%%)",
        len(X_train), len(X_test), TEST_SIZE * 100,
    )
    return X_train, X_test, y_train, y_test


# ── Model trainer ─────────────────────────────────────────────────────────────

def train_mlp(
    X_train_scaled,
    y_train,
    hidden_layer_sizes: tuple = DEFAULT_HIDDEN_LAYER_SIZES,
    activation: str = DEFAULT_ACTIVATION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> MLPClassifier:

    logger.info(
        "Training MLP — layers=%s  activation=%s  batch_size=%d  max_iter=%d",
        hidden_layer_sizes, activation, batch_size, max_iter,
    )
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)
    logger.info(
        "Training complete — converged=%s  n_iter=%d  final_loss=%.4f",
        model.n_iter_ < max_iter, model.n_iter_, model.loss_,
    )
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_scaled, y_true, dataset_name: str = "Test") -> dict:
    y_pred = model.predict(X_scaled)
    acc    = accuracy_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    logger.info("%s set  →  Accuracy: %.4f", dataset_name, acc)
    logger.info("Confusion matrix (%s):\n%s", dataset_name, cm)
    return {"accuracy": acc, "confusion_matrix": cm, "report": report, "y_pred": y_pred}


def cross_validate_model(model_params: dict, X_train_scaled, y_train) -> dict:

    base_model = MLPClassifier(random_state=RANDOM_STATE, **model_params)
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(base_model, X_train_scaled, y_train, cv=kfold)
    logger.info(
        "CV-%d  mean=%.4f  std=%.4f",
        CV_FOLDS, scores.mean(), scores.std(),
    )
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}


# ── Persistence ──────────────────────────────────────────────────────────────

def save_artifacts(model, scaler):
    """Pickle the trained model and scaler to disk."""
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved model → %s", MODEL_PATH)
    logger.info("Saved scaler → %s", SCALER_PATH)


def load_artifacts():
    """Load the persisted model and scaler."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Loaded model and scaler from disk.")
    return model, scaler


# ── Full training pipeline (called by Streamlit) ─────────────────────────────

def run_training_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    hidden_layer_sizes: tuple = DEFAULT_HIDDEN_LAYER_SIZES,
    activation: str = DEFAULT_ACTIVATION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict:

    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler, X_train_scaled = fit_scaler(X_train)
    X_test_scaled = transform_scaler(scaler, X_test)

    model = train_mlp(
        X_train_scaled, y_train,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        batch_size=batch_size,
        max_iter=max_iter,
    )

    train_eval = evaluate_model(model, X_train_scaled, y_train, "Train")
    test_eval  = evaluate_model(model, X_test_scaled, y_test, "Test")

    cv_params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "batch_size": batch_size,
        "max_iter": max_iter,
    }
    cv_res = cross_validate_model(cv_params, X_train_scaled, y_train)

    save_artifacts(model, scaler)

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": list(X_train.columns),
        "train_eval": train_eval,
        "test_eval": test_eval,
        "cv": cv_res,
        "loss_curve": model.loss_curve_,
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "batch_size": batch_size,
        "max_iter": max_iter,
    }
