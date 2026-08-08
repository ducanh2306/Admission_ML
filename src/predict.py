"""
Single-applicant admission inference using the persisted MLP model.
"""

from src.logger import get_logger
from src.train import load_artifacts
from src.preprocessing import preprocess_single_input

logger = get_logger(__name__)


def predict_admission(raw_input: dict, feature_columns: list) -> dict:
 
    try:
        model, scaler = load_artifacts()
    except FileNotFoundError:
        logger.error("Model artifacts not found. Run training first.")
        raise

    X = preprocess_single_input(raw_input, feature_columns, scaler)
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    label = "Admit" if pred == 1 else "Reject"
    logger.info("Prediction: %s  (P(admit)=%.4f)", label, proba)

    return {"prediction": pred, "label": label, "probability": proba}
