"""
This file contains all configuration settings for project
--> Easy to fix if anything changes in the future
"""

import os

# Paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Admission.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "data")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

MODEL_PATH  = os.path.join(MODEL_DIR, "mlp_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

#  Data settings
TARGET_COL        = "Admit_Chance"
DROP_COLS         = ["Serial_No"]
ADMIT_THRESHOLD   = 0.80          # Admit_Chance >= 0.80 -> Admit = 1
CATEGORICAL_COLS  = ["University_Rating", "Research"]

TEST_SIZE    = 0.20
RANDOM_STATE = 123

# ── Neural network settings 
DEFAULT_HIDDEN_LAYER_SIZES = (19,)
DEFAULT_ACTIVATION         = "relu"
DEFAULT_BATCH_SIZE         = 50
DEFAULT_MAX_ITER           = 200

ACTIVATION_OPTIONS = ["relu", "tanh", "logistic", "identity"]

CV_FOLDS = 5

# ── Logging
LOG_FILE  = os.path.join(LOG_DIR, "app.log")
LOG_LEVEL = "INFO"
