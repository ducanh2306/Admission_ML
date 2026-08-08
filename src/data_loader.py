"""
Load and validate the UCLA Admission dataset.
"""

import pandas as pd
from src.config import DATA_PATH, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
  
    logger.info("Loading data from: %s", path)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Dataset not found: %s", path)
        raise

    logger.info("Loaded %d rows × %d columns.", *df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    logger.info("Missing values total: %d", df.isnull().sum().sum())
    return df
