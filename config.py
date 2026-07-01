"""Central configuration for the Smart Advisor MVP.

This module contains shared paths and column names. It does not include
business logic, model loading, training, prediction, or Streamlit code.
"""

from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


# Data and model artifacts
DATASET_PATH = DATA_DIR / "dataset.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"


# Dataset columns
PRICE_COLUMN = "PRICE"
LOG_PRICE_TARGET_COLUMN = "LOG_PRICE"
NEIGHBORHOOD_COLUMN = "LOCATIONNAME"
UNIT_PRICE_COLUMN = "UNITPRICE"
AREA_COLUMN = "CONSTRUCTEDAREA"


# Backward-compatible name used by the real estate analysis layer.
# advisor.py works with real prices, not with the model target.
TARGET_COLUMN = PRICE_COLUMN


# Training configuration
RANDOM_STATE = 42
N_ESTIMATORS = 100
