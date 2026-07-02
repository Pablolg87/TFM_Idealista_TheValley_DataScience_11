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
TARGET_COLUMN = "LOG_PRICE"
NEIGHBORHOOD_COLUMN = "LOCATIONNAME"
UNIT_PRICE_COLUMN = "UNITPRICE"
AREA_COLUMN = "CONSTRUCTEDAREA"
ROOMS_COLUMN = "ROOMNUMBER"
BATHROOMS_COLUMN = "BATHNUMBER"
HAS_LIFT_COLUMN = "HASLIFT"
HAS_TERRACE_COLUMN = "HASTERRACE"
HAS_PARKING_COLUMN = "HASPARKINGSPACE"


# MVP model input contract. These are the only variables expected by predict.py
# through models/feature_columns.pkl for the provisional demo model.
MVP_FEATURE_COLUMNS = [
    NEIGHBORHOOD_COLUMN,
    AREA_COLUMN,
    ROOMS_COLUMN,
    BATHROOMS_COLUMN,
    HAS_LIFT_COLUMN,
    HAS_TERRACE_COLUMN,
    HAS_PARKING_COLUMN,
]


# Training configuration
RANDOM_STATE = 42
N_ESTIMATORS = 100
