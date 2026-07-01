"""Prediction layer for the Smart Advisor MVP.

The trained model uses LOG_PRICE as target. This module hides that detail from
the rest of the application: predict_price always returns a real price in euros.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS_PATH, MODEL_PATH


_MODEL_CACHE: dict[Path, Any] = {}
_FEATURE_COLUMNS_CACHE: dict[Path, list[str]] = {}


def _ensure_file_exists(path: Path, artifact_name: str) -> None:
    """Raise a clear error when a required artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{artifact_name} not found at: {path}")


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the trained model from disk using an in-memory cache."""
    _ensure_file_exists(model_path, "Model file")
    cache_key = model_path.resolve()

    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = joblib.load(model_path)

    return _MODEL_CACHE[cache_key]


def load_feature_columns(
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
) -> list[str]:
    """Load the model feature column names using an in-memory cache."""
    _ensure_file_exists(feature_columns_path, "Feature columns file")
    cache_key = feature_columns_path.resolve()

    if cache_key in _FEATURE_COLUMNS_CACHE:
        return _FEATURE_COLUMNS_CACHE[cache_key]

    feature_columns = joblib.load(feature_columns_path)

    if isinstance(feature_columns, str):
        raise TypeError("Feature columns artifact must be a sequence of names.")

    try:
        columns = [str(column) for column in feature_columns]
    except TypeError as exc:
        raise TypeError(
            "Feature columns artifact must be a sequence of names."
        ) from exc

    if not columns:
        raise ValueError("Feature columns artifact cannot be empty.")

    _FEATURE_COLUMNS_CACHE[cache_key] = columns
    return columns


def validate_input_columns(
    input_data: Mapping[str, Any],
    feature_columns: Sequence[str],
) -> None:
    """Validate that input data contains every feature required by the model."""
    missing_columns = [
        column for column in feature_columns if column not in input_data
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing input columns for prediction: {missing}")


def build_input_dataframe(
    input_data: Mapping[str, Any],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Build a one-row DataFrame with columns ordered for the model."""
    validate_input_columns(input_data, feature_columns)
    ordered_values = {
        column: input_data[column] for column in feature_columns
    }
    return pd.DataFrame([ordered_values], columns=list(feature_columns))


def predict_price(input_data: Mapping[str, Any]) -> float:
    """Predict a property price in euros from model features.

    The model output is LOG_PRICE. The returned value is converted to PRICE
    with numpy.expm1 so callers never need to handle logarithmic prices.
    """
    model = load_model()
    feature_columns = load_feature_columns()
    input_dataframe = build_input_dataframe(input_data, feature_columns)
    log_price_prediction = model.predict(input_dataframe)

    if len(log_price_prediction) == 0:
        raise ValueError("Model returned an empty prediction.")

    return float(np.expm1(log_price_prediction[0]))
