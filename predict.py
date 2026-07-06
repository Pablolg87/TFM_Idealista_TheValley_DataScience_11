"""Prediction layer for the Smart Advisor MVP.

The definitive model uses LOG_PRICE as target and expects engineered features.
This module hides those details from the rest of the application: predict_price
always accepts business-level property inputs and returns a real price in euros.
"""

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import config
from config import FEATURE_COLUMNS_PATH, MODEL_PACKAGE_PATH, MODEL_PATH


_MODEL_CACHE: dict[Path, Any] = {}
_FEATURE_COLUMNS_CACHE: dict[Path, list[str]] = {}
_PACKAGE_CACHE: dict[Path, dict[str, Any]] = {}


def _ensure_file_exists(path: Path, artifact_name: str) -> None:
    """Raise a clear error when a required artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{artifact_name} not found at: {path}")


def _load_model_package(package_path: Path = MODEL_PACKAGE_PATH) -> dict[str, Any] | None:
    """Load the definitive model package when it is available."""
    if not package_path.exists():
        return None

    cache_key = package_path.resolve()
    if cache_key not in _PACKAGE_CACHE:
        package = joblib.load(package_path)
        if not isinstance(package, dict):
            raise TypeError("Model package must be a dictionary.")
        if "model" not in package or "features" not in package:
            raise ValueError("Model package must contain 'model' and 'features'.")
        _PACKAGE_CACHE[cache_key] = package

    return _PACKAGE_CACHE[cache_key]


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the trained model from disk using an in-memory cache."""
    package = _load_model_package()
    if package is not None:
        return package["model"]

    _ensure_file_exists(model_path, "Model file")
    cache_key = model_path.resolve()

    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = joblib.load(model_path)

    return _MODEL_CACHE[cache_key]


def _coerce_feature_columns(feature_columns: Any) -> list[str]:
    """Validate and normalize feature column names."""
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

    return columns


def load_feature_columns(
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
) -> list[str]:
    """Load the model feature column names using an in-memory cache."""
    package = _load_model_package()
    if package is not None:
        return _coerce_feature_columns(package["features"])

    _ensure_file_exists(feature_columns_path, "Feature columns file")
    cache_key = feature_columns_path.resolve()

    if cache_key not in _FEATURE_COLUMNS_CACHE:
        _FEATURE_COLUMNS_CACHE[cache_key] = _coerce_feature_columns(
            joblib.load(feature_columns_path)
        )

    return _FEATURE_COLUMNS_CACHE[cache_key]


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    """Load the dataset used to derive neighborhood-level model features."""
    _ensure_file_exists(config.DATASET_PATH, "Dataset file")
    dataset = pd.read_csv(config.DATASET_PATH)
    if dataset.empty:
        raise ValueError("Dataset cannot be empty.")
    return dataset


def _filter_neighborhood(dataset: pd.DataFrame, neighborhood: str) -> pd.DataFrame:
    """Return rows from the requested neighborhood using exact normalized match."""
    if config.NEIGHBORHOOD_COLUMN not in dataset.columns:
        raise ValueError(f"Dataset missing column: {config.NEIGHBORHOOD_COLUMN}")

    normalized = str(neighborhood).strip().casefold()
    rows = dataset[
        dataset[config.NEIGHBORHOOD_COLUMN]
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq(normalized)
    ]
    if rows.empty:
        raise ValueError(f"Neighborhood not found in dataset: {neighborhood}")
    return rows


def _mean_for_neighborhood(rows: pd.DataFrame, column: str) -> float:
    """Calculate a numeric neighborhood mean for an engineered feature."""
    if column not in rows.columns:
        raise ValueError(f"Dataset missing column needed for prediction: {column}")

    value = pd.to_numeric(rows[column], errors="coerce").mean()
    if pd.isna(value):
        raise ValueError(f"Cannot calculate neighborhood value for: {column}")
    return float(value)


def enrich_input_features(input_data: Mapping[str, Any]) -> dict[str, Any]:
    """Build all model features from user inputs and dataset-derived context."""
    enriched = dict(input_data)

    area = enriched.get(config.AREA_COLUMN)
    if area in (None, ""):
        raise ValueError(f"Missing input column for prediction: {config.AREA_COLUMN}")
    area_value = float(area)
    if area_value <= 0:
        raise ValueError("Constructed area must be greater than zero.")
    enriched[config.LOG_AREA_COLUMN] = float(np.log1p(area_value))

    neighborhood = enriched.get(config.NEIGHBORHOOD_COLUMN)
    if not neighborhood:
        raise ValueError(f"Missing input column for prediction: {config.NEIGHBORHOOD_COLUMN}")

    neighborhood_rows = _filter_neighborhood(_load_dataset(), str(neighborhood))
    for column in config.NEIGHBORHOOD_DERIVED_FEATURE_COLUMNS:
        enriched[column] = _mean_for_neighborhood(neighborhood_rows, column)

    for column in config.BOOLEAN_FEATURE_COLUMNS:
        enriched[column] = int(enriched.get(column, 0) or 0)

    return enriched


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
    enriched_input = enrich_input_features(input_data)
    validate_input_columns(enriched_input, feature_columns)
    ordered_values = {
        column: enriched_input[column] for column in feature_columns
    }
    return pd.DataFrame([ordered_values], columns=list(feature_columns))


def predict_price(input_data: Mapping[str, Any]) -> float:
    """Predict a property price in euros from business-level property inputs."""
    model = load_model()
    feature_columns = load_feature_columns()
    input_dataframe = build_input_dataframe(input_data, feature_columns)
    log_price_prediction = model.predict(input_dataframe)

    if len(log_price_prediction) == 0:
        raise ValueError("Model returned an empty prediction.")

    return float(np.expm1(log_price_prediction[0]))
