"""Real estate analysis utilities for the Smart Advisor MVP.

This module contains reusable business logic for neighborhood analysis and
property price comparison. It does not include Streamlit code, charts, or
conversational logic.
"""

from pathlib import Path
from typing import Any

import pandas as pd

import config


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "data" / "dataset.csv"
DATASET_PATH = getattr(config, "DATASET_PATH", DEFAULT_DATASET_PATH)

NEIGHBORHOOD_COLUMN = getattr(config, "NEIGHBORHOOD_COLUMN", "LOCATIONNAME")
PRICE_COLUMN = getattr(config, "TARGET_COLUMN", "PRICE")
UNIT_PRICE_COLUMN = getattr(config, "UNIT_PRICE_COLUMN", "UNITPRICE")
AREA_COLUMN = getattr(config, "AREA_COLUMN", "CONSTRUCTEDAREA")

UNDERVALUED_THRESHOLD_PERCENT = -10.0
OVERVALUED_THRESHOLD_PERCENT = 10.0


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the configured real estate dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    if dataset.empty:
        raise ValueError("Dataset cannot be empty.")

    return dataset


def _validate_columns(dataset: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that the dataset contains all required columns."""
    missing_columns = [
        column for column in required_columns if column not in dataset.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required dataset columns: {missing}")


def _normalize_text(value: str) -> str:
    """Normalize text values for case-insensitive comparisons."""
    return value.strip().casefold()


def filter_neighborhood(
    dataset: pd.DataFrame,
    neighborhood: str,
) -> pd.DataFrame:
    """Return dataset rows for a given neighborhood."""
    _validate_columns(dataset, [NEIGHBORHOOD_COLUMN])

    if not neighborhood or not neighborhood.strip():
        raise ValueError("Neighborhood cannot be empty.")

    normalized_neighborhood = _normalize_text(neighborhood)
    mask = (
        dataset[NEIGHBORHOOD_COLUMN]
        .astype(str)
        .map(_normalize_text)
        .eq(normalized_neighborhood)
    )
    neighborhood_data = dataset.loc[mask].copy()

    if neighborhood_data.empty:
        raise ValueError(f"Neighborhood not found: {neighborhood}")

    return neighborhood_data


def calculate_difference(
    property_price: float,
    reference_price: float,
) -> dict[str, float]:
    """Calculate absolute and percentage difference versus a reference price."""
    if reference_price == 0:
        raise ValueError("Reference price cannot be zero.")

    absolute_difference = float(property_price - reference_price)
    percentage_difference = float(
        absolute_difference / reference_price * 100
    )

    return {
        "absolute_difference": absolute_difference,
        "percentage_difference": percentage_difference,
    }


def classify_property(percentage_difference: float) -> str:
    """Classify a property according to its deviation from neighborhood price."""
    if percentage_difference <= UNDERVALUED_THRESHOLD_PERCENT:
        return "Infravalorada"

    if percentage_difference >= OVERVALUED_THRESHOLD_PERCENT:
        return "Sobrevalorada"

    return "En precio"


def get_neighborhood_stats(
    neighborhood: str,
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Calculate key real estate statistics for a neighborhood."""
    dataset = dataset if dataset is not None else load_dataset()
    _validate_columns(
        dataset,
        [NEIGHBORHOOD_COLUMN, PRICE_COLUMN, UNIT_PRICE_COLUMN, AREA_COLUMN],
    )
    neighborhood_data = filter_neighborhood(dataset, neighborhood)

    return {
        "neighborhood": neighborhood,
        "property_count": int(len(neighborhood_data)),
        "average_price": float(neighborhood_data[PRICE_COLUMN].mean()),
        "median_price": float(neighborhood_data[PRICE_COLUMN].median()),
        "average_unit_price": float(
            neighborhood_data[UNIT_PRICE_COLUMN].mean()
        ),
        "average_area": float(neighborhood_data[AREA_COLUMN].mean()),
        "min_price": float(neighborhood_data[PRICE_COLUMN].min()),
        "max_price": float(neighborhood_data[PRICE_COLUMN].max()),
    }


def compare_property_with_neighborhood(
    neighborhood: str,
    property_price: float,
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compare a property price with the average price of its neighborhood."""
    stats = get_neighborhood_stats(neighborhood, dataset)
    difference = calculate_difference(
        property_price=float(property_price),
        reference_price=stats["average_price"],
    )
    classification = classify_property(difference["percentage_difference"])

    return {
        "neighborhood": stats["neighborhood"],
        "property_price": float(property_price),
        "neighborhood_average_price": stats["average_price"],
        "absolute_difference": difference["absolute_difference"],
        "percentage_difference": difference["percentage_difference"],
        "classification": classification,
        "neighborhood_stats": stats,
    }
