"""Real estate analysis utilities for the Smart Advisor MVP.

This module contains reusable business logic for neighborhood analysis and
property price comparison. It does not include Streamlit code, charts, or
conversational logic.
"""

import difflib
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

import config


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "data" / "dataset.csv"
DATASET_PATH = getattr(config, "DATASET_PATH", DEFAULT_DATASET_PATH)

NEIGHBORHOOD_COLUMN = getattr(config, "NEIGHBORHOOD_COLUMN", "LOCATIONNAME")
PRICE_COLUMN = getattr(config, "PRICE_COLUMN", "PRICE")
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


def compare_neighborhoods(
    first_neighborhood: str,
    second_neighborhood: str,
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compare two neighborhoods using the same market statistics."""
    dataset = dataset if dataset is not None else load_dataset()
    first_stats = get_neighborhood_stats(first_neighborhood, dataset)
    second_stats = get_neighborhood_stats(second_neighborhood, dataset)
    price_difference = calculate_difference(
        property_price=first_stats["average_price"],
        reference_price=second_stats["average_price"],
    )
    unit_price_difference = calculate_difference(
        property_price=first_stats["average_unit_price"],
        reference_price=second_stats["average_unit_price"],
    )

    return {
        "first_neighborhood": first_stats,
        "second_neighborhood": second_stats,
        "average_price_difference": price_difference,
        "average_unit_price_difference": unit_price_difference,
    }


def recommend_neighborhoods_by_budget(
    budget: float,
    dataset: pd.DataFrame | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """Recommend neighborhoods whose average price fits a given budget."""
    dataset = dataset if dataset is not None else load_dataset()
    _validate_columns(
        dataset,
        [NEIGHBORHOOD_COLUMN, PRICE_COLUMN, UNIT_PRICE_COLUMN, AREA_COLUMN],
    )

    if budget <= 0:
        raise ValueError("Budget must be greater than zero.")

    grouped = (
        dataset.groupby(NEIGHBORHOOD_COLUMN)
        .agg(
            property_count=(PRICE_COLUMN, "size"),
            average_price=(PRICE_COLUMN, "mean"),
            median_price=(PRICE_COLUMN, "median"),
            average_unit_price=(UNIT_PRICE_COLUMN, "mean"),
            average_area=(AREA_COLUMN, "mean"),
        )
        .reset_index()
    )
    candidates = grouped[grouped["average_price"] <= float(budget)].copy()
    candidates["budget_gap"] = float(budget) - candidates["average_price"]
    candidates = candidates.sort_values(
        by=["budget_gap", "property_count"],
        ascending=[True, False],
    ).head(max_results)

    recommendations = [
        {
            "neighborhood": str(row[NEIGHBORHOOD_COLUMN]),
            "property_count": int(row["property_count"]),
            "average_price": float(row["average_price"]),
            "median_price": float(row["median_price"]),
            "average_unit_price": float(row["average_unit_price"]),
            "average_area": float(row["average_area"]),
            "budget_gap": float(row["budget_gap"]),
        }
        for _, row in candidates.iterrows()
    ]

    return {
        "budget": float(budget),
        "recommendations": recommendations,
        "result_count": len(recommendations),
    }


def get_neighborhood_intelligence(
    neighborhood: str,
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return dataset-derived market intelligence for a neighborhood."""
    dataset = dataset if dataset is not None else load_dataset()
    required_columns = [
        NEIGHBORHOOD_COLUMN,
        PRICE_COLUMN,
        UNIT_PRICE_COLUMN,
        getattr(config, "DISTANCE_TO_CITY_CENTER_COLUMN", "DISTANCE_TO_CITY_CENTER"),
        getattr(config, "DISTANCE_TO_CASTELLANA_COLUMN", "DISTANCE_TO_CASTELLANA"),
        getattr(config, "DISTANCE_TO_METRO_COLUMN", "DISTANCE_TO_METRO"),
        getattr(config, "MEAN_UNITPRICE_BY_LOCATION_COLUMN", "MEAN_UNITPRICE_BY_LOCATION"),
    ]
    _validate_columns(dataset, required_columns)
    neighborhood_data = filter_neighborhood(dataset, neighborhood)

    madrid_average_unit_price = float(dataset[UNIT_PRICE_COLUMN].mean())
    neighborhood_average_unit_price = float(neighborhood_data[UNIT_PRICE_COLUMN].mean())
    relative_price_percent = calculate_difference(
        property_price=neighborhood_average_unit_price,
        reference_price=madrid_average_unit_price,
    )["percentage_difference"]

    if relative_price_percent >= 15:
        relative_price_level = "Alto"
    elif relative_price_percent <= -15:
        relative_price_level = "Bajo"
    else:
        relative_price_level = "Medio"

    return {
        "neighborhood": neighborhood,
        "average_price": float(neighborhood_data[PRICE_COLUMN].mean()),
        "average_unit_price": neighborhood_average_unit_price,
        "madrid_average_unit_price": madrid_average_unit_price,
        "relative_price_percent": float(relative_price_percent),
        "relative_price_level": relative_price_level,
        "distance_to_city_center": float(
            neighborhood_data[config.DISTANCE_TO_CITY_CENTER_COLUMN].mean()
        ),
        "distance_to_castellana": float(
            neighborhood_data[config.DISTANCE_TO_CASTELLANA_COLUMN].mean()
        ),
        "distance_to_metro": float(
            neighborhood_data[config.DISTANCE_TO_METRO_COLUMN].mean()
        ),
        "mean_unitprice_by_location": float(
            neighborhood_data[config.MEAN_UNITPRICE_BY_LOCATION_COLUMN].mean()
        ),
        "property_count": int(len(neighborhood_data)),
    }


def get_demo_property(dataset: pd.DataFrame | None = None) -> dict[str, Any]:
    """Return one real dataset row compatible with the valuation form."""
    dataset = dataset if dataset is not None else load_dataset()
    required_columns = list(config.USER_INPUT_FEATURE_COLUMNS) + [PRICE_COLUMN]
    _validate_columns(dataset, required_columns)

    clean_dataset = dataset.dropna(subset=required_columns).copy()
    if clean_dataset.empty:
        raise ValueError("No demo property available with complete input data.")

    row = clean_dataset.sort_values(PRICE_COLUMN).iloc[len(clean_dataset) // 2]
    input_data = {
        config.NEIGHBORHOOD_COLUMN: str(row[config.NEIGHBORHOOD_COLUMN]),
        config.AREA_COLUMN: float(row[config.AREA_COLUMN]),
        config.ROOMS_COLUMN: int(row[config.ROOMS_COLUMN]),
        config.BATHROOMS_COLUMN: int(row[config.BATHROOMS_COLUMN]),
        config.HAS_TERRACE_COLUMN: int(row[config.HAS_TERRACE_COLUMN]),
        config.HAS_LIFT_COLUMN: int(row[config.HAS_LIFT_COLUMN]),
        config.HAS_AIR_CONDITIONING_COLUMN: int(row[config.HAS_AIR_CONDITIONING_COLUMN]),
        config.HAS_PARKING_COLUMN: int(row[config.HAS_PARKING_COLUMN]),
        config.HAS_BOXROOM_COLUMN: int(row[config.HAS_BOXROOM_COLUMN]),
        config.HAS_SWIMMING_POOL_COLUMN: int(row[config.HAS_SWIMMING_POOL_COLUMN]),
    }

    return {
        "input": input_data,
        "actual_price": float(row[PRICE_COLUMN]),
        "actual_unit_price": float(row[UNIT_PRICE_COLUMN]) if UNIT_PRICE_COLUMN in row else None,
    }



def _normalize_for_matching(value: str) -> str:
    """Normalize text for accent-insensitive neighborhood suggestions."""
    normalized = unicodedata.normalize("NFKD", str(value).casefold().strip())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def suggest_neighborhoods(
    query: str,
    dataset: pd.DataFrame | None = None,
    max_results: int = 6,
) -> list[str]:
    """Suggest available LOCATIONNAME values for a non-exact user query."""
    dataset = dataset if dataset is not None else load_dataset()
    _validate_columns(dataset, [NEIGHBORHOOD_COLUMN])

    values = sorted(dataset[NEIGHBORHOOD_COLUMN].dropna().astype(str).unique())
    normalized_values = {value: _normalize_for_matching(value) for value in values}
    normalized_query = _normalize_for_matching(query)

    district_hints = {
        "hortaleza": ["Canillas", "Pinar del Rey", "Conde Orgaz-Piovera", "Palomas", "Sanchinarro", "Virgen del Cortijo - Manoteras"],
        "chamartin": ["Castilla", "Nueva España", "El Viso", "Prosperidad", "Ciudad Jardín", "Bernabéu-Hispanoamérica"],
        "chamartín": ["Castilla", "Nueva España", "El Viso", "Prosperidad", "Ciudad Jardín", "Bernabéu-Hispanoamérica"],
        "salamanca": ["Castellana", "Goya", "Lista", "Recoletos", "Fuente del Berro", "Guindalera"],
        "centro": ["Palacio", "Sol", "Chueca-Justicia", "Huertas-Cortes", "Lavapiés-Embajadores", "Malasaña-Universidad"],
    }
    hinted = [value for value in district_hints.get(normalized_query, []) if value in values]
    if hinted:
        return hinted[:max_results]

    contains = [
        value
        for value, normalized_value in normalized_values.items()
        if normalized_query and normalized_query in normalized_value
    ]
    if contains:
        return contains[:max_results]

    close_normalized = difflib.get_close_matches(
        normalized_query,
        list(normalized_values.values()),
        n=max_results,
        cutoff=0.55,
    )
    suggestions: list[str] = []
    for close_match in close_normalized:
        for value, normalized_value in normalized_values.items():
            if normalized_value == close_match and value not in suggestions:
                suggestions.append(value)
                break

    return suggestions[:max_results]
