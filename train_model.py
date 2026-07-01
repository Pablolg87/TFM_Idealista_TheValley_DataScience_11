"""Train the MVP model for the Smart Advisor architecture.

This script is intentionally independent from the Streamlit interface, the
agent, and advisor logic. The definitive target for model training is
LOG_PRICE. The prediction layer is responsible for converting model output
back to real prices in euros.
"""

import re

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import DATASET_PATH, FEATURE_COLUMNS_PATH, MODEL_PATH, MODELS_DIR
from config import LOG_PRICE_TARGET_COLUMN, N_ESTIMATORS, PRICE_COLUMN
from config import RANDOM_STATE


EXPLICIT_EXCLUDED_COLUMNS = {
    PRICE_COLUMN: "real price column; excluded to avoid target leakage",
    "UNITPRICE": "price-derived column; excluded to avoid target leakage",
    "LOG_UNITPRICE": "price-derived column; excluded to avoid target leakage",
    "MEAN_UNITPRICE_BY_LOCATION": (
        "aggregated price-derived column; excluded to avoid target leakage"
    ),
}


def load_dataset() -> pd.DataFrame:
    """Load the configured training dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    dataset = pd.read_csv(DATASET_PATH)
    if dataset.empty:
        raise ValueError("Training dataset cannot be empty.")

    return dataset


def _normalized_column_name(column: str) -> str:
    """Normalize column names for robust exclusion rules."""
    return re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")


def validate_target_column(dataset: pd.DataFrame) -> None:
    """Validate that the definitive LOG_PRICE target exists."""
    if LOG_PRICE_TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column '{LOG_PRICE_TARGET_COLUMN}' was not found."
        )


def _is_identifier_column(column: str) -> bool:
    """Return True for identifiers and external reference columns."""
    normalized_column = _normalized_column_name(column)
    identifier_tokens = ("id", "assetid", "identifier", "url", "link", "href")

    return (
        normalized_column in identifier_tokens
        or normalized_column.endswith("_id")
        or normalized_column.endswith("id")
        or any(token in normalized_column for token in ("url", "link", "href"))
    )


def get_excluded_columns(dataset: pd.DataFrame) -> dict[str, str]:
    """Return columns excluded from training and the reason for each one."""
    excluded_columns = {
        LOG_PRICE_TARGET_COLUMN: "definitive training target",
    }

    for column, reason in EXPLICIT_EXCLUDED_COLUMNS.items():
        if column in dataset.columns:
            excluded_columns[column] = reason

    for column in dataset.columns:
        if column in excluded_columns:
            continue

        if _is_identifier_column(column):
            excluded_columns[column] = "identifier or external reference"

    return excluded_columns


def split_features_and_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    """Split predictors and LOG_PRICE target."""
    validate_target_column(dataset)
    excluded_columns = get_excluded_columns(dataset)
    feature_columns = [
        column for column in dataset.columns if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No valid predictor columns remain after filtering.")

    training_data = dataset[feature_columns + [LOG_PRICE_TARGET_COLUMN]].dropna(
        subset=[LOG_PRICE_TARGET_COLUMN]
    )
    if training_data.empty:
        raise ValueError("No training rows remain after removing missing targets.")

    features = training_data[feature_columns]
    target = training_data[LOG_PRICE_TARGET_COLUMN]

    return features, target, excluded_columns


def build_model(features: pd.DataFrame) -> Pipeline:
    """Build a robust and simple regression pipeline for the MVP."""
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    regressor = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def train_model() -> tuple[Pipeline, list[str], dict[str, str]]:
    """Train the model using LOG_PRICE and return artifacts metadata."""
    dataset = load_dataset()
    features, target, excluded_columns = split_features_and_target(dataset)
    model = build_model(features)
    model.fit(features, target)

    return model, features.columns.tolist(), excluded_columns


def save_artifacts(model: Pipeline, feature_columns: list[str]) -> None:
    """Persist the trained model and the feature column list."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)


def main() -> None:
    """Train the LOG_PRICE model and export application artifacts."""
    model, feature_columns, excluded_columns = train_model()
    save_artifacts(model, feature_columns)

    print(f"Training target: {LOG_PRICE_TARGET_COLUMN}")
    print("Excluded columns:")
    for column, reason in excluded_columns.items():
        print(f"- {column}: {reason}")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Feature columns saved at: {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
