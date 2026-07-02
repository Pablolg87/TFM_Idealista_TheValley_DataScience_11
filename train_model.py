"""Train the provisional MVP model for the Smart Advisor demo.

The model is intentionally simple and uses only the variables that the
Streamlit interface can provide. The target is LOG_PRICE, and predict.py is
responsible for converting predictions back to real prices in euros.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import DATASET_PATH, FEATURE_COLUMNS_PATH, MODEL_PATH, MODELS_DIR
from config import MVP_FEATURE_COLUMNS, N_ESTIMATORS, RANDOM_STATE, TARGET_COLUMN


def load_dataset() -> pd.DataFrame:
    """Load the configured training dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    dataset = pd.read_csv(DATASET_PATH)
    if dataset.empty:
        raise ValueError("Training dataset cannot be empty.")

    return dataset


def validate_training_columns(dataset: pd.DataFrame) -> None:
    """Validate that the dataset supports the MVP training contract."""
    required_columns = MVP_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in dataset.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required training columns: {missing}")


def split_features_and_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the MVP predictors and LOG_PRICE target."""
    validate_training_columns(dataset)
    training_columns = MVP_FEATURE_COLUMNS + [TARGET_COLUMN]
    training_data = dataset[training_columns].dropna(subset=[TARGET_COLUMN])

    if training_data.empty:
        raise ValueError("No training rows remain after removing missing targets.")

    features = training_data[MVP_FEATURE_COLUMNS]
    target = training_data[TARGET_COLUMN]

    return features, target


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


def train_model() -> tuple[Pipeline, list[str]]:
    """Train the provisional model using the MVP feature contract."""
    dataset = load_dataset()
    features, target = split_features_and_target(dataset)
    model = build_model(features)
    model.fit(features, target)

    return model, list(MVP_FEATURE_COLUMNS)


def save_artifacts(model: Pipeline, feature_columns: list[str]) -> None:
    """Persist the trained model and feature column list."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)


def main() -> None:
    """Train the provisional MVP model and export application artifacts."""
    model, feature_columns = train_model()
    save_artifacts(model, feature_columns)

    print(f"Training target: {TARGET_COLUMN}")
    print("Feature columns:")
    for column in feature_columns:
        print(f"- {column}")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Feature columns saved at: {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
