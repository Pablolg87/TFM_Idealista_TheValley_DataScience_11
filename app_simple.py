"""Simple Streamlit MVP for property valuation and neighborhood comparison.

This app is intentionally independent from the existing frontend and backend.
It uses only native Streamlit UI components and reads the definitive model
package directly from models/best_rf_model_package.pkl.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PACKAGE_PATH = BASE_DIR / "models" / "best_rf_model_package.pkl"

PRICE_COLUMN = "PRICE"
UNIT_PRICE_COLUMN = "UNITPRICE"
NEIGHBORHOOD_COLUMN = "LOCATIONNAME"
AREA_COLUMN = "CONSTRUCTEDAREA"
LOG_AREA_COLUMN = "LOG_CONSTRUCTEDAREA"
ROOMS_COLUMN = "ROOMNUMBER"
BATHROOMS_COLUMN = "BATHNUMBER"
HAS_TERRACE_COLUMN = "HASTERRACE"
HAS_LIFT_COLUMN = "HASLIFT"
HAS_AIR_CONDITIONING_COLUMN = "HASAIRCONDITIONING"
HAS_PARKING_COLUMN = "HASPARKINGSPACE"
HAS_BOXROOM_COLUMN = "HASBOXROOM"
HAS_SWIMMING_POOL_COLUMN = "HASSWIMMINGPOOL"
DISTANCE_TO_CITY_CENTER_COLUMN = "DISTANCE_TO_CITY_CENTER"
DISTANCE_TO_CASTELLANA_COLUMN = "DISTANCE_TO_CASTELLANA"
DISTANCE_TO_METRO_COLUMN = "DISTANCE_TO_METRO"
MEAN_UNITPRICE_BY_LOCATION_COLUMN = "MEAN_UNITPRICE_BY_LOCATION"

USER_COLUMNS = [
    AREA_COLUMN,
    ROOMS_COLUMN,
    BATHROOMS_COLUMN,
    HAS_TERRACE_COLUMN,
    HAS_LIFT_COLUMN,
    HAS_AIR_CONDITIONING_COLUMN,
    HAS_PARKING_COLUMN,
    HAS_BOXROOM_COLUMN,
    HAS_SWIMMING_POOL_COLUMN,
]

DERIVED_COLUMNS = [
    DISTANCE_TO_CITY_CENTER_COLUMN,
    DISTANCE_TO_CASTELLANA_COLUMN,
    DISTANCE_TO_METRO_COLUMN,
    MEAN_UNITPRICE_BY_LOCATION_COLUMN,
]

COMPARISON_COLUMNS = [
    PRICE_COLUMN,
    UNIT_PRICE_COLUMN,
    AREA_COLUMN,
    ROOMS_COLUMN,
    BATHROOMS_COLUMN,
    HAS_TERRACE_COLUMN,
    HAS_LIFT_COLUMN,
    HAS_AIR_CONDITIONING_COLUMN,
    HAS_PARKING_COLUMN,
    HAS_BOXROOM_COLUMN,
    HAS_SWIMMING_POOL_COLUMN,
    DISTANCE_TO_CITY_CENTER_COLUMN,
    DISTANCE_TO_CASTELLANA_COLUMN,
    DISTANCE_TO_METRO_COLUMN,
]


def load_dataset() -> pd.DataFrame:
    """Load and validate the project dataset."""
    dataset = pd.read_csv(DATASET_PATH)
    required_columns = [NEIGHBORHOOD_COLUMN] + COMPARISON_COLUMNS + [
        MEAN_UNITPRICE_BY_LOCATION_COLUMN,
        LOG_AREA_COLUMN,
    ]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")
    return dataset.dropna(subset=[NEIGHBORHOOD_COLUMN]).copy()


def load_model_package() -> dict[str, Any]:
    """Load and validate the definitive model package."""
    package = joblib.load(MODEL_PACKAGE_PATH)
    if not isinstance(package, dict):
        raise TypeError("Model package must be a dictionary.")
    for key in ("model", "features", "metrics"):
        if key not in package:
            raise ValueError(f"Model package missing key: {key}")
    return package


def get_neighborhoods(dataset: pd.DataFrame) -> list[str]:
    """Return sorted neighborhood names from LOCATIONNAME."""
    return sorted(dataset[NEIGHBORHOOD_COLUMN].dropna().astype(str).unique())


def get_neighborhood_rows(dataset: pd.DataFrame, neighborhood: str) -> pd.DataFrame:
    """Return rows for the selected neighborhood."""
    rows = dataset[dataset[NEIGHBORHOOD_COLUMN].astype(str).eq(str(neighborhood))].copy()
    if rows.empty:
        raise ValueError(f"Neighborhood not found: {neighborhood}")
    return rows


def neighborhood_mean(rows: pd.DataFrame, column: str) -> float:
    """Calculate a numeric mean for a neighborhood column."""
    value = pd.to_numeric(rows[column], errors="coerce").mean()
    if pd.isna(value):
        raise ValueError(f"Cannot calculate neighborhood mean for {column}")
    return float(value)


def build_model_input(
    features: list[str],
    dataset: pd.DataFrame,
    neighborhood: str,
    property_input: dict[str, Any],
) -> pd.DataFrame:
    """Build the one-row DataFrame expected by the Random Forest model."""
    rows = get_neighborhood_rows(dataset, neighborhood)
    area = float(property_input[AREA_COLUMN])
    if area <= 0:
        raise ValueError("Constructed area must be greater than zero.")

    model_values = {
        LOG_AREA_COLUMN: float(np.log1p(area)),
        ROOMS_COLUMN: int(property_input[ROOMS_COLUMN]),
        BATHROOMS_COLUMN: int(property_input[BATHROOMS_COLUMN]),
        HAS_TERRACE_COLUMN: int(property_input[HAS_TERRACE_COLUMN]),
        HAS_LIFT_COLUMN: int(property_input[HAS_LIFT_COLUMN]),
        HAS_AIR_CONDITIONING_COLUMN: int(property_input[HAS_AIR_CONDITIONING_COLUMN]),
        HAS_PARKING_COLUMN: int(property_input[HAS_PARKING_COLUMN]),
        HAS_BOXROOM_COLUMN: int(property_input[HAS_BOXROOM_COLUMN]),
        HAS_SWIMMING_POOL_COLUMN: int(property_input[HAS_SWIMMING_POOL_COLUMN]),
    }

    for column in DERIVED_COLUMNS:
        model_values[column] = neighborhood_mean(rows, column)

    missing_features = [feature for feature in features if feature not in model_values]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    return pd.DataFrame([{feature: model_values[feature] for feature in features}])


def predict_price(
    package: dict[str, Any],
    dataset: pd.DataFrame,
    neighborhood: str,
    property_input: dict[str, Any],
) -> float:
    """Predict LOG_PRICE and convert it to euros."""
    features = [str(feature) for feature in package["features"]]
    model_input = build_model_input(features, dataset, neighborhood, property_input)
    log_price = package["model"].predict(model_input)[0]
    return float(np.expm1(log_price))


def classify_price(difference_percent: float) -> str:
    """Classify the estimated price against neighborhood unit price."""
    if difference_percent < -10:
        return "Infravalorado"
    if difference_percent > 10:
        return "Sobrevalorado"
    return "En mercado"


def format_euros(value: float) -> str:
    """Format a numeric value as euros."""
    return f"{value:,.0f} EUR".replace(",", ".")


def calculate_valuation_context(
    package: dict[str, Any],
    dataset: pd.DataFrame,
    neighborhood: str,
    property_input: dict[str, Any],
) -> dict[str, Any]:
    """Calculate prediction, unit-price comparison and neighborhood context."""
    rows = get_neighborhood_rows(dataset, neighborhood)
    estimated_price = predict_price(package, dataset, neighborhood, property_input)
    area = float(property_input[AREA_COLUMN])
    estimated_unit_price = estimated_price / area
    neighborhood_unit_price = neighborhood_mean(rows, UNIT_PRICE_COLUMN)
    difference_percent = (estimated_unit_price - neighborhood_unit_price) / neighborhood_unit_price * 100

    return {
        "neighborhood": neighborhood,
        "input": dict(property_input),
        "estimated_price": estimated_price,
        "estimated_unit_price": estimated_unit_price,
        "neighborhood_unit_price": neighborhood_unit_price,
        "difference_percent": difference_percent,
        "classification": classify_price(difference_percent),
        "distance_to_city_center": neighborhood_mean(rows, DISTANCE_TO_CITY_CENTER_COLUMN),
        "distance_to_castellana": neighborhood_mean(rows, DISTANCE_TO_CASTELLANA_COLUMN),
        "distance_to_metro": neighborhood_mean(rows, DISTANCE_TO_METRO_COLUMN),
        "top_feature": get_top_feature(package),
    }


def get_feature_importance(package: dict[str, Any]) -> pd.DataFrame:
    """Return global Random Forest feature importances."""
    model = package["model"]
    features = [str(feature) for feature in package["features"]]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["Variable", "Importancia"])

    importance_df = pd.DataFrame(
        {"Variable": features, "Importancia": importances}
    ).sort_values("Importancia", ascending=False)
    return importance_df.reset_index(drop=True)


def get_top_feature(package: dict[str, Any]) -> str:
    """Return the most important global feature name."""
    importance_df = get_feature_importance(package)
    if importance_df.empty:
        return "No disponible"
    return str(importance_df.iloc[0]["Variable"])


def answer_chat_question(
    question: str,
    package: dict[str, Any],
    dataset: pd.DataFrame,
    context: dict[str, Any],
) -> str:
    """Answer simple rule-based questions from the latest valuation."""
    normalized_question = question.casefold()

    if "piscina" in normalized_question:
        simulated_input = dict(context["input"])
        if int(simulated_input[HAS_SWIMMING_POOL_COLUMN]) == 1:
            return "La valoracion ya incluye piscina."
        simulated_input[HAS_SWIMMING_POOL_COLUMN] = 1
        simulated_price = predict_price(package, dataset, context["neighborhood"], simulated_input)
        difference = simulated_price - float(context["estimated_price"])
        return (
            "Si tuviera piscina, el modelo estimaria "
            f"{format_euros(simulated_price)}, una diferencia de {format_euros(difference)}."
        )

    if "variable" in normalized_question or "pesa" in normalized_question:
        return f"La variable con mayor importancia global en el modelo es {context['top_feature']}."

    if "comunic" in normalized_question or "metro" in normalized_question:
        return (
            "La conectividad se aproxima con las distancias medias del barrio: "
            f"centro {context['distance_to_city_center']:.2f}, "
            f"Castellana {context['distance_to_castellana']:.2f}, "
            f"metro {context['distance_to_metro']:.2f}."
        )

    if "caro" in normalized_question or "precio" in normalized_question:
        return (
            f"La vivienda queda clasificada como {context['classification']}. "
            f"Su precio estimado por m2 es {format_euros(context['estimated_unit_price'])}, "
            f"frente a {format_euros(context['neighborhood_unit_price'])} de media en el barrio."
        )

    return (
        "La valoracion se basa en superficie, habitaciones, banos, amenities "
        "y variables de contexto del barrio calculadas desde el dataset."
    )


def summarize_neighborhood(dataset: pd.DataFrame, neighborhood: str) -> dict[str, float | int | str]:
    """Calculate comparison metrics for one neighborhood."""
    rows = get_neighborhood_rows(dataset, neighborhood)
    return {
        "Barrio": neighborhood,
        "Precio medio": neighborhood_mean(rows, PRICE_COLUMN),
        "EUR/m2 medio": neighborhood_mean(rows, UNIT_PRICE_COLUMN),
        "Superficie media": neighborhood_mean(rows, AREA_COLUMN),
        "Habitaciones medias": neighborhood_mean(rows, ROOMS_COLUMN),
        "Banos medios": neighborhood_mean(rows, BATHROOMS_COLUMN),
        "% terraza": neighborhood_mean(rows, HAS_TERRACE_COLUMN) * 100,
        "% ascensor": neighborhood_mean(rows, HAS_LIFT_COLUMN) * 100,
        "% aire acondicionado": neighborhood_mean(rows, HAS_AIR_CONDITIONING_COLUMN) * 100,
        "% parking": neighborhood_mean(rows, HAS_PARKING_COLUMN) * 100,
        "% trastero": neighborhood_mean(rows, HAS_BOXROOM_COLUMN) * 100,
        "% piscina": neighborhood_mean(rows, HAS_SWIMMING_POOL_COLUMN) * 100,
        "Distancia centro": neighborhood_mean(rows, DISTANCE_TO_CITY_CENTER_COLUMN),
        "Distancia Castellana": neighborhood_mean(rows, DISTANCE_TO_CASTELLANA_COLUMN),
        "Distancia metro": neighborhood_mean(rows, DISTANCE_TO_METRO_COLUMN),
        "Viviendas": int(len(rows)),
    }


def build_comparison_conclusion(comparison_df: pd.DataFrame) -> str:
    """Build a short automatic conclusion for two neighborhoods."""
    first = comparison_df.iloc[0]
    second = comparison_df.iloc[1]
    pricier = first["Barrio"] if first["EUR/m2 medio"] > second["EUR/m2 medio"] else second["Barrio"]
    better_connected = first["Barrio"] if first["Distancia metro"] < second["Distancia metro"] else second["Barrio"]

    amenity_columns = [
        "% terraza",
        "% ascensor",
        "% aire acondicionado",
        "% parking",
        "% trastero",
        "% piscina",
    ]
    first_amenities = float(first[amenity_columns].mean())
    second_amenities = float(second[amenity_columns].mean())
    more_amenities = first["Barrio"] if first_amenities > second_amenities else second["Barrio"]

    return (
        f"{pricier} es el barrio mas caro por EUR/m2. "
        f"{better_connected} tiene mejor conectividad por distancia media al metro. "
        f"{more_amenities} presenta mayor disponibilidad media de amenities."
    )


def render_valuation_module(dataset: pd.DataFrame, package: dict[str, Any]) -> None:
    """Render the property valuation module."""
    st.subheader("Modulo 1: Valoracion de vivienda")
    neighborhoods = get_neighborhoods(dataset)

    col1, col2 = st.columns(2)
    with col1:
        neighborhood = st.selectbox("Barrio", neighborhoods)
        constructed_area = st.number_input("CONSTRUCTEDAREA", min_value=1.0, value=80.0, step=1.0)
        room_number = st.number_input("ROOMNUMBER", min_value=0, value=2, step=1)
        bath_number = st.number_input("BATHNUMBER", min_value=0, value=1, step=1)
    with col2:
        has_terrace = st.checkbox("HASTERRACE")
        has_lift = st.checkbox("HASLIFT", value=True)
        has_air_conditioning = st.checkbox("HASAIRCONDITIONING")
        has_parking = st.checkbox("HASPARKINGSPACE")
        has_boxroom = st.checkbox("HASBOXROOM")
        has_swimming_pool = st.checkbox("HASSWIMMINGPOOL")

    property_input = {
        AREA_COLUMN: constructed_area,
        ROOMS_COLUMN: room_number,
        BATHROOMS_COLUMN: bath_number,
        HAS_TERRACE_COLUMN: int(has_terrace),
        HAS_LIFT_COLUMN: int(has_lift),
        HAS_AIR_CONDITIONING_COLUMN: int(has_air_conditioning),
        HAS_PARKING_COLUMN: int(has_parking),
        HAS_BOXROOM_COLUMN: int(has_boxroom),
        HAS_SWIMMING_POOL_COLUMN: int(has_swimming_pool),
    }

    if st.button("Calcular valoracion"):
        context = calculate_valuation_context(package, dataset, neighborhood, property_input)
        st.session_state["simple_last_valuation"] = context
        st.session_state["simple_chat_messages"] = []

    context = st.session_state.get("simple_last_valuation")
    if context:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Precio estimado", format_euros(context["estimated_price"]))
        metric_col2.metric("EUR/m2 estimado", format_euros(context["estimated_unit_price"]))
        metric_col3.metric("EUR/m2 medio barrio", format_euros(context["neighborhood_unit_price"]))
        metric_col4.metric("Diferencia vs barrio", f"{context['difference_percent']:.2f}%")

        st.subheader(f"Clasificacion: {context['classification']}")

        importance_df = get_feature_importance(package)
        st.subheader("Importancia global de variables del modelo")
        if not importance_df.empty:
            chart_df = importance_df.set_index("Variable")
            st.bar_chart(chart_df)
            st.dataframe(importance_df)

        st.subheader("Chat sobre la ultima valoracion")
        messages = st.session_state.setdefault("simple_chat_messages", [])
        for message in messages:
            with st.chat_message(message["role"]):
                st.dataframe(pd.DataFrame({"Respuesta": [message["content"]]}))

        question = st.chat_input("Pregunta sobre la ultima valoracion")
        if question:
            answer = answer_chat_question(question, package, dataset, context)
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
            with st.chat_message("user"):
                st.dataframe(pd.DataFrame({"Pregunta": [question]}))
            with st.chat_message("assistant"):
                st.dataframe(pd.DataFrame({"Respuesta": [answer]}))


def render_comparison_module(dataset: pd.DataFrame) -> None:
    """Render the neighborhood comparison module."""
    st.subheader("Modulo 2: Comparador de barrios")
    neighborhoods = get_neighborhoods(dataset)

    col1, col2 = st.columns(2)
    with col1:
        first_neighborhood = st.selectbox("Barrio A", neighborhoods, key="comparison_a")
    with col2:
        second_index = 1 if len(neighborhoods) > 1 else 0
        second_neighborhood = st.selectbox("Barrio B", neighborhoods, index=second_index, key="comparison_b")

    if st.button("Comparar barrios"):
        comparison_df = pd.DataFrame(
            [
                summarize_neighborhood(dataset, first_neighborhood),
                summarize_neighborhood(dataset, second_neighborhood),
            ]
        )
        st.session_state["simple_comparison"] = comparison_df
        st.session_state["simple_comparison_conclusion"] = build_comparison_conclusion(comparison_df)

    comparison_df = st.session_state.get("simple_comparison")
    if comparison_df is not None:
        st.dataframe(comparison_df)
        st.subheader("Conclusion automatica")
        st.dataframe(pd.DataFrame({"Conclusion": [st.session_state["simple_comparison_conclusion"]]}))


def main() -> None:
    """Run the simple MVP app."""
    st.title("Smart Advisor MVP Simple")
    dataset = load_dataset()
    package = load_model_package()

    valuation_tab, comparison_tab = st.tabs(["Valoracion de vivienda", "Comparador de barrios"])
    with valuation_tab:
        render_valuation_module(dataset, package)
    with comparison_tab:
        render_comparison_module(dataset)


if __name__ == "__main__":
    main()
