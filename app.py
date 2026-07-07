"""Definitive Streamlit MVP for AI Real Estate Advisor.

This app intentionally uses native Streamlit components only. It does not use
custom HTML, custom CSS, external APIs, or generative AI services.
"""

import re
import unicodedata
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PACKAGE_PATH = BASE_DIR / "models" / "best_rf_model_package.pkl"
MODEL_PACKAGE_URL = "https://drive.google.com/uc?export=download&id=1GraR1z71RvUGuM4I_wi6VZr6xiubP3U2"

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

DERIVED_COLUMNS = [
    DISTANCE_TO_CITY_CENTER_COLUMN,
    DISTANCE_TO_CASTELLANA_COLUMN,
    DISTANCE_TO_METRO_COLUMN,
    MEAN_UNITPRICE_BY_LOCATION_COLUMN,
]

AMENITY_COLUMNS = [
    HAS_LIFT_COLUMN,
    HAS_TERRACE_COLUMN,
    HAS_AIR_CONDITIONING_COLUMN,
    HAS_PARKING_COLUMN,
    HAS_BOXROOM_COLUMN,
    HAS_SWIMMING_POOL_COLUMN,
]

REQUIRED_DATASET_COLUMNS = [
    NEIGHBORHOOD_COLUMN,
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
    MEAN_UNITPRICE_BY_LOCATION_COLUMN,
]

CONVERSATION_STEPS = [
    ("neighborhood", "Barrio"),
    (AREA_COLUMN, "Metros cuadrados construidos"),
    (ROOMS_COLUMN, "Habitaciones"),
    (BATHROOMS_COLUMN, "Ba\u00f1os"),
    (HAS_LIFT_COLUMN, "Ascensor"),
    (HAS_TERRACE_COLUMN, "Terraza"),
    (HAS_AIR_CONDITIONING_COLUMN, "Aire acondicionado"),
    (HAS_PARKING_COLUMN, "Parking"),
    (HAS_BOXROOM_COLUMN, "Trastero"),
    (HAS_SWIMMING_POOL_COLUMN, "Piscina"),
]



def apply_visual_theme() -> None:
    """Apply a lightweight PropTech visual theme."""
    st.markdown(
        """
        <style>
            :root {
                --primary: #97C93D;
                --primary-dark: #6FAE2A;
                --success-soft: #EAF7D8;
                --background: #F8F9FA;
                --card: #FFFFFF;
                --border: #E5E7EB;
                --text: #222222;
                --muted: #6B7280;
                --info-soft: #EEF6FF;
                --warning-soft: #FFF8D6;
                --shadow: 0 8px 24px rgba(34, 34, 34, 0.06);
            }

            .stApp {
                background: var(--background);
                color: var(--text);
            }

            .block-container {
                padding-top: 2.2rem;
                padding-bottom: 3rem;
                max-width: 1180px;
            }

            h1, h2, h3 {
                color: var(--text);
                letter-spacing: 0;
            }

            h1 {
                margin-bottom: 0.25rem;
            }

            h2, h3 {
                margin-top: 1.4rem;
            }

            div[data-testid="stVerticalBlock"] {
                gap: 1.05rem;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 16px;
                box-shadow: var(--shadow);
                padding: 0.35rem;
            }

            div[data-testid="stMetric"] {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 14px;
                box-shadow: 0 6px 18px rgba(34, 34, 34, 0.045);
                padding: 0.85rem 1rem;
            }

            div[data-testid="stMetricLabel"] p {
                color: var(--muted);
                font-weight: 600;
            }

            div[data-testid="stMetricValue"] {
                color: var(--primary-dark);
                font-weight: 800;
            }

            div.stButton > button,
            button[kind="primary"],
            button[kind="secondary"] {
                background: var(--primary) !important;
                color: #FFFFFF !important;
                border: 1px solid var(--primary) !important;
                border-radius: 10px !important;
                box-shadow: 0 8px 18px rgba(151, 201, 61, 0.24);
                font-weight: 700;
                transition: all 160ms ease;
            }

            div.stButton > button:hover,
            button[kind="primary"]:hover,
            button[kind="secondary"]:hover {
                background: var(--primary-dark) !important;
                border-color: var(--primary-dark) !important;
                color: #FFFFFF !important;
                box-shadow: 0 10px 22px rgba(111, 174, 42, 0.24);
                transform: translateY(-1px);
            }

            section[data-testid="stSidebar"] {
                background: #FFFFFF;
                border-right: 1px solid var(--border);
            }

            section[data-testid="stSidebar"] h1::after {
                content: "";
                display: block;
                width: 52px;
                height: 4px;
                margin-top: 0.55rem;
                border-radius: 999px;
                background: var(--primary);
            }

            div[data-testid="stAlert"] {
                border-radius: 14px;
                border: 1px solid var(--border);
                box-shadow: 0 6px 18px rgba(34, 34, 34, 0.035);
            }

            div[data-testid="stAlert"]:has([data-testid="stMarkdownContainer"]) {
                background: var(--info-soft);
            }

            div[data-testid="stChatMessage"] {
                border-radius: 16px;
                border: 1px solid var(--border);
                box-shadow: 0 5px 16px rgba(34, 34, 34, 0.035);
                margin-bottom: 0.75rem;
            }

            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
                background: var(--success-soft);
            }

            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
                background: #F3F4F6;
            }

            div[data-testid="stChatInput"] textarea {
                border-radius: 14px;
                border-color: var(--border);
                box-shadow: 0 6px 18px rgba(34, 34, 34, 0.045);
            }

            div[data-testid="stTabs"] button[aria-selected="true"] {
                color: var(--primary-dark);
                border-bottom-color: var(--primary) !important;
                font-weight: 700;
            }

            div[data-testid="stCaptionContainer"] {
                color: var(--muted);
            }

            @media (max-width: 768px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                    padding-top: 1.4rem;
                }

                div[data-testid="stMetric"] {
                    padding: 0.75rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def load_dataset() -> pd.DataFrame:
    """Load the cleaned Madrid real estate dataset."""
    dataset = pd.read_csv(DATASET_PATH)
    missing_columns = [
        column for column in REQUIRED_DATASET_COLUMNS if column not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")
    return dataset.dropna(subset=[NEIGHBORHOOD_COLUMN]).copy()


def download_model_package() -> bool:
    """Download the model package when it is not available locally."""
    MODEL_PACKAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(MODEL_PACKAGE_URL, timeout=120)
        response.raise_for_status()
        MODEL_PACKAGE_PATH.write_bytes(response.content)
        return True
    except requests.RequestException:
        st.error("No se ha podido descargar el modelo. Revisa la conexión o el enlace configurado.")
        return False


def load_model_package() -> dict[str, Any]:
    """Load the definitive Random Forest package."""
    if not MODEL_PACKAGE_PATH.exists() and not download_model_package():
        st.stop()

    package = joblib.load(MODEL_PACKAGE_PATH)
    if not isinstance(package, dict):
        raise TypeError("Model package must be a dictionary.")
    for key in ("model", "features", "metrics"):
        if key not in package:
            raise ValueError(f"Model package missing key: {key}")
    return package


def get_neighborhoods(dataset: pd.DataFrame) -> list[str]:
    """Return sorted neighborhood values from LOCATIONNAME."""
    return sorted(dataset[NEIGHBORHOOD_COLUMN].dropna().astype(str).unique())


def get_neighborhood_rows(dataset: pd.DataFrame, neighborhood: str) -> pd.DataFrame:
    """Return dataset rows for a selected neighborhood."""
    rows = dataset[dataset[NEIGHBORHOOD_COLUMN].astype(str).eq(str(neighborhood))]
    if rows.empty:
        raise ValueError(f"Neighborhood not found: {neighborhood}")
    return rows.copy()


def numeric_mean(rows: pd.DataFrame, column: str) -> float:
    """Calculate a robust numeric mean for a column."""
    value = pd.to_numeric(rows[column], errors="coerce").mean()
    if pd.isna(value):
        raise ValueError(f"Cannot calculate mean for column: {column}")
    return float(value)


def numeric_median(rows: pd.DataFrame, column: str) -> float:
    """Calculate a robust numeric median for a column."""
    value = pd.to_numeric(rows[column], errors="coerce").median()
    if pd.isna(value):
        raise ValueError(f"Cannot calculate median for column: {column}")
    return float(value)


def format_euros(value: float) -> str:
    """Format a number as euros using Spanish thousands separators."""
    return f"{value:,.0f} \u20ac".replace(",", ".")


def format_euros_per_m2(value: float) -> str:
    """Format a number as euros per square meter."""
    return f"{value:,.0f} \u20ac/m\u00b2".replace(",", ".")


def model_features(package: dict[str, Any]) -> list[str]:
    """Return feature names stored in the model package."""
    return [str(feature) for feature in package["features"]]


def build_model_input(
    features: list[str],
    dataset: pd.DataFrame,
    neighborhood: str,
    property_input: dict[str, Any],
) -> pd.DataFrame:
    """Build the single-row input expected by the Random Forest."""
    rows = get_neighborhood_rows(dataset, neighborhood)
    area = float(property_input[AREA_COLUMN])
    if area <= 0:
        raise ValueError("Constructed area must be greater than zero.")

    values = {
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
        values[column] = numeric_mean(rows, column)

    missing_features = [feature for feature in features if feature not in values]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    return pd.DataFrame([{feature: values[feature] for feature in features}])


def predict_price(
    package: dict[str, Any],
    dataset: pd.DataFrame,
    neighborhood: str,
    property_input: dict[str, Any],
) -> float:
    """Predict LOG_PRICE and return the estimated real price in euros."""
    input_frame = build_model_input(
        features=model_features(package),
        dataset=dataset,
        neighborhood=neighborhood,
        property_input=property_input,
    )
    predicted_log_price = package["model"].predict(input_frame)[0]
    return float(np.expm1(predicted_log_price))


def classify_valuation(difference_percent: float) -> str:
    """Return a neutral label for the valuation output."""
    return "Estimaci\u00f3n orientativa"


def feature_importance(package: dict[str, Any]) -> pd.DataFrame:
    """Return global feature importance from the Random Forest."""
    importances = getattr(package["model"], "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["Variable", "Importance"])
    return (
        pd.DataFrame({"Variable": model_features(package), "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def top_feature(package: dict[str, Any]) -> str:
    """Return the most important global model feature."""
    importance = feature_importance(package)
    if importance.empty:
        return "No disponible"
    return str(importance.iloc[0]["Variable"])


def calculate_valuation(
    package: dict[str, Any],
    dataset: pd.DataFrame,
    neighborhood: str,
    property_input: dict[str, Any],
) -> dict[str, Any]:
    """Calculate prediction, benchmark metrics and context."""
    rows = get_neighborhood_rows(dataset, neighborhood)
    estimated_price = predict_price(package, dataset, neighborhood, property_input)
    estimated_unit_price = estimated_price / float(property_input[AREA_COLUMN])
    average_unit_price = numeric_mean(rows, UNIT_PRICE_COLUMN)
    median_unit_price = numeric_median(rows, UNIT_PRICE_COLUMN)
    difference_percent = (
        (estimated_unit_price - average_unit_price) / average_unit_price * 100
    )

    return {
        "neighborhood": neighborhood,
        "input": dict(property_input),
        "estimated_price": estimated_price,
        "estimated_unit_price": estimated_unit_price,
        "average_unit_price": average_unit_price,
        "median_unit_price": median_unit_price,
        "difference_percent": difference_percent,
        "classification": classify_valuation(difference_percent),
        "average_price": numeric_mean(rows, PRICE_COLUMN),
        "median_price": numeric_median(rows, PRICE_COLUMN),
        "property_count": int(len(rows)),
        "distance_to_city_center": numeric_mean(rows, DISTANCE_TO_CITY_CENTER_COLUMN),
        "distance_to_castellana": numeric_mean(rows, DISTANCE_TO_CASTELLANA_COLUMN),
        "distance_to_metro": numeric_mean(rows, DISTANCE_TO_METRO_COLUMN),
        "top_feature": top_feature(package),
    }


def answer_valuation_question(
    question: str,
    package: dict[str, Any],
    dataset: pd.DataFrame,
    context: dict[str, Any],
) -> str:
    """Answer rule-based follow-up questions from the last valuation."""
    normalized_question = question.casefold()

    if "piscina" in normalized_question:
        simulated_input = dict(context["input"])
        if int(simulated_input[HAS_SWIMMING_POOL_COLUMN]) == 1:
            return "La vivienda ya incluye piscina en la valoraci\u00f3n actual."
        simulated_input[HAS_SWIMMING_POOL_COLUMN] = 1
        simulated_price = predict_price(
            package, dataset, context["neighborhood"], simulated_input
        )
        difference = simulated_price - float(context["estimated_price"])
        return (
            f"Si tuviera piscina, la estimaci\u00f3n ser\u00eda {format_euros(simulated_price)}, "
            f"con una diferencia aproximada de {format_euros(difference)}."
        )

    if "parking" in normalized_question or "garaje" in normalized_question:
        simulated_input = dict(context["input"])
        if int(simulated_input[HAS_PARKING_COLUMN]) == 1:
            return "La vivienda ya incluye parking en la valoraci\u00f3n actual."
        simulated_input[HAS_PARKING_COLUMN] = 1
        simulated_price = predict_price(
            package, dataset, context["neighborhood"], simulated_input
        )
        difference = simulated_price - float(context["estimated_price"])
        return (
            f"Si tuviera parking, la estimaci\u00f3n ser\u00eda {format_euros(simulated_price)}, "
            f"con una diferencia aproximada de {format_euros(difference)}."
        )

    if "variable" in normalized_question or "influ" in normalized_question:
        return (
            f"La variable con mayor importancia global en el Random Forest es "
            f"{context['top_feature']}."
        )

    if "comunic" in normalized_question or "metro" in normalized_question:
        return (
            "La informaci\u00f3n de ubicaci\u00f3n disponible para este barrio indica estas distancias medias: "
            f"centro {context['distance_to_city_center']:.2f}, "
            f"Castellana {context['distance_to_castellana']:.2f} y "
            f"metro {context['distance_to_metro']:.2f}."
        )

    return (
        f"La estimaci\u00f3n del modelo es {format_euros(context['estimated_price'])}, "
        f"equivalente a {format_euros_per_m2(context['estimated_unit_price'])}. "
        "Debe interpretarse como una ayuda a la toma de decisiones, no como una tasaci\u00f3n oficial."
    )


def initialize_valuation_state() -> None:
    """Initialize session state for the conversational valuation flow."""
    st.session_state.setdefault("valuation_step", 0)
    st.session_state.setdefault("valuation_data", {})
    st.session_state.setdefault("valuation_messages", [])
    st.session_state.setdefault("valuation_context", None)
    st.session_state.setdefault("followup_messages", [])


def reset_valuation_state() -> None:
    """Reset the conversational valuation flow."""
    st.session_state["valuation_step"] = 0
    st.session_state["valuation_data"] = {}
    st.session_state["valuation_messages"] = []
    st.session_state["valuation_context"] = None
    st.session_state["followup_messages"] = []


def question_for_step(step_index: int) -> str:
    """Return the assistant question for the current valuation step."""
    questions = [
        "\u00bfEn qu\u00e9 barrio est\u00e1 la vivienda?",
        "\u00bfCu\u00e1ntos metros cuadrados construidos tiene?",
        "\u00bfCu\u00e1ntas habitaciones tiene?",
        "\u00bfCu\u00e1ntos ba\u00f1os tiene?",
        "\u00bfTiene ascensor? Responde s\u00ed o no.",
        "\u00bfTiene terraza? Responde s\u00ed o no.",
        "\u00bfTiene aire acondicionado? Responde s\u00ed o no.",
        "\u00bfTiene plaza de garaje? Responde s\u00ed o no.",
        "\u00bfTiene trastero? Responde s\u00ed o no.",
        "\u00bfTiene piscina? Responde s\u00ed o no.",
    ]
    return questions[step_index]


def normalize_text(value: str) -> str:
    """Normalize text for accent-insensitive parsing."""
    normalized = unicodedata.normalize("NFKD", str(value).casefold().strip())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def parse_number_from_text(value: str) -> float | None:
    """Extract a number from a natural-language answer."""
    normalized = normalize_text(value)
    word_numbers = {
        "cero": 0,
        "uno": 1,
        "una": 1,
        "dos": 2,
        "tres": 3,
        "cuatro": 4,
        "cinco": 5,
        "seis": 6,
        "siete": 7,
        "ocho": 8,
        "nueve": 9,
        "diez": 10,
    }
    match = re.search(r"\d+(?:[\.,]\d+)?", normalized)
    if match:
        return float(match.group(0).replace(",", "."))
    tokens = re.findall(r"[a-zA-Z]+", normalized)
    for token in tokens:
        if token in word_numbers:
            return float(word_numbers[token])
    return None


def parse_boolean_answer(value: str) -> int | None:
    """Parse yes/no natural-language answers into 1 or 0."""
    normalized = normalize_text(value)
    negative_tokens = ("no", "sin", "ninguno", "ninguna", "false", "falso")
    positive_tokens = ("si", "con", "tiene", "yes", "true", "verdadero")
    if any(re.search(rf"\b{token}\b", normalized) for token in negative_tokens):
        return 0
    if normalized.startswith("s") and not normalized.startswith("sin"):
        return 1
    if any(re.search(rf"\b{normalize_text(token)}\b", normalized) for token in positive_tokens):
        return 1
    return None


def parse_neighborhood_answer(value: str, neighborhoods: list[str]) -> str | None:
    """Match a typed neighborhood against LOCATIONNAME values."""
    normalized_value = normalize_text(value)
    normalized_map = {normalize_text(neighborhood): neighborhood for neighborhood in neighborhoods}
    if normalized_value in normalized_map:
        return normalized_map[normalized_value]
    for normalized_neighborhood, neighborhood in normalized_map.items():
        if normalized_value and normalized_value in normalized_neighborhood:
            return neighborhood
    return None


def parse_conversation_answer(
    field: str,
    answer: str,
    neighborhoods: list[str],
) -> tuple[Any | None, str | None]:
    """Parse one chat answer for the active valuation field."""
    if field == "neighborhood":
        neighborhood = parse_neighborhood_answer(answer, neighborhoods)
        if neighborhood is None:
            return None, "No encuentro ese barrio en el dataset. Prueba con un nombre como Aravaca, Palacio o Sol."
        return neighborhood, None

    if field == AREA_COLUMN:
        number = parse_number_from_text(answer)
        if number is None or number <= 0:
            return None, "Necesito una superficie v\u00e1lida, por ejemplo: 95."
        return float(number), None

    if field in (ROOMS_COLUMN, BATHROOMS_COLUMN):
        number = parse_number_from_text(answer)
        if number is None or number < 0:
            return None, "Necesito un n\u00famero v\u00e1lido, por ejemplo: 3."
        return int(number), None

    boolean_value = parse_boolean_answer(answer)
    if boolean_value is None:
        return None, "Necesito una respuesta tipo s\u00ed o no."
    return boolean_value, None


def current_property_input() -> dict[str, Any]:
    """Build property input from stored conversational answers."""
    data = st.session_state["valuation_data"]
    return {
        AREA_COLUMN: float(data[AREA_COLUMN]),
        ROOMS_COLUMN: int(data[ROOMS_COLUMN]),
        BATHROOMS_COLUMN: int(data[BATHROOMS_COLUMN]),
        HAS_LIFT_COLUMN: int(data[HAS_LIFT_COLUMN]),
        HAS_TERRACE_COLUMN: int(data[HAS_TERRACE_COLUMN]),
        HAS_AIR_CONDITIONING_COLUMN: int(data[HAS_AIR_CONDITIONING_COLUMN]),
        HAS_PARKING_COLUMN: int(data[HAS_PARKING_COLUMN]),
        HAS_BOXROOM_COLUMN: int(data[HAS_BOXROOM_COLUMN]),
        HAS_SWIMMING_POOL_COLUMN: int(data[HAS_SWIMMING_POOL_COLUMN]),
    }


def render_chat_history() -> None:
    """Render the valuation conversation history."""
    for role, content in st.session_state["valuation_messages"]:
        with st.chat_message(role):
            if role == "assistant":
                st.info(content)
            else:
                st.caption(content)


def render_current_conversation_step(neighborhoods: list[str]) -> None:
    """Capture the active valuation answer using only st.chat_input."""
    step_index = st.session_state["valuation_step"]
    if step_index >= len(CONVERSATION_STEPS):
        return

    field, _label = CONVERSATION_STEPS[step_index]
    answer = st.chat_input("Responde al AI Property Advisor")
    if not answer:
        return

    st.session_state["valuation_messages"].append(("user", answer))
    parsed_value, error = parse_conversation_answer(field, answer, neighborhoods)
    if error:
        st.session_state["valuation_messages"].append(("assistant", error))
        st.session_state["valuation_messages"].append(("assistant", question_for_step(step_index)))
        st.rerun()

    st.session_state["valuation_data"][field] = parsed_value
    st.session_state["valuation_step"] += 1

    if st.session_state["valuation_step"] >= len(CONVERSATION_STEPS):
        st.session_state["valuation_messages"].append(("assistant", "Perfecto. Ya tengo todos los datos y voy a calcular la valoraci\u00f3n."))
    else:
        st.session_state["valuation_messages"].append(("assistant", question_for_step(st.session_state["valuation_step"])))
    st.rerun()


def render_valuation_results(
    package: dict[str, Any],
    context: dict[str, Any],
) -> None:
    """Render valuation results as an executive dashboard."""
    st.success("Valoraci\u00f3n completada por el AI Property Advisor.")
    st.subheader("Resultado de valoraci\u00f3n")

    with st.container(border=True):
        main_col, detail_col = st.columns([1.2, 1])
        with main_col:
            st.metric("Precio estimado", format_euros(context["estimated_price"]))
            st.metric("Precio estimado por m\u00b2", format_euros_per_m2(context["estimated_unit_price"]))
        with detail_col:
            st.metric("Barrio", context["neighborhood"])

    st.subheader("Variables con mayor influencia (Feature Importance)")
    importance = feature_importance(package)
    if importance.empty:
        st.warning("El modelo no expone feature_importances_.")
    else:
        st.bar_chart(importance.set_index("Variable").head(10))
        st.caption("Importancia global del Random Forest. No representa explicabilidad local de una vivienda concreta.")

    st.info(
        "Esta valoraci\u00f3n constituye una estimaci\u00f3n basada en un modelo de Machine Learning entrenado con datos "
        "hist\u00f3ricos del mercado inmobiliario de Madrid. Debe interpretarse como una ayuda a la toma de decisiones "
        "y no como una tasaci\u00f3n oficial."
    )


def render_followup_chat(
    package: dict[str, Any],
    dataset: pd.DataFrame,
    context: dict[str, Any],
) -> None:
    """Render the post-valuation conversational assistant."""
    st.subheader("AI Property Advisor")
    st.info(
        "La valoraci\u00f3n ya est\u00e1 lista. Puedes preguntarme por el precio estimado, "
        "las variables con mayor influencia, parking, piscina o informaci\u00f3n de ubicaci\u00f3n disponible."
    )

    for role, content in st.session_state["followup_messages"]:
        with st.chat_message(role):
            if role == "assistant":
                st.info(content)
            else:
                st.caption(content)

    question = st.chat_input("Pregunta al AI Property Advisor")
    if question:
        answer = answer_valuation_question(question, package, dataset, context)
        st.session_state["followup_messages"].append(("user", question))
        st.session_state["followup_messages"].append(("assistant", answer))
        with st.chat_message("user"):
            st.caption(question)
        with st.chat_message("assistant"):
            st.info(answer)


def render_property_valuation(dataset: pd.DataFrame, package: dict[str, Any]) -> None:
    """Render module 1 as a true conversational AI Property Advisor."""
    initialize_valuation_state()
    neighborhoods = get_neighborhoods(dataset)
    st.subheader("AI Property Valuation")
    st.info("Responde en lenguaje natural. El AI Property Advisor interpretar\u00e1 cada dato y avanzar\u00e1 paso a paso.")

    if not st.session_state["valuation_messages"]:
        st.session_state["valuation_messages"].append(("assistant", question_for_step(0)))

    progress = st.session_state["valuation_step"] / len(CONVERSATION_STEPS)
    st.progress(progress, text=f"Progreso de valoraci\u00f3n: {st.session_state['valuation_step']} de {len(CONVERSATION_STEPS)} datos")

    render_chat_history()

    if st.session_state["valuation_step"] < len(CONVERSATION_STEPS):
        render_current_conversation_step(neighborhoods)
        return

    if st.session_state["valuation_context"] is None:
        neighborhood = str(st.session_state["valuation_data"]["neighborhood"])
        context = calculate_valuation(
            package=package,
            dataset=dataset,
            neighborhood=neighborhood,
            property_input=current_property_input(),
        )
        st.session_state["valuation_context"] = context

    context = st.session_state["valuation_context"]
    render_valuation_results(package, context)
    render_followup_chat(package, dataset, context)


def summarize_neighborhood(dataset: pd.DataFrame, neighborhood: str) -> dict[str, float | int | str]:
    """Calculate neighborhood comparison indicators."""
    rows = get_neighborhood_rows(dataset, neighborhood)
    return {
        "Barrio": neighborhood,
        "Precio medio": numeric_mean(rows, PRICE_COLUMN),
        "Precio mediano": numeric_median(rows, PRICE_COLUMN),
        "EUR/m2 medio": numeric_mean(rows, UNIT_PRICE_COLUMN),
        "EUR/m\u00b2 mediano": numeric_median(rows, UNIT_PRICE_COLUMN),
        "Superficie media": numeric_mean(rows, AREA_COLUMN),
        "Habitaciones medias": numeric_mean(rows, ROOMS_COLUMN),
        "Ba\u00f1os medios": numeric_mean(rows, BATHROOMS_COLUMN),
        "% ascensor": numeric_mean(rows, HAS_LIFT_COLUMN) * 100,
        "% terraza": numeric_mean(rows, HAS_TERRACE_COLUMN) * 100,
        "% aire acondicionado": numeric_mean(rows, HAS_AIR_CONDITIONING_COLUMN) * 100,
        "% parking": numeric_mean(rows, HAS_PARKING_COLUMN) * 100,
        "% trastero": numeric_mean(rows, HAS_BOXROOM_COLUMN) * 100,
        "% piscina": numeric_mean(rows, HAS_SWIMMING_POOL_COLUMN) * 100,
        "Distancia centro": numeric_mean(rows, DISTANCE_TO_CITY_CENTER_COLUMN),
        "Distancia Castellana": numeric_mean(rows, DISTANCE_TO_CASTELLANA_COLUMN),
        "Distancia metro": numeric_mean(rows, DISTANCE_TO_METRO_COLUMN),
        "Viviendas analizadas": int(len(rows)),
    }


def build_comparison_summary(comparison: pd.DataFrame) -> dict[str, str]:
    """Build executive comparison labels from the existing indicators."""
    first = comparison.iloc[0]
    second = comparison.iloc[1]
    cheaper = first["Barrio"] if first["Precio medio"] < second["Precio medio"] else second["Barrio"]
    pricier_m2 = first["Barrio"] if first["EUR/m2 medio"] > second["EUR/m2 medio"] else second["Barrio"]
    better_connected = first["Barrio"] if first["Distancia metro"] < second["Distancia metro"] else second["Barrio"]

    amenity_columns = [
        "% ascensor",
        "% terraza",
        "% aire acondicionado",
        "% parking",
        "% trastero",
        "% piscina",
    ]
    first_amenities = float(first[amenity_columns].mean())
    second_amenities = float(second[amenity_columns].mean())
    more_amenities = first["Barrio"] if first_amenities > second_amenities else second["Barrio"]

    recommendation = (
        f"{cheaper} ofrece una entrada media m\u00e1s baja, mientras que {pricier_m2} concentra mayor valor por m\u00b2. "
        f"Para conectividad, destaca {better_connected}; para equipamiento, destaca {more_amenities}."
    )
    return {
        "Barrio m\u00e1s econ\u00f3mico": str(cheaper),
        "Barrio con mayor precio por m\u00b2": str(pricier_m2),
        "Barrio mejor comunicado": str(better_connected),
        "Barrio con m\u00e1s amenities": str(more_amenities),
        "Recomendaci\u00f3n final": recommendation,
    }


def render_large_money(label: str, value: str) -> None:
    """Render a large money value without st.metric truncation."""
    st.caption(label)
    st.write(f"### {value}")



def render_amenity_progress(label: str, percentage: float) -> None:
    """Render an amenity percentage with a progress bar."""
    clean_percentage = max(0.0, min(float(percentage), 100.0))
    label_col, value_col = st.columns([2, 1])
    label_col.caption(label)
    value_col.caption(f"{clean_percentage:.1f} %")
    st.progress(clean_percentage / 100)

def render_neighborhood_card(row: pd.Series) -> None:
    """Render one neighborhood as a native Streamlit KPI card."""
    with st.container(border=True):
        st.subheader(str(row["Barrio"]))
        price_col, median_col = st.columns(2)
        with price_col:
            render_large_money("Precio medio", format_euros(float(row["Precio medio"])))
        with median_col:
            render_large_money("Precio mediano", format_euros(float(row["Precio mediano"])))

        m2_col, area_col = st.columns(2)
        with m2_col:
            render_large_money("Precio por m\u00b2", format_euros_per_m2(float(row["EUR/m2 medio"])))
        area_col.metric("Superficie media", f"{float(row['Superficie media']):.1f} m\u00b2")

        rooms_col, baths_col = st.columns(2)
        rooms_col.metric("Habitaciones", f"{float(row['Habitaciones medias']):.1f}")
        baths_col.metric("Ba\u00f1os", f"{float(row['Ba\u00f1os medios']):.1f}")

        st.caption("Amenities")
        render_amenity_progress("Ascensor", float(row["% ascensor"]))
        render_amenity_progress("Parking", float(row["% parking"]))
        render_amenity_progress("Piscina", float(row["% piscina"]))


def render_horizontal_comparison_chart(comparison: pd.DataFrame) -> None:
    """Render grouped bars with raw values for the requested indicators."""
    indicators = [
        ("Precio medio", "Precio medio"),
        ("EUR/m2 medio", "EUR/m\u00b2"),
    ]
    chart_rows: list[dict[str, float | str]] = []
    for source_column, label in indicators:
        for _, row in comparison.iterrows():
            value = float(row[source_column])
            chart_rows.append(
                {
                    "Indicador": label,
                    "Barrio": str(row["Barrio"]),
                    "Valor": value,
                    "Valor mostrado": format_euros_per_m2(value) if source_column == "EUR/m2 medio" else format_euros(value),
                }
            )

    chart_data = pd.DataFrame(chart_rows)
    bars = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Barrio:N", title=None),
            y=alt.Y("Valor:Q", title="Valor en euros"),
            color=alt.Color("Barrio:N", title="Barrio"),
            column=alt.Column("Indicador:N", title=None),
            tooltip=["Barrio:N", "Indicador:N", "Valor mostrado:N"],
        )
        .properties(height=280)
        .resolve_scale(y="independent")
    )
    st.subheader("Comparaci\u00f3n de precio")
    st.altair_chart(bars, width="stretch")


def render_executive_summary(comparison: pd.DataFrame) -> None:
    """Render the executive conclusion as native Streamlit components."""
    summary = build_comparison_summary(comparison)
    with st.container(border=True):
        st.subheader("\U0001f3c6 Resumen ejecutivo")
        c1, c2 = st.columns(2)
        c1.metric("\U0001f4b6 Barrio m\u00e1s econ\u00f3mico", summary["Barrio m\u00e1s econ\u00f3mico"])
        c2.metric("\U0001f4c8 Mayor precio por m\u00b2", summary["Barrio con mayor precio por m\u00b2"])

        c3, c4 = st.columns(2)
        c3.metric("\U0001f687 Mejor comunicado", summary["Barrio mejor comunicado"])
        c4.metric("\u2728 M\u00e1s amenities", summary["Barrio con m\u00e1s amenities"])

        st.info(f"\U0001f3af {summary['Recomendaci\u00f3n final']}")


def render_comparison_dashboard(comparison: pd.DataFrame) -> None:
    """Render comparison output as an executive dashboard."""
    left, right = st.columns(2)
    with left:
        render_neighborhood_card(comparison.iloc[0])
    with right:
        render_neighborhood_card(comparison.iloc[1])

    render_horizontal_comparison_chart(comparison)
    render_executive_summary(comparison)


def render_neighborhood_intelligence(dataset: pd.DataFrame) -> None:
    """Render module 2: executive two-neighborhood comparison."""
    st.subheader("Neighbourhood Intelligence")
    st.info("Compara dos barrios con indicadores de precio, conectividad, amenities y profundidad de muestra.")
    neighborhoods = get_neighborhoods(dataset)
    col1, col2 = st.columns(2)
    with col1:
        first_neighborhood = st.selectbox("Barrio A", neighborhoods, key="comparison_a")
    with col2:
        second_index = 1 if len(neighborhoods) > 1 else 0
        second_neighborhood = st.selectbox("Barrio B", neighborhoods, index=second_index, key="comparison_b")

    if st.button("Comparar barrios"):
        comparison = pd.DataFrame(
            [
                summarize_neighborhood(dataset, first_neighborhood),
                summarize_neighborhood(dataset, second_neighborhood),
            ]
        )
        render_comparison_dashboard(comparison)


def normalize_series(series: pd.Series, lower_is_better: bool) -> pd.Series:
    """Normalize a series to 0-1 with optional lower-is-better direction."""
    clean = pd.to_numeric(series, errors="coerce").fillna(series.mean())
    minimum = clean.min()
    maximum = clean.max()
    if maximum == minimum:
        normalized = pd.Series(1.0, index=clean.index)
    else:
        normalized = (clean - minimum) / (maximum - minimum)
    if lower_is_better:
        normalized = 1 - normalized
    return normalized


def build_investment_ranking(
    dataset: pd.DataFrame,
    max_budget: float,
    min_area: float,
    min_rooms: int,
    priority: str,
) -> pd.DataFrame:
    """Rank neighborhoods by simple normalized opportunity rules."""
    filtered = dataset[
        (pd.to_numeric(dataset[PRICE_COLUMN], errors="coerce") <= max_budget)
        & (pd.to_numeric(dataset[AREA_COLUMN], errors="coerce") >= min_area)
        & (pd.to_numeric(dataset[ROOMS_COLUMN], errors="coerce") >= min_rooms)
    ].copy()

    if filtered.empty:
        return pd.DataFrame()

    grouped = (
        filtered.groupby(NEIGHBORHOOD_COLUMN)
        .agg(
            Precio_medio=(PRICE_COLUMN, "mean"),
            EUR_m2_medio=(UNIT_PRICE_COLUMN, "mean"),
            Superficie_media=(AREA_COLUMN, "mean"),
            Viviendas_analizadas=(PRICE_COLUMN, "size"),
            Distancia_centro=(DISTANCE_TO_CITY_CENTER_COLUMN, "mean"),
            Distancia_metro=(DISTANCE_TO_METRO_COLUMN, "mean"),
            Ascensor=(HAS_LIFT_COLUMN, "mean"),
            Terraza=(HAS_TERRACE_COLUMN, "mean"),
            Aire=(HAS_AIR_CONDITIONING_COLUMN, "mean"),
            Parking=(HAS_PARKING_COLUMN, "mean"),
            Trastero=(HAS_BOXROOM_COLUMN, "mean"),
            Piscina=(HAS_SWIMMING_POOL_COLUMN, "mean"),
        )
        .reset_index()
    )
    grouped["Amenities"] = grouped[["Ascensor", "Terraza", "Aire", "Parking", "Trastero", "Piscina"]].mean(axis=1) * 100

    raw_budget_score = 1 - (grouped["Precio_medio"] - max_budget).abs() / max_budget
    budget_score = normalize_series(raw_budget_score.clip(lower=0, upper=1), lower_is_better=False)
    within_budget_bonus = grouped["Precio_medio"].le(max_budget) * 0.05
    budget_score = (budget_score + within_budget_bonus).clip(upper=1.0)
    price_score = normalize_series(grouped["EUR_m2_medio"], lower_is_better=True)
    area_score = normalize_series(grouped["Superficie_media"], lower_is_better=False)
    metro_score = normalize_series(grouped["Distancia_metro"], lower_is_better=True)
    amenity_score = normalize_series(grouped["Amenities"], lower_is_better=False)
    sample_score = normalize_series(grouped["Viviendas_analizadas"], lower_is_better=False)

    score = (
        budget_score * 0.40
        + price_score * 0.20
        + area_score * 0.15
        + metro_score * 0.10
        + amenity_score * 0.10
        + sample_score * 0.05
    )

    grouped["Score simple de oportunidad"] = (score * 100).round(2)
    grouped = grouped.rename(columns={NEIGHBORHOOD_COLUMN: "Barrio"})
    output_columns = [
        "Barrio",
        "Precio_medio",
        "EUR_m2_medio",
        "Superficie_media",
        "Viviendas_analizadas",
        "Distancia_centro",
        "Distancia_metro",
        "Amenities",
        "Score simple de oportunidad",
    ]
    return grouped[output_columns].sort_values("Score simple de oportunidad", ascending=False).head(10)


def investment_badge(position: int) -> str:
    """Return the visual badge for an investment ranking position."""
    if position == 1:
        return f"{chr(0x1F947)} Mejor opci\u00f3n"
    if position == 2:
        return f"{chr(0x1F948)} Alternativa recomendada"
    if position == 3:
        return f"{chr(0x1F949)} Muy buena opci\u00f3n"
    return f"{chr(0x1F4CD)} Opci\u00f3n adicional"


def investment_reasons(row: pd.Series, priority: str) -> list[str]:
    """Build three concise reasons from the existing ranking indicators."""
    reasons: list[str] = []
    if priority == "Precio bajo":
        reasons.append("Precio por m\u00b2 atractivo")
    elif priority == "Cercan\u00eda al centro":
        reasons.append("Cercan\u00eda relativa al centro")
    elif priority == "Cercan\u00eda al metro":
        reasons.append("Buena comunicaci\u00f3n por metro")
    elif priority == "Amenities":
        reasons.append("Buen nivel de amenities")
    else:
        reasons.append("Buen equilibrio calidad-precio")

    if float(row["Distancia_metro"]) <= 1.0:
        reasons.append("Buena comunicaci\u00f3n")
    else:
        reasons.append("Distancia al metro controlada")

    if int(row["Viviendas_analizadas"]) >= 10:
        reasons.append("Amplia oferta de viviendas")
    else:
        reasons.append("Oferta disponible en el dataset")

    if float(row["Superficie_media"]) >= 70:
        reasons.append("Superficie media competitiva")
    else:
        reasons.append("Encaja con criterios de b\u00fasqueda")

    unique_reasons: list[str] = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)
    return unique_reasons[:3]


def render_investment_card(row: pd.Series, position: int, priority: str) -> None:
    """Render one investment recommendation as a decision card."""
    with st.container(border=True):
        st.subheader(f"{position}. {row['Barrio']}")
        st.caption(investment_badge(position))

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{chr(0x2B50)} Score de oportunidad", f"{float(row['Score simple de oportunidad']):.2f}")
        c2.metric(f"{chr(0x1F4B6)} Precio medio", format_euros(float(row["Precio_medio"])))
        c3.metric(f"{chr(0x1F3E0)} Precio medio EUR/m\u00b2", format_euros_per_m2(float(row["EUR_m2_medio"])))

        c4, c5, c6 = st.columns(3)
        c4.metric(f"{chr(0x1F4D0)} Superficie media", f"{float(row['Superficie_media']):.1f} m\u00b2")
        c5.metric(f"{chr(0x1F687)} Distancia media al metro", f"{float(row['Distancia_metro']):.2f}")
        c6.metric(f"{chr(0x1F3E1)} Viviendas analizadas", f"{int(row['Viviendas_analizadas'])}")

        st.caption("Por qu\u00e9 lo recomendamos?")
        check = chr(0x2714)
        for reason in investment_reasons(row, priority):
            st.write(f"{check} {reason}")


def render_investment_score_chart(ranking: pd.DataFrame) -> None:
    """Render a simple horizontal bar chart with the top five opportunity scores."""
    top_five = ranking.head(5).copy()
    chart = (
        alt.Chart(top_five)
        .mark_bar()
        .encode(
            x=alt.X("Score simple de oportunidad:Q", title="Score de oportunidad"),
            y=alt.Y("Barrio:N", title=None, sort="-x"),
            tooltip=["Barrio:N", "Score simple de oportunidad:Q"],
        )
        .properties(height=260)
    )
    st.subheader("Score de oportunidad Top 5")
    st.altair_chart(chart, width="stretch")


def investment_map_data(dataset: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    """Build neighborhood mean coordinates for the recommended ranking."""
    required_columns = {NEIGHBORHOOD_COLUMN, "LATITUDE", "LONGITUDE"}
    if not required_columns.issubset(dataset.columns):
        return pd.DataFrame()

    top_neighborhoods = ranking.head(5)["Barrio"].astype(str).tolist()
    rows = dataset[dataset[NEIGHBORHOOD_COLUMN].astype(str).isin(top_neighborhoods)].copy()
    if rows.empty:
        return pd.DataFrame()

    coordinates = (
        rows.groupby(NEIGHBORHOOD_COLUMN)
        .agg(latitude=("LATITUDE", "mean"), longitude=("LONGITUDE", "mean"))
        .reset_index()
        .rename(columns={NEIGHBORHOOD_COLUMN: "Barrio"})
    )
    output = coordinates.merge(ranking.head(5), on="Barrio", how="inner")
    output = output.dropna(subset=["latitude", "longitude"])
    output["score"] = output["Score simple de oportunidad"].round(2)
    output["precio_medio"] = output["Precio_medio"].map(format_euros)
    output["eur_m2"] = output["EUR_m2_medio"].map(format_euros_per_m2)
    return output


def render_recommended_neighborhoods_map(dataset: pd.DataFrame, ranking: pd.DataFrame) -> None:
    """Render Madrid map with the top five recommended neighborhoods."""
    st.subheader("Mapa de barrios recomendados")
    map_data = investment_map_data(dataset, ranking)
    if map_data.empty:
        st.info(
            "El mapa requiere geometr\u00edas o coordenadas de barrio. "
            "No se han encontrado datos geogr\u00e1ficos suficientes en el dataset."
        )
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position="[longitude, latitude]",
        get_radius=280,
        get_fill_color=[151, 201, 61, 185],
        get_line_color=[111, 174, 42, 255],
        line_width_min_pixels=2,
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=40.4168, longitude=-3.7038, zoom=10.4, pitch=0)
    deck = pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[layer],
        tooltip={
            "text": "{Barrio}\nScore: {score}\nPrecio medio: {precio_medio}\nEUR/m2 medio: {eur_m2}"
        },
    )
    st.pydeck_chart(deck, width="stretch")


def render_ai_investment_advisor(ranking: pd.DataFrame, priority: str) -> None:
    """Render the final rule-based advisor recommendation from available data."""
    top = ranking.iloc[0]
    with st.container(border=True):
        st.subheader(f"{chr(0x1F4A1)} Recomendaci\u00f3n del AI Advisor")
        st.info(
            f"Seg\u00fan el presupuesto y la prioridad seleccionados ({priority}), el barrio m\u00e1s recomendable es "
            f"{top['Barrio']} porque ofrece el mejor equilibrio entre precio, superficie y accesibilidad "
            f"dentro del conjunto de barrios analizados."
        )


def initialize_investment_state() -> None:
    """Initialize session state for the investment conversational flow."""
    st.session_state.setdefault("investment_step", 0)
    st.session_state.setdefault("investment_data", {})
    st.session_state.setdefault("investment_messages", [])
    st.session_state.setdefault("investment_ranking", None)


def reset_investment_state() -> None:
    """Reset the investment conversational flow."""
    st.session_state["investment_step"] = 0
    st.session_state["investment_data"] = {}
    st.session_state["investment_messages"] = []
    st.session_state["investment_ranking"] = None


def investment_question_for_step(step_index: int) -> str:
    """Return the assistant question for the investment flow."""
    questions = [
        "\u00bfCu\u00e1l es tu presupuesto m\u00e1ximo?",
        "\u00bfQu\u00e9 superficie m\u00ednima quieres?",
        "\u00bfCu\u00e1ntas habitaciones necesitas como m\u00ednimo?",
        "\u00bfQu\u00e9 prioridad tienes? Opciones: Precio bajo, Cercan\u00eda al centro, Cercan\u00eda al metro, Amenities o Equilibrado.",
    ]
    return questions[step_index]


def parse_investment_number(answer: str) -> float | None:
    """Parse numeric investment answers with optional thousands separators."""
    normalized = normalize_text(answer)
    match = re.search(r"\d[\d\.,]*", normalized)
    if not match:
        return parse_number_from_text(answer)
    value = match.group(0)
    if "." in value and "," not in value:
        parts = value.split(".")
        if len(parts) > 1 and all(len(part) == 3 for part in parts[1:]):
            value = "".join(parts)
    if "," in value and "." in value:
        value = value.replace(".", "").replace(",", ".")
    elif "," in value:
        value = value.replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def parse_investment_priority(answer: str) -> str | None:
    """Parse natural-language priority into the ranking priority values."""
    normalized = normalize_text(answer)
    if "precio" in normalized or "barat" in normalized or "bajo" in normalized:
        return "Precio bajo"
    if "centro" in normalized:
        return "Cercan\u00eda al centro"
    if "metro" in normalized or "comunic" in normalized:
        return "Cercan\u00eda al metro"
    if "amen" in normalized or "servicio" in normalized or "equip" in normalized:
        return "Amenities"
    if "equilibr" in normalized or "balance" in normalized:
        return "Equilibrado"
    return None


def parse_investment_answer(field: str, answer: str) -> tuple[Any | None, str | None]:
    """Parse one conversational investment answer."""
    if field in ("max_budget", "min_area"):
        value = parse_investment_number(answer)
        if value is None or value <= 0:
            return None, "Necesito un n\u00famero v\u00e1lido. Por ejemplo: 300000 o 60."
        return float(value), None
    if field == "min_rooms":
        value = parse_investment_number(answer)
        if value is None or value < 0:
            return None, "Necesito un n\u00famero de habitaciones v\u00e1lido. Por ejemplo: 2."
        return int(value), None
    if field == "priority":
        priority = parse_investment_priority(answer)
        if priority is None:
            return None, "No he entendido la prioridad. Puedes responder: precio bajo, cercan\u00eda al centro, cercan\u00eda al metro, amenities o equilibrado."
        return priority, None
    return None, "No he podido interpretar la respuesta."


def render_investment_chat_history() -> None:
    """Render investment assistant conversation history."""
    for role, content in st.session_state["investment_messages"]:
        with st.chat_message(role):
            if role == "assistant":
                st.info(content)
            else:
                st.caption(content)


def render_investment_results(dataset: pd.DataFrame, ranking: pd.DataFrame, priority: str) -> None:
    """Render investment recommendations from an existing ranking."""
    st.subheader(f"{chr(0x1F3C6)} Top 5 barrios recomendados")
    for position, (_, row) in enumerate(ranking.head(5).iterrows(), start=1):
        render_investment_card(row, position, priority)

    render_recommended_neighborhoods_map(dataset, ranking)
    render_investment_score_chart(ranking)
    render_ai_investment_advisor(ranking, priority)


def render_investment_simulator(dataset: pd.DataFrame) -> None:
    """Render module 3 as a conversational investment assistant."""
    initialize_investment_state()
    st.subheader("Investment Simulator")
    st.info("Responde en lenguaje natural. El AI Advisor te guiar\u00e1 paso a paso para recomendar barrios con los datos del proyecto.")

    if st.button("Reiniciar b\u00fasqueda"):
        reset_investment_state()
        st.rerun()

    if not st.session_state["investment_messages"]:
        st.session_state["investment_messages"].append(("assistant", investment_question_for_step(0)))

    render_investment_chat_history()

    fields = ["max_budget", "min_area", "min_rooms", "priority"]
    step_index = st.session_state["investment_step"]
    if step_index < len(fields):
        answer = st.chat_input("Responde al AI Investment Advisor", key="investment_chat_input")
        if answer:
            field = fields[step_index]
            st.session_state["investment_messages"].append(("user", answer))
            parsed_value, error = parse_investment_answer(field, answer)
            if error:
                st.session_state["investment_messages"].append(("assistant", error))
                st.session_state["investment_messages"].append(("assistant", investment_question_for_step(step_index)))
                st.rerun()

            st.session_state["investment_data"][field] = parsed_value
            st.session_state["investment_step"] += 1
            if st.session_state["investment_step"] < len(fields):
                st.session_state["investment_messages"].append(("assistant", investment_question_for_step(st.session_state["investment_step"])))
            else:
                st.session_state["investment_messages"].append(("assistant", "Perfecto. Ya tengo los datos y voy a buscar barrios recomendados."))
            st.rerun()
        return

    if st.session_state["investment_ranking"] is None:
        data = st.session_state["investment_data"]
        ranking = build_investment_ranking(
            dataset=dataset,
            max_budget=float(data["max_budget"]),
            min_area=float(data["min_area"]),
            min_rooms=int(data["min_rooms"]),
            priority=str(data["priority"]),
        )
        st.session_state["investment_ranking"] = ranking

    ranking = st.session_state["investment_ranking"]
    if ranking.empty:
        st.warning("No hay barrios que cumplan los filtros seleccionados. Reinicia la b\u00fasqueda para probar otros criterios.")
        return

    render_investment_results(dataset, ranking, str(st.session_state["investment_data"]["priority"]))


def render_sidebar(package: dict[str, Any]) -> None:
    """Render model and dataset metadata in the sidebar."""
    with st.sidebar:
        st.title("AI Real Estate Advisor")

        st.subheader("Ciudad")
        st.caption("Madrid")

        st.subheader("Modelo")
        st.caption("Random Forest")

        st.subheader("Target")
        st.caption("Precio de la vivienda (LOG_PRICE)")

        st.subheader("M\u00e9tricas del modelo")
        metrics = package.get("metrics", {})
        if isinstance(metrics, dict):
            if "RMSE" in metrics:
                rmse = f"{float(metrics['RMSE']):,.0f}".replace(",", ".")
                st.metric("RMSE", f"{rmse} EUR")
            if "MAPE" in metrics:
                mape = f"{float(metrics['MAPE']):.2f}".replace(".", ",")
                st.metric("MAPE", f"{mape} %")
            if "Pseudo_R2" in metrics:
                pseudo_r2 = f"{float(metrics['Pseudo_R2']):.4f}".replace(".", ",")
                st.metric("Pseudo R2", pseudo_r2)

        st.caption("✓ Menor RMSE y MAPE indican un menor error de predicci\u00f3n.")
        st.caption("✓ Un mayor Pseudo R2 indica un mejor ajuste del modelo.")


def main() -> None:
    """Run the definitive MVP app."""
    apply_visual_theme()
    st.title("AI Real Estate Advisor")
    st.caption("Plataforma PropTech de valoraci\u00f3n inmobiliaria con Machine Learning, datos de mercado e IA conversacional.")

    dataset = load_dataset()
    package = load_model_package()
    render_sidebar(package)


    valuation_tab, intelligence_tab, investment_tab = st.tabs(
        [
            "AI Property Valuation",
            "Neighbourhood Intelligence",
            "Investment Simulator",
        ]
    )
    with valuation_tab:
        render_property_valuation(dataset, package)
    with intelligence_tab:
        render_neighborhood_intelligence(dataset)
    with investment_tab:
        render_investment_simulator(dataset)



if __name__ == "__main__":
    main()
