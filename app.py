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
import gdown
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

def dataset_load_error(reason: str, path: Path, size_bytes: int | None, preview: str) -> None:
    """Show a clear dataset loading error and stop the Streamlit app."""
    size_label = "No disponible" if size_bytes is None else f"{size_bytes} bytes"
    st.error(
        "No se ha podido cargar el dataset. "
        f"{reason} "
        "Revisa que data/dataset.csv exista en el despliegue y contenga un CSV v\u00e1lido."
    )
    st.code(
        f"Ruta utilizada: {path}\n"
        f"Tama\u00f1o del archivo: {size_label}\n"
        f"Primeras 200 letras del contenido:\n{preview or '[sin contenido]'}",
        language="text",
    )
    st.stop()


def load_dataset() -> pd.DataFrame:
    """Load the cleaned Madrid real estate dataset with deployment diagnostics."""
    dataset_path = DATASET_PATH.resolve()

    if not DATASET_PATH.exists():
        dataset_load_error("El archivo no existe.", dataset_path, None, "")

    size_bytes = DATASET_PATH.stat().st_size
    preview = DATASET_PATH.read_text(encoding="utf-8", errors="replace")[:200]
    normalized_preview = preview.lstrip().casefold()

    if size_bytes == 0:
        dataset_load_error("El archivo est\u00e1 vac\u00edo.", dataset_path, size_bytes, preview)

    if normalized_preview.startswith("<!doctype html") or normalized_preview.startswith("<html") or "<html" in normalized_preview[:80]:
        dataset_load_error("El archivo parece HTML, no un CSV.", dataset_path, size_bytes, preview)

    try:
        dataset = pd.read_csv(DATASET_PATH)
    except pd.errors.EmptyDataError:
        dataset_load_error("Pandas no ha encontrado columnas que leer.", dataset_path, size_bytes, preview)
    except pd.errors.ParserError as exc:
        dataset_load_error(f"Pandas no ha podido interpretar el CSV: {exc}", dataset_path, size_bytes, preview)

    if dataset.empty or len(dataset.columns) == 0:
        dataset_load_error("El CSV se ha le\u00eddo sin filas o sin columnas.", dataset_path, size_bytes, preview)

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
        output = gdown.download(
            MODEL_PACKAGE_URL,
            str(MODEL_PACKAGE_PATH),
            quiet=False,
            fuzzy=True,
        )
        return output is not None and MODEL_PACKAGE_PATH.exists()
    except Exception:
        st.error("No se ha podido descargar el modelo. Revisa la conexi\u00f3n o el enlace configurado.")
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
    """Return the most important global model feature shown to the user."""
    importance = feature_importance(package)
    importance = importance[importance["Variable"] != DISTANCE_TO_METRO_COLUMN]
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


def translate_model_feature(feature: Any) -> str:
    """Translate internal model feature names into user-facing labels."""
    if str(feature) == DISTANCE_TO_METRO_COLUMN:
        return "Ubicaci\u00f3n de barrio"
    translations = {
        LOG_AREA_COLUMN: "Superficie construida",
        AREA_COLUMN: "Superficie construida",
        MEAN_UNITPRICE_BY_LOCATION_COLUMN: "Precio medio del barrio",
        HAS_LIFT_COLUMN: "Ascensor",
        DISTANCE_TO_CITY_CENTER_COLUMN: "Distancia al centro",
        "DISTANCE_TO_CENTER": "Distancia al centro",
        DISTANCE_TO_CASTELLANA_COLUMN: "Distancia al Paseo de la Castellana",
        ROOMS_COLUMN: "Habitaciones",
        BATHROOMS_COLUMN: "Ba\u00f1os",
        HAS_SWIMMING_POOL_COLUMN: "Piscina",
        HAS_PARKING_COLUMN: "Garaje",
        HAS_TERRACE_COLUMN: "Terraza",
        HAS_AIR_CONDITIONING_COLUMN: "Aire acondicionado",
        HAS_BOXROOM_COLUMN: "Trastero",
    }
    return translations.get(str(feature), str(feature).replace("_", " ").title())


def format_amenities(names: list[str]) -> str:
    """Format amenity labels naturally for VERA responses."""
    if not names:
        return "sin amenities destacados"
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} y {names[-1]}"


def vera_context(context: dict[str, Any], package: dict[str, Any]) -> dict[str, Any]:
    """Build the valuation context VERA uses before every answer."""
    property_input = context["input"]
    amenities = [
        ("ascensor", property_input.get(HAS_LIFT_COLUMN, 0)),
        ("terraza", property_input.get(HAS_TERRACE_COLUMN, 0)),
        ("aire acondicionado", property_input.get(HAS_AIR_CONDITIONING_COLUMN, 0)),
        ("garaje", property_input.get(HAS_PARKING_COLUMN, 0)),
        ("trastero", property_input.get(HAS_BOXROOM_COLUMN, 0)),
        ("piscina", property_input.get(HAS_SWIMMING_POOL_COLUMN, 0)),
    ]
    importance = feature_importance(package)
    importance = importance[importance["Variable"] != DISTANCE_TO_METRO_COLUMN].head(5).copy()
    importance["Label"] = importance["Variable"].map(translate_model_feature)
    top_feature_details = [
        {"label": str(row["Label"]), "importance": float(row["Importance"])}
        for _, row in importance.iterrows()
    ]
    top_labels = [item["label"] for item in top_feature_details]
    top_feature_lines = [
        explain_feature_importance(item["label"], item["importance"], index)
        for index, item in enumerate(top_feature_details[:3])
    ]
    rooms = int(property_input[ROOMS_COLUMN])
    bathrooms = int(property_input[BATHROOMS_COLUMN])
    return {
        "neighborhood": context["neighborhood"],
        "area": float(property_input[AREA_COLUMN]),
        "rooms": rooms,
        "bathrooms": bathrooms,
        "rooms_text": f"{rooms} habitaci\u00f3n" if rooms == 1 else f"{rooms} habitaciones",
        "bathrooms_text": f"{bathrooms} ba\u00f1o" if bathrooms == 1 else f"{bathrooms} ba\u00f1os",
        "active_amenities": [name for name, value in amenities if int(value) == 1],
        "missing_amenities": [name for name, value in amenities if int(value) == 0],
        "estimated_price": format_euros(context["estimated_price"]),
        "estimated_unit_price": format_euros_per_m2(context["estimated_unit_price"]),
        "average_price": format_euros(context["average_price"]),
        "average_unit_price": format_euros_per_m2(context["average_unit_price"]),
        "distance_to_city_center": float(context["distance_to_city_center"]),
        "distance_to_castellana": float(context["distance_to_castellana"]),
        "top_features": top_labels,
        "top_feature_details": top_feature_details,
        "top_feature_lines": top_feature_lines,
        "top_features_text": format_amenities(top_labels[:3]) if top_labels else "los datos principales de la vivienda y el barrio",
    }


def explain_feature_importance(label: str, importance: float, index: int) -> str:
    """Explain model feature importance in plain real-estate language."""
    percent = round(float(importance) * 100)
    if index == 0:
        meaning = "es el factor que m\u00e1s influye en esta valoraci\u00f3n realizada por el modelo"
    elif index == 1:
        meaning = "es el segundo factor m\u00e1s relevante para estimar el precio"
    elif percent >= 10:
        meaning = "tiene una influencia relevante y ayuda a ajustar la estimaci\u00f3n"
    else:
        meaning = "tiene una influencia menor, aunque puede marcar diferencias entre viviendas similares"
    return f"**{label} ({percent}%)**: {meaning}."


def feature_relevance(ctx: dict[str, Any], label: str) -> str:
    """Explain whether a user-facing factor is prominent in the current model reading."""
    if label in ctx["top_features"]:
        return f"**{label}** aparece entre los factores con m\u00e1s peso en esta valoraci\u00f3n."
    return f"**{label}** no aparece entre los primeros factores, pero puede mejorar la percepci\u00f3n comercial si encaja con el barrio."


def vera_followup_suggestion(intent: str, question: str) -> str:
    """Return one varied natural continuation suggestion for VERA."""
    suggestions = {
        "PRICE_EXPLANATION": [
            "Si quieres profundizar, podemos revisar cada factor uno por uno.",
            "Tambi\u00e9n puedo explicarte qu\u00e9 parte parece venir m\u00e1s condicionada por el barrio.",
        ],
        "LOCATION_EXPLANATION": [
            "Puedo explicarte por qu\u00e9 el barrio influye tanto en esta valoraci\u00f3n.",
            "Tambi\u00e9n puedo comparar esta vivienda con otro barrio.",
        ],
        "HYPOTHETICAL_SCENARIO": [
            "Si quieres, puedo analizar otra mejora.",
            "Tambi\u00e9n puedo ayudarte a priorizar qu\u00e9 cambio tendr\u00eda m\u00e1s sentido comercial.",
        ],
        "INVESTMENT": [
            "Si quieres, puedo preparar una lista de puntos a revisar antes de comprar.",
            "Tambi\u00e9n puedo ayudarte a interpretar el resultado desde una negociaci\u00f3n de compra.",
        ],
        "MODEL_LIMITATIONS": [
            "Puedo separar qu\u00e9 aspectos s\u00ed ve VERA y cu\u00e1les conviene validar en una visita.",
            "Si quieres, podemos revisar qu\u00e9 incertidumbres pesan m\u00e1s antes de tomar una decisi\u00f3n.",
        ],
        "FEATURE_IMPORTANCE": [
            "Si quieres, puedo traducir cada factor a una lectura inmobiliaria sencilla.",
            "Tambi\u00e9n puedo explicar cu\u00e1l de esos factores ser\u00eda m\u00e1s accionable.",
        ],
        "PROPERTY_IMPROVEMENTS": [
            "Si quieres, puedo ordenar las mejoras por impacto comercial esperado.",
            "Tambi\u00e9n puedo analizar otra mejora concreta que tengas en mente.",
        ],
        "NEIGHBOURHOOD": [
            "Tambi\u00e9n puedo comparar esta vivienda con otro barrio.",
            "Puedo explicarte c\u00f3mo encaja el precio por m\u00b2 en esta zona.",
        ],
    }
    options = suggestions.get(intent, ["Puedo ayudarte a interpretar el resultado desde otro \u00e1ngulo."])
    return options[len(question) % len(options)]


def build_vera_response(
    response: str,
    why: str,
    factors: list[str],
    recommendation: str,
    intent: str,
    question: str,
) -> str:
    """Build a structured VERA answer."""
    factor_text = "\n".join(f"- {factor}" for factor in factors)
    return (
        f"## Respuesta breve\n\n{response}\n\n"
        f"## \u00bfPor qu\u00e9?\n\n{why}\n\n"
        f"## Factores relevantes\n\n{factor_text}\n\n"
        f"## Recomendaci\u00f3n\n\n{recommendation} {vera_followup_suggestion(intent, question)}"
    )


def detect_vera_intent(question: str) -> str:
    """Classify a VERA user question into one rule-based intent."""
    normalized_question = normalize_text(question)

    def has_any(*keywords: str) -> bool:
        return any(keyword in normalized_question for keyword in keywords)

    location_terms = (
        "distancia al centro", "distancia a castellana", "castellana", "ubicacion", "localizacion",
        "centro", "accesibilidad", "cerca", "lejos", "ubic", "localiz", "comunic",
        "influencia de la ubicacion", "influye la ubicacion", "influye el barrio",
    )
    if has_any(*location_terms):
        return "LOCATION_EXPLANATION"
    if has_any("limitacion", "limitaciones", "error", "fiable", "precision", "confianza", "tasacion", "oficial"):
        return "MODEL_LIMITATIONS"
    if has_any("modelo", "random forest", "machine learning", "algoritmo", "como calcula", "como funciona"):
        return "MODEL_EXPLANATION"
    if has_any("y si", "que pasaria", "si tuviera", "si cambiara", "tuviera", "cambiara", "anadiera", "reformada", "reformado", "habitacion", "piscina", "garaje", "parking", "terraza"):
        return "HYPOTHETICAL_SCENARIO"
    if has_any("inversion", "invertir", "rentabilidad", "oportunidad", "comprar", "compraria", "buena compra"):
        return "INVESTMENT"
    if has_any("mejora", "mejorar", "reforma", "reformar", "subir valor", "equipamiento", "amenities"):
        return "PROPERTY_IMPROVEMENTS"
    if has_any("variable", "variables", "influ", "importancia", "importante", "pesa", "feature"):
        return "FEATURE_IMPORTANCE"
    if has_any("barrio", "zona", "entorno"):
        return "NEIGHBOURHOOD"
    if has_any("precio", "valoracion", "estimacion", "resultado", "euros", "m2", "por que", "porque"):
        return "PRICE_EXPLANATION"
    return "GENERAL"


def detect_hypothetical_changes(normalized_question: str) -> list[tuple[str, str, str]]:
    """Detect all hypothetical changes mentioned in a VERA question."""
    checks = [
        (("piscina",), "Piscina", HAS_SWIMMING_POOL_COLUMN, "aumentar\u00eda el atractivo al a\u00f1adir un atributo de ocio y calidad"),
        (("garaje", "parking"), "Garaje", HAS_PARKING_COLUMN, "mejorar\u00eda comodidad y diferenciaci\u00f3n para compradores que valoran aparcamiento"),
        (("terraza",), "Terraza", HAS_TERRACE_COLUMN, "sumar\u00eda espacio exterior, un atributo muy visible en la decisi\u00f3n de compra"),
        (("ascensor",), "Ascensor", HAS_LIFT_COLUMN, "mejorar\u00eda accesibilidad y liquidez comercial"),
        (("aire", "aire acondicionado"), "Aire acondicionado", HAS_AIR_CONDITIONING_COLUMN, "reforzar\u00eda confort y equipamiento percibido"),
        (("trastero",), "Trastero", HAS_BOXROOM_COLUMN, "aportar\u00eda almacenamiento y funcionalidad"),
        (("habitacion", "habitaci", "dormitorio"), "Habitaciones", ROOMS_COLUMN, "cambiar\u00eda la distribuci\u00f3n y el perfil de comprador objetivo"),
        (("bano", "banos", "ba\u00f1o"), "Ba\u00f1os", BATHROOMS_COLUMN, "mejorar\u00eda funcionalidad para hogares con m\u00e1s ocupantes"),
        (("superficie", "metros", "m2"), "Superficie construida", AREA_COLUMN, "afectar\u00eda directamente al valor total y al precio por m\u00b2"),
        (("barrio", "zona", "otro barrio", "cambiara"), "Barrio", NEIGHBORHOOD_COLUMN, "cambiar\u00eda la referencia de mercado y el precio medio comparable"),
        (("reform",), "Reforma", "REFORM", "mejorar\u00eda percepci\u00f3n y comercializaci\u00f3n, aunque no est\u00e1 medida directamente como variable"),
    ]
    changes = []
    for keywords, label, feature, explanation in checks:
        if any(keyword in normalized_question for keyword in keywords):
            changes.append((label, feature, explanation))
    return changes


def answer_valuation_question(
    question: str,
    package: dict[str, Any],
    dataset: pd.DataFrame,
    context: dict[str, Any],
) -> str:
    """Answer VERA follow-up questions from the last valuation context."""
    normalized_question = normalize_text(question)
    clean_question = normalized_question.translate(str.maketrans("", "", "\u00bf?!.:,;"))
    clean_question = " ".join(clean_question.split())
    topic_question = clean_question
    for prefix in ("y si ", "y ", "o sea ", "entonces "):
        if topic_question.startswith(prefix):
            topic_question = topic_question[len(prefix):].strip()
            break
    intent = detect_vera_intent(question)
    ctx = vera_context(context, package)
    active_amenities_text = format_amenities(ctx["active_amenities"])
    missing_amenities_text = format_amenities(ctx["missing_amenities"][:4])
    importance_factors = ctx["top_feature_lines"] or ["No hay importancia de variables disponible en el paquete del modelo."]
    property_summary = f"{ctx['area']:.0f} m\u00b2, {ctx['rooms_text']} y {ctx['bathrooms_text']}"

    amenity_feature_labels = {
        HAS_LIFT_COLUMN: "Ascensor",
        HAS_TERRACE_COLUMN: "Terraza",
        HAS_AIR_CONDITIONING_COLUMN: "Aire acondicionado",
        HAS_PARKING_COLUMN: "Garaje",
        HAS_BOXROOM_COLUMN: "Trastero",
        HAS_SWIMMING_POOL_COLUMN: "Piscina",
    }
    amenity_importance = feature_importance(package)
    if not amenity_importance.empty:
        amenity_importance = amenity_importance[amenity_importance["Variable"].isin(amenity_feature_labels)].copy()
        amenity_importance["Label"] = amenity_importance["Variable"].map(amenity_feature_labels)
        amenity_importance = amenity_importance.sort_values("Importance", ascending=False)
    top_amenity = None if amenity_importance.empty else amenity_importance.iloc[0]
    top_amenity_label = None if top_amenity is None else str(top_amenity["Label"])
    top_amenity_percent = 0 if top_amenity is None else round(float(top_amenity["Importance"]) * 100)

    previous_user_questions = [
        content for role, content in st.session_state.get("followup_messages", [])
        if role == "user"
    ]
    previous_intent = detect_vera_intent(previous_user_questions[-1]) if previous_user_questions else "GENERAL"
    followup_terms = {"y entonces", "entonces", "cual", "cu\u00e1l", "por que", "por qu\u00e9", "y de amenities", "amenities", "como asi", "c\u00f3mo as\u00ed", "eso", "ese"}
    is_followup = clean_question in followup_terms or topic_question in followup_terms or (len(clean_question) <= 22 and previous_user_questions)
    if is_followup and intent == "GENERAL":
        if "amenit" in topic_question:
            intent = "PROPERTY_IMPROVEMENTS"
        elif topic_question in {"cual", "cu\u00e1l"}:
            intent = "FEATURE_IMPORTANCE"
        else:
            intent = previous_intent

    previous_question = normalize_text(previous_user_questions[-1]).translate(str.maketrans("", "", "\u00bf?!.:,;")) if previous_user_questions else ""
    previous_topic_amenity = any(term in previous_question for term in ["amenity", "amenities", "ascensor", "garaje", "parking", "piscina", "terraza"])
    asks_top_amenity = "amenit" in topic_question and any(term in topic_question for term in ["influ", "peso", "pesa", "mas", "mayor"] )
    refers_to_ascensor = "ascensor" in topic_question
    asks_followup_why = topic_question in {"por que", "por qu\u00e9", "como asi", "c\u00f3mo as\u00ed"}

    if asks_top_amenity and top_amenity_label:
        return (
            f"Entre las amenities consideradas por el modelo, **{top_amenity_label.lower()}** es la que m\u00e1s peso tiene en esta valoraci\u00f3n "
            f"({top_amenity_percent} % de importancia global).\n\n"
            f"Aun as\u00ed, su impacto sigue siendo bastante menor que **{ctx['top_features_text']}**, que son los factores realmente determinantes para explicar el precio."
        )

    if previous_topic_amenity and refers_to_ascensor and top_amenity_label:
        if top_amenity_label == "Ascensor":
            return (
                "S\u00ed. Entre las amenities consideradas por el modelo, **el ascensor** es la que m\u00e1s peso tiene en esta valoraci\u00f3n.\n\n"
                f"Aun as\u00ed, conviene ponerlo en contexto: su impacto es menor que **{ctx['top_features_text']}**, que son los factores que m\u00e1s explican el precio estimado."
            )
        return (
            f"No exactamente. En esta valoraci\u00f3n, la amenity con m\u00e1s peso es **{top_amenity_label.lower()}**, no el ascensor.\n\n"
            f"El ascensor puede aportar valor comercial, pero queda por detr\u00e1s de los factores principales: **{ctx['top_features_text']}**."
        )

    if asks_followup_why and previous_topic_amenity and top_amenity_label:
        return (
            f"Porque **{top_amenity_label.lower()}** ayuda a diferenciar viviendas parecidas cuando el comprador compara comodidad, accesibilidad y equipamiento. "
            "Es un atributo f\u00e1cil de entender comercialmente y puede inclinar la decisi\u00f3n entre inmuebles similares.\n\n"
            f"Pero no es el motor principal del precio: en esta valoraci\u00f3n pesan m\u00e1s **{ctx['top_features_text']}**."
        )

    if intent == "LOCATION_EXPLANATION" and any(term in topic_question for term in ["cerca", "lejos", "centro"]):
        direction = "m\u00e1s cerca del centro" if "cerca" in topic_question else "m\u00e1s lejos del centro"
        expected = "reforzar\u00eda el atractivo de ubicaci\u00f3n" if "cerca" in topic_question else "podr\u00eda reducir parte del atractivo de ubicaci\u00f3n"
        return (
            f"Si estuviera **{direction}**, {expected}, porque el contexto de ubicaci\u00f3n ayuda a situar el barrio dentro de Madrid.\n\n"
            "No puedo convertir ese cambio en euros sin ejecutar una nueva valoraci\u00f3n, pero desde un punto de vista inmobiliario la cercan\u00eda al centro suele mejorar la lectura comercial frente a alternativas m\u00e1s perif\u00e9ricas."
        )

    if intent == "PRICE_EXPLANATION":
        return (
            f"En esta vivienda, el precio estimado de **{ctx['estimated_price']}** se entiende por la combinaci\u00f3n de "
            f"**{ctx['neighborhood']}**, la superficie y el equipamiento. No es una lectura de un \u00fanico factor: el barrio "
            f"marca una referencia de mercado de **{ctx['average_unit_price']}**, y la vivienda aporta {property_summary} "
            f"con {active_amenities_text}.\n\n"
            f"Lo que m\u00e1s pesa en la interpretaci\u00f3n es **{ctx['top_features_text']}**. Como asesor, usar\u00eda este valor "
            "como referencia para comparar inmuebles similares antes de tomar una decisi\u00f3n."
        )

    if intent == "FEATURE_IMPORTANCE":
        return (
            "Los factores que m\u00e1s explican esta valoraci\u00f3n son:\n\n"
            + "\n".join(f"- {factor}" for factor in importance_factors[:4])
            + "\n\nEn t\u00e9rminos inmobiliarios, primero miraría esos elementos antes que detalles secundarios: suelen ser los que mejor explican por qu\u00e9 una vivienda se sit\u00faa en un rango de precio u otro."
        )

    if intent == "LOCATION_EXPLANATION":
        return (
            f"S\u00ed, la ubicaci\u00f3n influye. En este caso, **{ctx['neighborhood']}** aporta el contexto de mercado: "
            f"precio medio del barrio, distancia al centro y distancia al eje de Castellana.\n\n"
            f"Para esta valoraci\u00f3n, la referencia del barrio es **{ctx['average_unit_price']}**. Esa informaci\u00f3n ayuda a situar la vivienda dentro de Madrid, pero el resultado final tambi\u00e9n depende de superficie, distribuci\u00f3n y amenities."
        )

    if intent == "HYPOTHETICAL_SCENARIO":
        changes = detect_hypothetical_changes(normalized_question)
        if not changes:
            changes = [("Caracter\u00edsticas", "GENERAL", "cambiar\u00edan la comparaci\u00f3n con viviendas similares")]
        lines = []
        for label, feature, explanation in changes:
            translated = translate_model_feature(feature)
            if feature == "REFORM":
                relevance = "La reforma no est\u00e1 medida directamente por el modelo, pero puede cambiar mucho la percepci\u00f3n comercial."
            else:
                relevance = feature_relevance(ctx, translated)
            lines.append(f"- **{label}:** {explanation}. {relevance}")
        return (
            "S\u00ed, ese cambio probablemente modificar\u00eda el atractivo comercial de la vivienda, pero no ser\u00eda profesional inventar un nuevo precio sin recalcular.\n\n"
            + "\n".join(lines)
            + "\n\nPara cuantificarlo, ejecutar\u00eda una nueva valoraci\u00f3n con esas caracter\u00edsticas incorporadas."
        )

    if intent == "PROPERTY_IMPROVEMENTS":
        if ctx["missing_amenities"]:
            return (
                f"Si tuviera que priorizar, revisar\u00eda primero las amenities ausentes: **{missing_amenities_text}**. "
                "No todas tienen el mismo efecto, pero son elementos que un comprador entiende r\u00e1pido y que pueden ayudar a diferenciar la vivienda.\n\n"
                f"Aun as\u00ed, en esta valoraci\u00f3n siguen pesando mucho **{ctx['top_features_text']}**. Mi recomendaci\u00f3n ser\u00eda probar una nueva valoraci\u00f3n solo con las mejoras que sean realistas."
            )
        return (
            "La vivienda ya incorpora las amenities principales del MVP. En ese caso, no centrar\u00eda el an\u00e1lisis en a\u00f1adir equipamiento, sino en compararla bien con viviendas similares del barrio."
        )

    if intent == "INVESTMENT":
        return (
            "No lo responder\u00eda con un s\u00ed o no cerrado. Como asesor, lo tratar\u00eda como una posible operaci\u00f3n que necesita contraste con comparables, estado real y margen de negociaci\u00f3n.\n\n"
            f"La referencia de partida es **{ctx['estimated_price']}** y **{ctx['estimated_unit_price']}** en **{ctx['neighborhood']}**. Si el inmueble real confirma buen estado y encaja con precios comparables, la valoraci\u00f3n puede servir como buena base para negociar."
        )

    if intent == "MODEL_LIMITATIONS":
        return (
            "La estimaci\u00f3n es \u00fatil como referencia, pero no sustituye una visita ni una tasaci\u00f3n oficial.\n\n"
            "VERA considera barrio, superficie, habitaciones, ba\u00f1os, amenities y mercado hist\u00f3rico. No puede ver estado real, orientaci\u00f3n, reformas, luminosidad, ruido, vistas ni capacidad de negociaci\u00f3n. Esos puntos conviene validarlos antes de decidir."
        )

    if intent == "NEIGHBOURHOOD":
        return (
            f"El barrio importa mucho: **{ctx['neighborhood']}** marca la referencia local con un precio medio de **{ctx['average_price']}** y **{ctx['average_unit_price']}**.\n\n"
            "Por eso no compararía esta vivienda con Madrid en abstracto, sino con alternativas similares dentro del mismo entorno o en barrios equivalentes."
        )

    if intent == "MODEL_EXPLANATION":
        return (
            "VERA usa el modelo como una herramienta de apoyo para leer la vivienda frente al mercado hist\u00f3rico de Madrid.\n\n"
            f"En esta valoraci\u00f3n combina el contexto de **{ctx['neighborhood']}**, el precio medio del barrio y las caracter\u00edsticas del inmueble. Yo lo usar\u00eda como una br\u00fajula profesional, no como una tasaci\u00f3n cerrada."
        )

    if is_followup:
        return (
            f"Siguiendo con la idea anterior, lo m\u00e1s importante aqu\u00ed es no perder de vista **{ctx['top_features_text']}**. "
            f"Para esta vivienda en **{ctx['neighborhood']}**, esos factores explican mejor el resultado que un detalle aislado.\n\n"
            "Si quieres, puedo centrarme en precio, ubicación, amenities o limitaciones del modelo."
        )

    return (
        f"Puedo ayudarte a interpretar la valoraci\u00f3n de **{ctx['estimated_price']}** desde un punto de vista inmobiliario. "
        "Para darte una respuesta m\u00e1s precisa, preg\u00fantame por precio, barrio, amenities, mejoras o limitaciones del modelo."
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


def format_yes_no(value: Any) -> str:
    """Format binary amenity values for presentation."""
    return "S\u00ed" if int(value) == 1 else "No"


def render_property_summary_card(context: dict[str, Any]) -> None:
    """Render a visual summary of the user-provided property characteristics."""
    data = st.session_state.get("valuation_data", {})
    neighborhood = context.get("neighborhood", data.get("neighborhood", "No informado"))
    area = f"{float(data.get(AREA_COLUMN, 0)):,.0f}".replace(",", ".") + " m\u00b2"
    rooms = str(int(data.get(ROOMS_COLUMN, 0)))

    left_items = [
        ("\U0001f3e2 Ascensor", format_yes_no(data.get(HAS_LIFT_COLUMN, 0))),
        ("\U0001f6bf Ba\u00f1os", str(int(data.get(BATHROOMS_COLUMN, 0)))),
    ]
    right_items = [
        ("\U0001f697 Garaje", format_yes_no(data.get(HAS_PARKING_COLUMN, 0))),
        ("\U0001f33f Terraza", format_yes_no(data.get(HAS_TERRACE_COLUMN, 0))),
        ("\U0001f3ca Piscina", format_yes_no(data.get(HAS_SWIMMING_POOL_COLUMN, 0))),
        ("\u2744 Aire acondicionado", format_yes_no(data.get(HAS_AIR_CONDITIONING_COLUMN, 0))),
        ("\U0001f4e6 Trastero", format_yes_no(data.get(HAS_BOXROOM_COLUMN, 0))),
    ]

    st.subheader("\U0001f4cb Resumen de la vivienda analizada")
    with st.container(border=True):
        kpi_neighborhood, kpi_area, kpi_rooms = st.columns(3)
        kpi_neighborhood.metric("\U0001f4cd Barrio", str(neighborhood))
        kpi_area.metric("\U0001f4d0 Superficie", area)
        kpi_rooms.metric("\U0001f6cf Habitaciones", rooms)

        left_col, right_col = st.columns(2)
        for column, items in ((left_col, left_items), (right_col, right_items)):
            with column:
                for label, value in items:
                    label_col, value_col = st.columns([1.4, 1])
                    label_col.write(label)
                    value_col.markdown(f"**{value}**")
        st.caption(
            "La valoraci\u00f3n se ha realizado considerando las caracter\u00edsticas anteriores y el comportamiento "
            "hist\u00f3rico del mercado inmobiliario en Madrid."
        )


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



def render_vera_valuation_recommendation(package: dict[str, Any], context: dict[str, Any]) -> None:
    """Render VERA's executive interpretation of the completed valuation."""
    ctx = vera_context(context, package)

    def interpret_feature(label: str, importance: float, position: int) -> str:
        percent = round(float(importance) * 100)
        prefix = f"**{label} ({percent}%)**"
        if label == "Superficie construida":
            return f"{prefix}: es el principal factor que explica la valoraci\u00f3n, porque define el tama\u00f1o real del producto inmobiliario."
        if label == "Precio medio del barrio":
            return f"{prefix}: sit\u00faa la vivienda dentro del contexto de mercado de {ctx['neighborhood']}."
        if label in {"Garaje", "Terraza", "Piscina", "Aire acondicionado", "Trastero", "Ascensor"}:
            return f"{prefix}: aporta lectura comercial adicional, aunque normalmente pesa menos que superficie y barrio."
        if position == 0:
            return f"{prefix}: es el factor con mayor influencia en esta estimaci\u00f3n."
        return f"{prefix}: ayuda a ajustar la comparaci\u00f3n frente a viviendas similares."

    interpreted_features = [
        interpret_feature(item["label"], item["importance"], index)
        for index, item in enumerate(ctx["top_feature_details"][:4])
    ]

    with st.container(border=True):
        st.subheader("\U0001f9e0 Recomendaci\u00f3n de VERA")

        st.markdown("## \U0001f3e0 Valoraci\u00f3n general")
        st.write(
            f"La estimaci\u00f3n obtenida encaja con una lectura de mercado para **{ctx['neighborhood']}**. "
            "VERA interpreta el resultado como una referencia profesional para comparar la vivienda con inmuebles similares, no como una tasaci\u00f3n oficial."
        )

        st.markdown("## \U0001f4ca Factores clave")
        if interpreted_features:
            for line in interpreted_features:
                st.markdown(f"- {line}")
        else:
            st.write("No hay informaci\u00f3n de importancia de variables disponible para interpretar el resultado.")

        st.markdown("## \u2705 Recomendaci\u00f3n de VERA")
        st.write(
            f"Si estuviera asesorando esta operaci\u00f3n, usar\u00eda la valoraci\u00f3n como punto de partida y revisar\u00eda comparables reales en **{ctx['neighborhood']}**. "
            "El criterio clave es confirmar si precio, superficie, distribuci\u00f3n y amenities mantienen coherencia con viviendas similares del entorno."
        )

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

    render_property_summary_card(context)

    st.subheader("Variables con mayor influencia (Feature Importance)")
    importance = feature_importance(package)
    importance = importance[importance["Variable"] != DISTANCE_TO_METRO_COLUMN]
    if importance.empty:
        st.warning("El modelo no expone feature_importances_.")
    else:
        display_importance = importance.copy()
        display_importance["Variable"] = display_importance["Variable"].map(translate_model_feature)
        display_importance = display_importance.groupby("Variable", as_index=False)["Importance"].sum()
        display_importance = display_importance.sort_values("Importance", ascending=False)
        st.bar_chart(display_importance.set_index("Variable").head(10), color="#78BE20")
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
    """Render VERA, the persistent post-valuation conversational advisor."""
    st.subheader("\U0001f916 VERA")
    st.caption("Valuation & Real Estate Advisor")

    initial_message = (
        "Hola, soy VERA, tu especialista en valoraci\u00f3n inmobiliaria.\n\n"
        "Ya tengo el contexto completo de la vivienda que acabas de valorar y puedo ayudarte a leer el resultado como lo har\u00eda un asesor.\n\n"
        "Puedes preguntarme, por ejemplo:\n\n"
        "\u2022 \U0001f4b6 \u00bfPor qu\u00e9 la vivienda tiene ese precio?\n"
        "\u2022 \U0001f4c8 \u00bfQu\u00e9 pasar\u00eda si tuviera piscina o garaje?\n"
        "\u2022 \U0001f4cd \u00bfC\u00f3mo influye la ubicaci\u00f3n?\n"
        "\u2022 \U0001f4ca \u00bfQu\u00e9 pesa m\u00e1s en esta valoraci\u00f3n?\n"
        "\u2022 \u2696\ufe0f \u00bfQu\u00e9 limitaciones tiene esta estimaci\u00f3n?"
    )
    if not st.session_state["followup_messages"]:
        st.session_state["followup_messages"].append(("assistant", initial_message))

    for role, content in st.session_state["followup_messages"]:
        with st.chat_message(role):
            if role == "assistant":
                st.markdown("**\U0001f916 VERA**")
                st.info(content)
            else:
                st.markdown("**\U0001f464 Usuario**")
                st.caption(content)

    question = st.chat_input("Pregunta a VERA")
    if question:
        answer = answer_valuation_question(question, package, dataset, context)
        st.session_state["followup_messages"].append(("user", question))
        st.session_state["followup_messages"].append(("assistant", answer))
        st.rerun()
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
    """Render compact horizontal bar charts for price comparison."""
    price_data = comparison[["Barrio", "Precio medio"]].copy()
    max_average_price = float(price_data["Precio medio"].max())
    price_data["Valor mostrado"] = price_data["Precio medio"].map(format_euros)
    price_data["Color"] = price_data["Precio medio"].apply(
        lambda value: "#5E9E1D" if float(value) == max_average_price else "#B7D95A"
    )
    price_bars = (
        alt.Chart(price_data)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X(
                "Precio medio:Q",
                title=None,
                axis=alt.Axis(labels=False, ticks=False, domain=False, grid=False),
                scale=alt.Scale(domain=[0, max_average_price * 1.22]),
            ),
            y=alt.Y(
                "Barrio:N",
                title=None,
                sort="-x",
                axis=alt.Axis(labelFontSize=12, labelColor="#222222", labelLimit=220, labelPadding=4, ticks=False, domain=False),
            ),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Barrio:N", "Valor mostrado:N"],
        )
        .properties(height=200)
    )
    price_labels = price_bars.mark_text(align="left", baseline="middle", dx=12, color="#222222", fontSize=12).encode(
        text="Valor mostrado:N"
    )

    unit_price_data = comparison[["Barrio", "EUR/m2 medio"]].copy()
    max_unit_price = float(unit_price_data["EUR/m2 medio"].max())
    unit_price_data["Valor mostrado"] = unit_price_data["EUR/m2 medio"].map(format_euros_per_m2)
    unit_price_data["Color"] = unit_price_data["EUR/m2 medio"].apply(
        lambda value: "#5E9E1D" if float(value) == max_unit_price else "#B7D95A"
    )
    unit_price_bars = (
        alt.Chart(unit_price_data)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X(
                "EUR/m2 medio:Q",
                title=None,
                axis=alt.Axis(labels=False, ticks=False, domain=False, grid=False),
                scale=alt.Scale(domain=[0, max_unit_price * 1.22]),
            ),
            y=alt.Y(
                "Barrio:N",
                title=None,
                sort="-x",
                axis=alt.Axis(labelFontSize=12, labelColor="#222222", labelLimit=220, labelPadding=4, ticks=False, domain=False),
            ),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Barrio:N", "Valor mostrado:N"],
        )
        .properties(height=200)
    )
    unit_price_labels = unit_price_bars.mark_text(align="left", baseline="middle", dx=12, color="#222222", fontSize=12).encode(
        text="Valor mostrado:N"
    )

    st.markdown("#### \U0001f4b6 Comparaci\u00f3n del precio medio")
    st.altair_chart(price_bars + price_labels, width="stretch")
    st.markdown("#### \U0001f3e0 Comparaci\u00f3n del precio por m\u00b2")
    st.altair_chart(unit_price_bars + unit_price_labels, width="stretch")


def render_executive_summary(comparison: pd.DataFrame) -> None:
    """Render VERA's executive conclusion as native Streamlit components."""
    summary = build_comparison_summary(comparison)
    with st.container(border=True):
        st.subheader("\U0001f9e0 Recomendaci\u00f3n de VERA")
        c1, c2 = st.columns(2)
        c1.metric("\U0001f4b6 Barrio m\u00e1s econ\u00f3mico", summary["Barrio m\u00e1s econ\u00f3mico"])
        c2.metric("\U0001f4c8 Mayor precio por m\u00b2", summary["Barrio con mayor precio por m\u00b2"])

        c3, c4 = st.columns(2)
        c3.metric("\U0001f687 Mejor comunicado", summary["Barrio mejor comunicado"])
        c4.metric("\u2728 M\u00e1s amenities", summary["Barrio con m\u00e1s amenities"])

        st.info(
            "VERA interpreta estos indicadores como una primera lectura de mercado: el precio medio ayuda a entender el esfuerzo de compra, "
            "el precio por m\u00b2 permite comparar intensidad de valor, la comunicaci\u00f3n orienta la comodidad diaria y las amenities muestran el nivel de equipamiento medio. "
            "Antes de analizar oportunidades de inversi\u00f3n, usar\u00eda esta comparaci\u00f3n para decidir qu\u00e9 barrio encaja mejor con el perfil del comprador."
        )

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

        st.caption("\u00bfPor qu\u00e9 lo recomendamos?")
        check = chr(0x2714)
        for reason in investment_reasons(row, priority):
            st.write(f"{check} {reason}")


def render_investment_score_chart(ranking: pd.DataFrame) -> None:
    """Render a polished horizontal bar chart with the top five opportunity scores."""
    top_five = ranking.head(5).copy()
    colors = ["#6CB33F", "#7BC653", "#8CCF66", "#A7DB86", "#C4E8A8"]
    top_five["Color"] = colors[: len(top_five)]
    top_five["Score label"] = top_five["Score simple de oportunidad"].map(lambda value: f"{float(value):.2f}")

    bars = (
        alt.Chart(top_five)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X(
                "Score simple de oportunidad:Q",
                title="Score de oportunidad",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(grid=False),
            ),
            y=alt.Y("Barrio:N", title=None, sort=list(top_five["Barrio"])),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Barrio:N", "Score simple de oportunidad:Q"],
        )
    )
    labels = (
        alt.Chart(top_five)
        .mark_text(align="left", baseline="middle", dx=8, color="#222222", fontWeight="bold")
        .encode(
            x="Score simple de oportunidad:Q",
            y=alt.Y("Barrio:N", title=None, sort=list(top_five["Barrio"])),
            text="Score label:N",
        )
    )
    chart = (bars + labels).properties(height=260).configure_view(strokeWidth=0)
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
    """Render VERA's final interpretation of the investment ranking."""
    top = ranking.iloc[0]
    investment_data = st.session_state.get("investment_data", {})
    budget = investment_data.get("max_budget")
    budget_text = format_euros(float(budget)) if budget is not None else "el presupuesto indicado"
    score = float(top["Score simple de oportunidad"])
    alternatives = ranking.head(3)["Barrio"].astype(str).tolist()

    with st.container(border=True):
        st.subheader("\U0001f9e0 Recomendaci\u00f3n de VERA")
        st.markdown("**Barrio recomendado**")
        st.metric(str(top["Barrio"]), f"Score {score:.2f}")

        st.markdown("**Interpretaci\u00f3n**")
        st.write(
            f"Para un presupuesto de **{budget_text}** y una prioridad de **{priority}**, **{top['Barrio']}** aparece como la zona m\u00e1s equilibrada del ranking. "
            f"Combina un precio medio de **{format_euros(float(top['Precio_medio']))}**, **{format_euros_per_m2(float(top['EUR_m2_medio']))}**, "
            f"amenities medias del **{float(top['Amenities']):.1f} %**, una distancia media al metro de **{float(top['Distancia_metro']):.2f}** y un score de **{score:.2f}**. "
            "La lectura no es que sea la opci\u00f3n m\u00e1s barata, sino la que mejor balance ofrece dentro de los criterios introducidos."
        )

        st.markdown("**Recomendaci\u00f3n**")
        if len(alternatives) >= 3:
            st.write(
                f"Usar\u00eda este resultado para empezar la b\u00fasqueda en **{top['Barrio']}** y contrastarla con **{alternatives[1]}** y **{alternatives[2]}**. "
                "La decisi\u00f3n final deber\u00eda salir de comparar viviendas concretas, no solo de elegir el primer barrio del ranking."
            )
        else:
            st.write(
                f"Usar\u00eda este resultado para empezar la b\u00fasqueda en **{top['Barrio']}** y validar con viviendas concretas si encaja con el comprador."
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
        st.title("\U0001f4ca Modelo y m\u00e9tricas")

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


def render_footer() -> None:
    """Render the final academic project footer once."""
    st.divider()
    st.caption(
        "Developed by: Pablo Gonz\u00e1lez Dom\u00ednguez \u00b7 Pablo Jos\u00e9 Lozano Garnacho \u00b7 "
        "Clara Mafalda Pedrajas V\u00e1zquez \u00b7 Flavia Ram\u00edrez Plantamor"
    )
    st.caption(
        "TFM | Master in Data Science & Machine Learning with Artificial Intelligence | "
        "The Valley Business & Tech School | Academic Year 2025\u20132026"
    )

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

    render_footer()


if __name__ == "__main__":
    main()
