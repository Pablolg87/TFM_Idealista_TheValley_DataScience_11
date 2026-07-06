"""Conversational router agent for the Smart Advisor MVP.

The agent interprets simple natural-language requests without generative AI,
external APIs, or LangChain. It extracts basic entities, routes the request to
predict.py or advisor.py, and always returns a structured dictionary.
"""

import re
import unicodedata
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import advisor
import config
import predict


INTENT_VALUATION = "valuation"
INTENT_NEIGHBORHOOD_ANALYSIS = "neighborhood_analysis"
INTENT_NEIGHBORHOOD_COMPARISON = "neighborhood_comparison"
INTENT_BUDGET_RECOMMENDATION = "budget_recommendation"
INTENT_CONTEXTUAL_ADVICE = "contextual_advice"
INTENT_DEMO_PROPERTY = "demo_property"
INTENT_UNKNOWN = "unknown"

INTENT_ALIASES = {
    "neighborhood_stats": INTENT_NEIGHBORHOOD_ANALYSIS,
    "property_comparison": "property_comparison",
    "comparison": INTENT_NEIGHBORHOOD_COMPARISON,
    "recommendation": INTENT_BUDGET_RECOMMENDATION,
}


def _normalize_text(value: str) -> str:
    """Normalize text for simple rule-based interpretation."""
    return value.strip().casefold()


def _normalize_for_matching(value: str) -> str:
    """Normalize text for accent-insensitive matching."""
    normalized = unicodedata.normalize("NFKD", str(value).casefold().strip())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _get_request_text(request: str | Mapping[str, Any]) -> str:
    """Extract text from a string request or from common dictionary fields."""
    if isinstance(request, str):
        return request

    for key in ("message", "query", "text", "prompt"):
        value = request.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def _canonical_intent(intent: str) -> str:
    """Return the internal canonical name for an intent."""
    normalized_intent = _normalize_text(intent)
    return INTENT_ALIASES.get(normalized_intent, normalized_intent)


def detect_intent(request: str | Mapping[str, Any]) -> str:
    """Detect supported intents from explicit values or keywords."""
    if isinstance(request, Mapping):
        explicit_intent = request.get("intent")
        if isinstance(explicit_intent, str) and explicit_intent.strip():
            return _canonical_intent(explicit_intent)

    text = _normalize_text(_get_request_text(request))

    if any(keyword in text for keyword in ("por que", "por qué", "influye", "influencia", "este precio", "barrio caro", "caro", "comunicado", "comunicacion", "comunicación", "piscina")):
        return INTENT_CONTEXTUAL_ADVICE

    if any(
        keyword in text
        for keyword in (
            "presupuesto",
            "recomienda",
            "recomendar",
            "puedo comprar",
            "barrios puedo",
            "tengo",
        )
    ):
        return INTENT_BUDGET_RECOMMENDATION

    if any(keyword in text for keyword in ("comparar barrios", "compara", "comparame")):
        return INTENT_NEIGHBORHOOD_COMPARISON

    if any(keyword in text for keyword in ("valora", "valorar", "valoracion", "tasar", "estima", "precio")):
        return INTENT_VALUATION

    if any(keyword in text for keyword in ("analiza", "analizar", "barrio", "zona")):
        return INTENT_NEIGHBORHOOD_ANALYSIS

    return INTENT_UNKNOWN


@lru_cache(maxsize=1)
def _known_neighborhoods() -> tuple[str, ...]:
    """Return known neighborhood names from the dataset."""
    dataset = advisor.load_dataset()
    neighborhoods = (
        dataset[advisor.NEIGHBORHOOD_COLUMN]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    return tuple(sorted(neighborhoods, key=len, reverse=True))


def _extract_neighborhoods(text: str) -> list[str]:
    """Extract known neighborhood names mentioned in text."""
    normalized_text = _normalize_for_matching(text)
    neighborhoods: list[str] = []

    for neighborhood in _known_neighborhoods():
        normalized_neighborhood = _normalize_for_matching(neighborhood)
        pattern = rf"(?<!\w){re.escape(normalized_neighborhood)}(?!\w)"
        if re.search(pattern, normalized_text):
            neighborhoods.append(neighborhood)

    return neighborhoods


def _extract_neighborhood_query(text: str) -> str | None:
    """Extract a raw neighborhood phrase when it is not an exact dataset value."""
    patterns = [
        r"(?:barrio|zona)\s+(?:de\s+)?([a-zA-ZÀ-ÿ0-9 '\-]+)",
        r"analiza(?:r)?\s+(?:el\s+|la\s+)?(?:barrio\s+|zona\s+)?(?:de\s+)?([a-zA-ZÀ-ÿ0-9 '\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" .,:;¿?")
            if candidate:
                return candidate
    return None


def _parse_number(value: str) -> float:
    """Parse Spanish-style numeric text into a float."""
    cleaned_value = value.strip().lower().replace(" ", "")
    multiplier = 1_000 if cleaned_value.endswith("k") else 1
    cleaned_value = cleaned_value.rstrip("k").replace(".", "").replace(",", ".")
    return float(cleaned_value) * multiplier


def _extract_first_number(patterns: list[str], text: str) -> float | None:
    """Return the first numeric value matching any regex pattern."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _parse_number(match.group(1))

    return None


def _extract_boolean_feature(text: str, keywords: tuple[str, ...]) -> int:
    """Extract simple yes/no equipment mentions from text."""
    normalized_text = _normalize_text(text)

    for keyword in keywords:
        if f"sin {keyword}" in normalized_text:
            return 0

    for keyword in keywords:
        if f"con {keyword}" in normalized_text or keyword in normalized_text:
            return 1

    return 0


def extract_entities(request: str | Mapping[str, Any]) -> dict[str, Any]:
    """Extract basic real estate entities from text or structured input."""
    text = _get_request_text(request)
    entities: dict[str, Any] = {}

    if isinstance(request, Mapping):
        entities.update(
            {
                key: value
                for key, value in request.items()
                if key not in {"intent", "message", "query", "text", "prompt"}
            }
        )

    neighborhoods = entities.get("neighborhoods")
    if not neighborhoods:
        neighborhoods = _extract_neighborhoods(text)
    if neighborhoods:
        entities["neighborhoods"] = list(neighborhoods)
        entities.setdefault("neighborhood", list(neighborhoods)[0])
    elif text:
        neighborhood_query = _extract_neighborhood_query(text)
        if neighborhood_query:
            entities["neighborhood_query"] = neighborhood_query

    budget = entities.get("budget")
    if budget is None:
        budget = _extract_first_number(
            [
                r"(?:presupuesto|hasta|maximo|maxima|max)\s*(?:de)?\s*(\d[\d.,]*k?)",
                r"(\d[\d.,]*k?)\s*(?:eur|euro|euros)",
            ],
            text,
        )
    if budget is not None:
        entities["budget"] = float(budget)

    area = entities.get(config.AREA_COLUMN)
    if area is None:
        area = _extract_first_number(
            [r"(\d[\d.,]*)\s*(?:m2|m²|metros)"],
            text,
        )
    if area is not None:
        entities[config.AREA_COLUMN] = float(area)

    rooms = entities.get(config.ROOMS_COLUMN)
    if rooms is None:
        rooms = _extract_first_number(
            [r"(\d+)\s*(?:habitacion|habitaciones|dormitorio|dormitorios)"],
            text,
        )
    if rooms is not None:
        entities[config.ROOMS_COLUMN] = int(rooms)

    bathrooms = entities.get(config.BATHROOMS_COLUMN)
    if bathrooms is None:
        bathrooms = _extract_first_number(
            [r"(\d+)\s*(?:bano|banos|baño|baños)"],
            text,
        )
    if bathrooms is not None:
        entities[config.BATHROOMS_COLUMN] = int(bathrooms)

    entities.setdefault(config.HAS_LIFT_COLUMN, _extract_boolean_feature(text, ("ascensor",)))
    entities.setdefault(config.HAS_TERRACE_COLUMN, _extract_boolean_feature(text, ("terraza",)))
    entities.setdefault(config.HAS_PARKING_COLUMN, _extract_boolean_feature(text, ("parking", "garaje")))
    entities.setdefault(config.HAS_AIR_CONDITIONING_COLUMN, _extract_boolean_feature(text, ("aire acondicionado", "aire")))
    entities.setdefault(config.HAS_BOXROOM_COLUMN, _extract_boolean_feature(text, ("trastero",)))
    entities.setdefault(config.HAS_SWIMMING_POOL_COLUMN, _extract_boolean_feature(text, ("piscina",)))

    return entities


def _success_response(
    intent: str,
    data: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    """Build a successful structured response."""
    return {
        "success": True,
        "intent": intent,
        "data": data,
        "message": message,
        "error": None,
    }


def _error_response(intent: str, error: str, message: str | None = None) -> dict[str, Any]:
    """Build an error structured response."""
    return {
        "success": False,
        "intent": intent,
        "data": {},
        "message": message or "No he podido completar la solicitud.",
        "error": error,
    }


def _require_fields(entities: Mapping[str, Any], fields: list[str]) -> None:
    """Validate that all required entity fields are present."""
    missing_fields = [
        field
        for field in fields
        if field not in entities or entities[field] in (None, "")
    ]

    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Faltan datos para completar la solicitud: {missing}")


def _route_valuation(entities: Mapping[str, Any]) -> dict[str, Any]:
    """Route a valuation request to predict.py."""
    location_name = entities.get(config.NEIGHBORHOOD_COLUMN, entities.get("neighborhood"))
    payload = {
        config.NEIGHBORHOOD_COLUMN: location_name,
        config.AREA_COLUMN: entities.get(config.AREA_COLUMN),
        config.ROOMS_COLUMN: entities.get(config.ROOMS_COLUMN),
        config.BATHROOMS_COLUMN: entities.get(config.BATHROOMS_COLUMN),
        config.HAS_TERRACE_COLUMN: int(entities.get(config.HAS_TERRACE_COLUMN, 0)),
        config.HAS_LIFT_COLUMN: int(entities.get(config.HAS_LIFT_COLUMN, 0)),
        config.HAS_AIR_CONDITIONING_COLUMN: int(entities.get(config.HAS_AIR_CONDITIONING_COLUMN, 0)),
        config.HAS_PARKING_COLUMN: int(entities.get(config.HAS_PARKING_COLUMN, 0)),
        config.HAS_BOXROOM_COLUMN: int(entities.get(config.HAS_BOXROOM_COLUMN, 0)),
        config.HAS_SWIMMING_POOL_COLUMN: int(entities.get(config.HAS_SWIMMING_POOL_COLUMN, 0)),
    }
    _require_fields(
        payload,
        [config.NEIGHBORHOOD_COLUMN, config.AREA_COLUMN, config.ROOMS_COLUMN, config.BATHROOMS_COLUMN],
    )
    prediction = predict.predict_price(payload)
    neighborhood = str(payload[config.NEIGHBORHOOD_COLUMN])

    return {
        "estimated_price": float(prediction),
        "input": payload,
        "neighborhood_intelligence": advisor.get_neighborhood_intelligence(neighborhood),
    }


def _route_neighborhood_analysis(entities: Mapping[str, Any]) -> dict[str, Any]:
    """Route a neighborhood analysis request to advisor.py."""
    neighborhood = entities.get("neighborhood")
    if not neighborhood:
        query = str(entities.get("neighborhood_query", "")).strip()
        suggestions = advisor.suggest_neighborhoods(query) if query else []
        if suggestions:
            available = ", ".join(suggestions)
            raise ValueError(
                f"No encuentro '{query}' como barrio exacto en el dataset. "
                f"Puedes probar con: {available}."
            )
        raise ValueError(
            "No he podido identificar el barrio. Prueba con un nombre disponible como Palacio, Sol o Goya."
        )

    stats = advisor.get_neighborhood_stats(neighborhood=str(neighborhood))
    stats["neighborhood_intelligence"] = advisor.get_neighborhood_intelligence(
        neighborhood=str(neighborhood)
    )
    return stats


def _route_neighborhood_comparison(entities: Mapping[str, Any]) -> dict[str, Any]:
    """Route a neighborhood comparison request to advisor.py."""
    neighborhoods = list(entities.get("neighborhoods", []))
    if len(neighborhoods) < 2:
        raise ValueError("Necesito dos barrios para hacer la comparacion.")

    return advisor.compare_neighborhoods(
        first_neighborhood=neighborhoods[0],
        second_neighborhood=neighborhoods[1],
    )


def _route_budget_recommendation(entities: Mapping[str, Any]) -> dict[str, Any]:
    """Route a budget recommendation request to advisor.py."""
    budget = entities.get("budget")
    _require_fields({"budget": budget}, ["budget"])

    return advisor.recommend_neighborhoods_by_budget(budget=float(budget))


def _route_property_comparison(entities: Mapping[str, Any]) -> dict[str, Any]:
    """Keep compatibility with structured property comparison requests."""
    neighborhood = entities.get("neighborhood")
    property_price = entities.get("property_price", entities.get("price"))
    _require_fields(
        {"neighborhood": neighborhood, "property_price": property_price},
        ["neighborhood", "property_price"],
    )

    comparison = advisor.compare_property_with_neighborhood(
        neighborhood=str(neighborhood),
        property_price=float(property_price),
    )
    comparison["neighborhood_intelligence"] = advisor.get_neighborhood_intelligence(
        neighborhood=str(neighborhood)
    )
    return comparison


def _route_demo_property() -> dict[str, Any]:
    """Load a real dataset property and value it with the active model."""
    demo = advisor.get_demo_property()
    valuation = _route_valuation(demo["input"])
    return {
        **demo,
        "estimated_price": valuation["estimated_price"],
        "neighborhood_intelligence": valuation["neighborhood_intelligence"],
    }


def _route_contextual_advice(
    text: str,
    last_valuation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Answer simple follow-up questions using the latest valuation context."""
    if not last_valuation:
        raise ValueError("Primero necesito una valoracion para poder contextualizar la respuesta.")

    data = last_valuation.get("data", last_valuation)
    input_data = data.get("input", {})
    intelligence = data.get("neighborhood_intelligence", {})
    estimated_price = float(data.get("estimated_price", 0) or 0)
    area = float(input_data.get(config.AREA_COLUMN, 0) or 0)
    unit_price = estimated_price / area if area > 0 else 0
    normalized_text = _normalize_text(text)

    response_parts = [
        f"La ultima valoracion estima la vivienda en {estimated_price:,.0f} euros",
        f"equivalente a {unit_price:,.0f} euros/m2" if unit_price else "",
    ]

    if "caro" in normalized_text:
        response_parts.append(
            "El barrio tiene un nivel relativo de precio "
            f"{intelligence.get('relative_price_level', 'no disponible')} "
            f"frente a Madrid ({float(intelligence.get('relative_price_percent', 0) or 0):.2f}%)."
        )
    elif "comunic" in normalized_text:
        response_parts.append(
            "La conectividad se aproxima con las distancias medias del dataset: "
            f"centro {float(intelligence.get('distance_to_city_center', 0) or 0):.2f}, "
            f"Castellana {float(intelligence.get('distance_to_castellana', 0) or 0):.2f}, "
            f"metro {float(intelligence.get('distance_to_metro', 0) or 0):.2f}."
        )
    elif "piscina" in normalized_text:
        if int(input_data.get(config.HAS_SWIMMING_POOL_COLUMN, 0)) == 1:
            response_parts.append("La valoracion ya incluye piscina como caracteristica de la vivienda.")
        else:
            simulated_input = dict(input_data)
            simulated_input[config.HAS_SWIMMING_POOL_COLUMN] = 1
            simulated_price = predict.predict_price(simulated_input)
            response_parts.append(
                "Si la vivienda tuviera piscina, el modelo estimaria aproximadamente "
                f"{simulated_price:,.0f} euros, una diferencia de {simulated_price - estimated_price:,.0f} euros."
            )
    else:
        response_parts.append(
            "El precio se explica por la superficie, habitaciones, banos, amenities "
            "y por el contexto del barrio calculado desde el dataset: precio medio, "
            "distancias y precio medio por m2 de la zona."
        )

    return {
        "estimated_price": estimated_price,
        "input": input_data,
        "neighborhood_intelligence": intelligence,
        "answer": " ".join(part for part in response_parts if part),
    }


def handle_request(
    request: str | Mapping[str, Any],
    last_valuation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Identify intent, extract entities, delegate work, and return one response."""
    intent = detect_intent(request)
    entities = extract_entities(request)
    response_intent = intent
    if isinstance(request, Mapping):
        explicit_intent = request.get("intent")
        if explicit_intent in {"neighborhood_stats", "property_comparison"}:
            response_intent = str(explicit_intent)

    try:
        if intent == INTENT_VALUATION:
            data = _route_valuation(entities)
            message = "He estimado el precio de la vivienda."
        elif intent == INTENT_NEIGHBORHOOD_ANALYSIS:
            data = _route_neighborhood_analysis(entities)
            message = "He analizado el barrio solicitado."
        elif intent == INTENT_NEIGHBORHOOD_COMPARISON:
            data = _route_neighborhood_comparison(entities)
            message = "He comparado los barrios solicitados."
        elif intent == INTENT_BUDGET_RECOMMENDATION:
            data = _route_budget_recommendation(entities)
            message = "He buscado barrios compatibles con el presupuesto."
        elif intent == "property_comparison":
            data = _route_property_comparison(entities)
            message = "He comparado la vivienda con la media del barrio."
        elif intent == INTENT_DEMO_PROPERTY:
            data = _route_demo_property()
            message = "He cargado una vivienda real del dataset y generado su valoracion."
        elif intent == INTENT_CONTEXTUAL_ADVICE:
            data = _route_contextual_advice(_get_request_text(request), last_valuation)
            message = data["answer"]
        else:
            raise ValueError("No he reconocido la intencion de la solicitud.")
    except Exception as exc:
        error_text = str(exc)
        friendly_message = error_text if "No encuentro" in error_text or "No he podido identificar" in error_text else None
        return _error_response(intent=response_intent, error=error_text, message=friendly_message)

    return _success_response(intent=response_intent, data=data, message=message)


class SmartAdvisorAgent:
    """Stable facade for the Smart Advisor routing layer."""

    def __init__(self) -> None:
        """Initialize the agent with optional valuation memory."""
        self.last_valuation_response: dict[str, Any] | None = None

    def process_request(self, request: str | Mapping[str, Any]) -> dict[str, Any]:
        """Process a structured or natural-language request."""
        response = handle_request(request, self.last_valuation_response)
        if response.get("success") and response.get("intent") in {INTENT_VALUATION, INTENT_DEMO_PROPERTY}:
            if response.get("intent") == INTENT_DEMO_PROPERTY:
                valuation_data = {
                    "estimated_price": response["data"].get("estimated_price"),
                    "input": response["data"].get("input", {}),
                    "neighborhood_intelligence": response["data"].get("neighborhood_intelligence", {}),
                }
                self.last_valuation_response = _success_response(
                    INTENT_VALUATION,
                    valuation_data,
                    "He estimado el precio de la vivienda.",
                )
            else:
                self.last_valuation_response = response
        return response

    def process_message(self, message: str) -> dict[str, Any]:
        """Process a natural-language message."""
        return self.process_request(message)
