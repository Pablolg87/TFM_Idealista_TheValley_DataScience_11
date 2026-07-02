"""Conversational router agent for the Smart Advisor MVP.

The agent interprets simple natural-language requests without generative AI,
external APIs, or LangChain. It extracts basic entities, routes the request to
predict.py or advisor.py, and always returns a structured dictionary.
"""

import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import advisor
import predict


INTENT_VALUATION = "valuation"
INTENT_NEIGHBORHOOD_ANALYSIS = "neighborhood_analysis"
INTENT_NEIGHBORHOOD_COMPARISON = "neighborhood_comparison"
INTENT_BUDGET_RECOMMENDATION = "budget_recommendation"
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
    normalized_text = _normalize_text(text)
    neighborhoods: list[str] = []

    for neighborhood in _known_neighborhoods():
        normalized_neighborhood = _normalize_text(neighborhood)
        pattern = rf"(?<!\w){re.escape(normalized_neighborhood)}(?!\w)"
        if re.search(pattern, normalized_text):
            neighborhoods.append(neighborhood)

    return neighborhoods


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


def _extract_boolean_feature(text: str, keyword: str) -> int:
    """Extract simple yes/no equipment mentions from text."""
    normalized_text = _normalize_text(text)

    if f"sin {keyword}" in normalized_text:
        return 0

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

    area = entities.get("CONSTRUCTEDAREA")
    if area is None:
        area = _extract_first_number(
            [r"(\d[\d.,]*)\s*(?:m2|m²|metros)"],
            text,
        )
    if area is not None:
        entities["CONSTRUCTEDAREA"] = float(area)

    rooms = entities.get("ROOMNUMBER")
    if rooms is None:
        rooms = _extract_first_number(
            [r"(\d+)\s*(?:habitacion|habitaciones|dormitorio|dormitorios)"],
            text,
        )
    if rooms is not None:
        entities["ROOMNUMBER"] = int(rooms)

    bathrooms = entities.get("BATHNUMBER")
    if bathrooms is None:
        bathrooms = _extract_first_number(
            [r"(\d+)\s*(?:bano|banos|baño|baños)"],
            text,
        )
    if bathrooms is not None:
        entities["BATHNUMBER"] = int(bathrooms)

    entities.setdefault("HASLIFT", _extract_boolean_feature(text, "ascensor"))
    entities.setdefault("HASTERRACE", _extract_boolean_feature(text, "terraza"))
    entities.setdefault("HASPARKINGSPACE", _extract_boolean_feature(text, "parking"))
    normalized_text = _normalize_text(text)
    if "sin garaje" in normalized_text:
        entities["HASPARKINGSPACE"] = 0
    elif "garaje" in normalized_text:
        entities["HASPARKINGSPACE"] = 1

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
    if "LOCATIONNAME" not in entities and "neighborhood" in entities:
        location_name = entities["neighborhood"]
    else:
        location_name = entities.get("LOCATIONNAME")

    payload = {
        "LOCATIONNAME": location_name,
        "CONSTRUCTEDAREA": entities.get("CONSTRUCTEDAREA"),
        "ROOMNUMBER": entities.get("ROOMNUMBER"),
        "BATHNUMBER": entities.get("BATHNUMBER"),
        "HASLIFT": int(entities.get("HASLIFT", 0)),
        "HASTERRACE": int(entities.get("HASTERRACE", 0)),
        "HASPARKINGSPACE": int(entities.get("HASPARKINGSPACE", 0)),
    }
    _require_fields(
        payload,
        ["LOCATIONNAME", "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER"],
    )
    prediction = predict.predict_price(payload)

    return {
        "estimated_price": float(prediction),
        "input": payload,
    }


def _route_neighborhood_analysis(entities: Mapping[str, Any]) -> dict[str, Any]:
    """Route a neighborhood analysis request to advisor.py."""
    neighborhood = entities.get("neighborhood")
    _require_fields({"neighborhood": neighborhood}, ["neighborhood"])

    return advisor.get_neighborhood_stats(neighborhood=str(neighborhood))


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

    return advisor.compare_property_with_neighborhood(
        neighborhood=str(neighborhood),
        property_price=float(property_price),
    )


def handle_request(request: str | Mapping[str, Any]) -> dict[str, Any]:
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
        else:
            raise ValueError("No he reconocido la intencion de la solicitud.")
    except Exception as exc:
        return _error_response(intent=response_intent, error=str(exc))

    return _success_response(intent=response_intent, data=data, message=message)


class SmartAdvisorAgent:
    """Stable facade for the Smart Advisor routing layer."""

    def process_request(self, request: str | Mapping[str, Any]) -> dict[str, Any]:
        """Process a structured or natural-language request."""
        return handle_request(request)

    def process_message(self, message: str) -> dict[str, Any]:
        """Process a natural-language message."""
        return handle_request(message)
