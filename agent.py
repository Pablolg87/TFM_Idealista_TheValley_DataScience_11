"""Simple router agent for the Smart Advisor MVP.

The agent does not use generative AI, external APIs, LangChain, or Streamlit.
It identifies a request intent, delegates the work to the appropriate module,
and returns a single structured dictionary.
"""

from collections.abc import Mapping
from typing import Any

import advisor
import predict


INTENT_VALUATION = "valuation"
INTENT_NEIGHBORHOOD_STATS = "neighborhood_stats"
INTENT_PROPERTY_COMPARISON = "property_comparison"
INTENT_UNKNOWN = "unknown"


def _normalize_text(value: str) -> str:
    """Normalize text for simple rule-based intent detection."""
    return value.strip().casefold()


def _get_request_text(request: str | Mapping[str, Any]) -> str:
    """Extract text from a string request or from common dictionary fields."""
    if isinstance(request, str):
        return request

    for key in ("intent", "message", "query", "text"):
        value = request.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


def detect_intent(request: str | Mapping[str, Any]) -> str:
    """Detect the request intent using explicit values or simple keywords."""
    if isinstance(request, Mapping):
        explicit_intent = request.get("intent")
        if isinstance(explicit_intent, str) and explicit_intent.strip():
            return _normalize_text(explicit_intent)

    text = _normalize_text(_get_request_text(request))

    if any(keyword in text for keyword in ("valorar", "valoracion", "precio")):
        return INTENT_VALUATION

    if any(keyword in text for keyword in ("comparar vivienda", "comparacion")):
        return INTENT_PROPERTY_COMPARISON

    if any(keyword in text for keyword in ("barrio", "zona", "neighborhood")):
        return INTENT_NEIGHBORHOOD_STATS

    return INTENT_UNKNOWN


def _success_response(intent: str, data: dict[str, Any]) -> dict[str, Any]:
    """Build a successful structured response."""
    return {
        "success": True,
        "intent": intent,
        "data": data,
        "error": None,
    }


def _error_response(intent: str, error: str) -> dict[str, Any]:
    """Build an error structured response."""
    return {
        "success": False,
        "intent": intent,
        "data": None,
        "error": error,
    }


def _require_mapping(request: str | Mapping[str, Any], intent: str) -> Mapping[str, Any]:
    """Validate that an intent has structured input data."""
    if not isinstance(request, Mapping):
        raise ValueError(
            f"Intent '{intent}' requires a dictionary with input data."
        )

    return request


def _route_valuation(request: str | Mapping[str, Any]) -> dict[str, Any]:
    """Route a property valuation request to predict.py."""
    payload = _require_mapping(request, INTENT_VALUATION)
    prediction = predict.predict_price(payload)

    return {
        "estimated_price": float(prediction),
        "input": dict(payload),
    }


def _route_neighborhood_stats(request: str | Mapping[str, Any]) -> dict[str, Any]:
    """Route a neighborhood analysis request to advisor.py."""
    payload = _require_mapping(request, INTENT_NEIGHBORHOOD_STATS)
    neighborhood = payload.get("neighborhood")

    if not isinstance(neighborhood, str) or not neighborhood.strip():
        raise ValueError("Neighborhood analysis requires 'neighborhood'.")

    return advisor.get_neighborhood_stats(neighborhood=neighborhood)


def _route_property_comparison(request: str | Mapping[str, Any]) -> dict[str, Any]:
    """Route a property comparison request to advisor.py."""
    payload = _require_mapping(request, INTENT_PROPERTY_COMPARISON)
    neighborhood = payload.get("neighborhood")
    property_price = payload.get("property_price", payload.get("price"))

    if not isinstance(neighborhood, str) or not neighborhood.strip():
        raise ValueError("Property comparison requires 'neighborhood'.")

    if property_price is None:
        raise ValueError("Property comparison requires 'property_price'.")

    return advisor.compare_property_with_neighborhood(
        neighborhood=neighborhood,
        property_price=float(property_price),
    )


def handle_request(request: str | Mapping[str, Any]) -> dict[str, Any]:
    """Identify intent, delegate to the right module, and return one response."""
    intent = detect_intent(request)

    try:
        if intent == INTENT_VALUATION:
            data = _route_valuation(request)
        elif intent == INTENT_NEIGHBORHOOD_STATS:
            data = _route_neighborhood_stats(request)
        elif intent == INTENT_PROPERTY_COMPARISON:
            data = _route_property_comparison(request)
        else:
            raise ValueError("Request intent was not recognized.")
    except Exception as exc:
        return _error_response(intent=intent, error=str(exc))

    return _success_response(intent=intent, data=data)


class SmartAdvisorAgent:
    """Stable facade for the Smart Advisor routing layer."""

    def process_request(self, request: str | Mapping[str, Any]) -> dict[str, Any]:
        """Process a request and return a structured dictionary."""
        return handle_request(request)

    def process_message(self, message: str) -> dict[str, Any]:
        """Backward-compatible entry point for text messages."""
        return handle_request(message)
