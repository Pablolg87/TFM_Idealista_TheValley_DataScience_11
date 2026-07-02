"""Streamlit interface for the Smart Advisor MVP."""

from typing import Any

import streamlit as st

from agent import SmartAdvisorAgent


APP_TITLE = "Idealista Smart Advisor"
FEATURE_VALUATION = "Valorar una vivienda"
FEATURE_COMPARISON = "Comparar una vivienda con el barrio"
FEATURE_NEIGHBORHOOD = "Analizar un barrio"


def get_agent() -> SmartAdvisorAgent:
    """Create or reuse the Smart Advisor agent."""
    if "agent" not in st.session_state:
        st.session_state.agent = SmartAdvisorAgent()

    return st.session_state.agent


def format_currency(value: float) -> str:
    """Format numeric values as euros for display."""
    return f"{value:,.0f} EUR".replace(",", ".")


def format_percent(value: float) -> str:
    """Format percentage values for display."""
    return f"{value:.2f}%"


def show_error(message: str | None) -> None:
    """Show a user-friendly backend error."""
    if not message:
        message = "No se ha podido completar la consulta."

    if "model" in message.lower() or "predict" in message.lower():
        st.warning(
            "El modelo de valoracion todavia no esta disponible. "
            "La interfaz ya esta preparada para integrarlo cuando existan "
            "los artefactos definitivos."
        )
        return

    st.error(message)


def show_neighborhood_stats(data: dict[str, Any]) -> None:
    """Render neighborhood statistics returned by the agent."""
    st.subheader(f"Analisis de {data['neighborhood']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Viviendas analizadas", f"{data['property_count']:,}".replace(",", "."))
    col2.metric("Precio medio", format_currency(data["average_price"]))
    col3.metric("Precio mediano", format_currency(data["median_price"]))

    col4, col5, col6 = st.columns(3)
    col4.metric("Precio medio por m2", format_currency(data["average_unit_price"]))
    col5.metric("Superficie media", f"{data['average_area']:.2f} m2")
    col6.metric("Rango de precios", f"{format_currency(data['min_price'])} - {format_currency(data['max_price'])}")


def show_property_comparison(data: dict[str, Any]) -> None:
    """Render property comparison results returned by the agent."""
    st.subheader("Resultado de la comparacion")

    col1, col2, col3 = st.columns(3)
    col1.metric("Precio vivienda", format_currency(data["property_price"]))
    col2.metric(
        "Media del barrio",
        format_currency(data["neighborhood_average_price"]),
    )
    col3.metric(
        "Diferencia",
        format_currency(data["absolute_difference"]),
        format_percent(data["percentage_difference"]),
    )

    classification = data["classification"]
    if classification == "Infravalorada":
        st.success(f"Clasificacion: {classification}")
    elif classification == "Sobrevalorada":
        st.warning(f"Clasificacion: {classification}")
    else:
        st.info(f"Clasificacion: {classification}")

    with st.expander("Ver estadisticas del barrio"):
        show_neighborhood_stats(data["neighborhood_stats"])


def show_valuation_result(data: dict[str, Any]) -> None:
    """Render valuation results returned by the agent."""
    st.subheader("Valoracion estimada")
    st.metric("Precio estimado", format_currency(data["estimated_price"]))

    with st.expander("Datos enviados al modelo"):
        st.json(data["input"])


def render_response(response: dict[str, Any]) -> None:
    """Render a structured response from SmartAdvisorAgent."""
    if not response.get("success"):
        show_error(response.get("error"))
        return

    intent = response.get("intent")
    data = response.get("data", {})

    if intent == "neighborhood_stats":
        show_neighborhood_stats(data)
    elif intent == "property_comparison":
        show_property_comparison(data)
    elif intent == "valuation":
        show_valuation_result(data)
    else:
        st.info("Consulta procesada correctamente.")
        st.json(response)


def render_valuation_form(agent: SmartAdvisorAgent) -> None:
    """Render the valuation form."""
    with st.form("valuation_form"):
        st.subheader("Valorar una vivienda")
        neighborhood = st.text_input("Barrio", placeholder="Ej. Palacio")
        area = st.number_input("Superficie construida (m2)", min_value=1.0, step=1.0)
        rooms = st.number_input("Habitaciones", min_value=0, step=1)
        bathrooms = st.number_input("Banos", min_value=0, step=1)
        has_lift = st.checkbox("Tiene ascensor")
        has_terrace = st.checkbox("Tiene terraza")
        has_parking = st.checkbox("Tiene plaza de garaje")
        submitted = st.form_submit_button("Analizar", use_container_width=True)

    if submitted:
        request = {
            "intent": "valuation",
            "LOCATIONNAME": neighborhood,
            "CONSTRUCTEDAREA": area,
            "ROOMNUMBER": rooms,
            "BATHNUMBER": bathrooms,
            "HASLIFT": int(has_lift),
            "HASTERRACE": int(has_terrace),
            "HASPARKINGSPACE": int(has_parking),
        }
        render_response(agent.process_request(request))


def render_comparison_form(agent: SmartAdvisorAgent) -> None:
    """Render the property comparison form."""
    with st.form("comparison_form"):
        st.subheader("Comparar una vivienda con su barrio")
        neighborhood = st.text_input("Barrio", placeholder="Ej. Palacio")
        property_price = st.number_input(
            "Precio de la vivienda (EUR)",
            min_value=1.0,
            step=1000.0,
        )
        submitted = st.form_submit_button("Analizar", use_container_width=True)

    if submitted:
        request = {
            "intent": "property_comparison",
            "neighborhood": neighborhood,
            "property_price": property_price,
        }
        render_response(agent.process_request(request))


def render_neighborhood_form(agent: SmartAdvisorAgent) -> None:
    """Render the neighborhood analysis form."""
    with st.form("neighborhood_form"):
        st.subheader("Analizar un barrio")
        neighborhood = st.text_input("Barrio", placeholder="Ej. Palacio")
        submitted = st.form_submit_button("Analizar", use_container_width=True)

    if submitted:
        request = {
            "intent": "neighborhood_stats",
            "neighborhood": neighborhood,
        }
        render_response(agent.process_request(request))


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="house",
        layout="wide",
    )

    agent = get_agent()

    with st.sidebar:
        st.title(APP_TITLE)
        st.caption("TFM - Data Science & Machine Learning")
        st.divider()
        selected_feature = st.radio(
            "Funcionalidad",
            [
                FEATURE_VALUATION,
                FEATURE_COMPARISON,
                FEATURE_NEIGHBORHOOD,
            ],
        )
        st.divider()
        st.caption(
            "Backend modular preparado para integrar el modelo definitivo "
            "sin cambiar la interfaz."
        )

    st.title(APP_TITLE)
    st.write(
        "Asistente inmobiliario inteligente para valorar viviendas, comparar "
        "precios con el barrio y consultar estadisticas inmobiliarias."
    )

    st.divider()

    if selected_feature == FEATURE_VALUATION:
        render_valuation_form(agent)
    elif selected_feature == FEATURE_COMPARISON:
        render_comparison_form(agent)
    elif selected_feature == FEATURE_NEIGHBORHOOD:
        render_neighborhood_form(agent)


if __name__ == "__main__":
    main()
