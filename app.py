"""Streamlit interface for the Smart Advisor MVP."""

from typing import Any

import streamlit as st

from agent import SmartAdvisorAgent


APP_TITLE = "Idealista Smart Advisor"
FEATURE_VALUATION = "Valorar una vivienda"
FEATURE_COMPARISON = "Comparar una vivienda con el barrio"
FEATURE_NEIGHBORHOOD = "Analizar un barrio"
FEATURE_CHAT = "Chat Smart Advisor"


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


def show_neighborhood_comparison(data: dict[str, Any]) -> None:
    """Render neighborhood comparison results returned by the agent."""
    first = data["first_neighborhood"]
    second = data["second_neighborhood"]
    price_difference = data["average_price_difference"]
    unit_price_difference = data["average_unit_price_difference"]

    st.subheader("Comparacion de barrios")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {first['neighborhood']}")
        st.metric("Precio medio", format_currency(first["average_price"]))
        st.metric("Precio medio por m2", format_currency(first["average_unit_price"]))
        st.metric("Viviendas analizadas", f"{first['property_count']:,}".replace(",", "."))

    with col2:
        st.markdown(f"### {second['neighborhood']}")
        st.metric("Precio medio", format_currency(second["average_price"]))
        st.metric("Precio medio por m2", format_currency(second["average_unit_price"]))
        st.metric("Viviendas analizadas", f"{second['property_count']:,}".replace(",", "."))

    st.info(
        "Diferencia de precio medio: "
        f"{format_currency(price_difference['absolute_difference'])} "
        f"({format_percent(price_difference['percentage_difference'])}). "
        "Diferencia de precio por m2: "
        f"{format_currency(unit_price_difference['absolute_difference'])} "
        f"({format_percent(unit_price_difference['percentage_difference'])})."
    )


def show_budget_recommendations(data: dict[str, Any]) -> None:
    """Render budget recommendation results returned by the agent."""
    st.subheader("Barrios recomendados por presupuesto")
    st.metric("Presupuesto", format_currency(data["budget"]))

    recommendations = data.get("recommendations", [])
    if not recommendations:
        st.warning("No se han encontrado barrios compatibles con ese presupuesto.")
        return

    for recommendation in recommendations:
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Barrio", recommendation["neighborhood"])
            col2.metric("Precio medio", format_currency(recommendation["average_price"]))
            col3.metric("Margen vs presupuesto", format_currency(recommendation["budget_gap"]))
            st.caption(
                f"{recommendation['property_count']} viviendas analizadas | "
                f"{format_currency(recommendation['average_unit_price'])}/m2 | "
                f"superficie media {recommendation['average_area']:.2f} m2"
            )


def render_response(response: dict[str, Any]) -> None:
    """Render a structured response from SmartAdvisorAgent."""
    message = response.get("message")
    if message:
        st.write(message)

    if not response.get("success"):
        show_error(response.get("error"))
        return

    intent = response.get("intent")
    data = response.get("data", {})

    if intent == "neighborhood_stats":
        show_neighborhood_stats(data)
    elif intent == "neighborhood_analysis":
        show_neighborhood_stats(data)
    elif intent == "property_comparison":
        show_property_comparison(data)
    elif intent == "neighborhood_comparison":
        show_neighborhood_comparison(data)
    elif intent == "budget_recommendation":
        show_budget_recommendations(data)
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


def render_chat_mode(agent: SmartAdvisorAgent) -> None:
    """Render a natural-language interaction mode."""
    st.subheader("Chat Smart Advisor")
    st.write(
        "Escribe una consulta inmobiliaria en lenguaje natural. "
        "El agente interpretara la intencion y usara el backend del MVP."
    )

    st.markdown("Ejemplos:")
    st.code(
        "\n".join(
            [
                "Valora una vivienda en Palacio de 80 m2 con 2 habitaciones y 1 baño con ascensor",
                "Analiza el barrio Palacio",
                "Compara Palacio y Sol",
                "Tengo 300000 euros, ¿en qué barrios puedo comprar?",
            ]
        ),
        language="text",
    )

    with st.form("chat_form"):
        message = st.text_area(
            "Consulta",
            placeholder="Ej. Analiza el barrio Palacio",
            height=120,
        )
        submitted = st.form_submit_button("Analizar", use_container_width=True)

    if submitted:
        if not message.strip():
            st.warning("Escribe una consulta para que pueda ayudarte.")
            return

        response = agent.process_message(message)
        st.subheader("Respuesta")
        render_response(response)

        with st.expander("Ver respuesta estructurada"):
            st.json(response)


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
                FEATURE_CHAT,
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

    if selected_feature == FEATURE_CHAT:
        render_chat_mode(agent)
    elif selected_feature == FEATURE_VALUATION:
        render_valuation_form(agent)
    elif selected_feature == FEATURE_COMPARISON:
        render_comparison_form(agent)
    elif selected_feature == FEATURE_NEIGHBORHOOD:
        render_neighborhood_form(agent)


if __name__ == "__main__":
    main()
