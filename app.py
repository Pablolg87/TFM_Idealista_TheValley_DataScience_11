"""Premium Streamlit interface for the Smart Advisor MVP."""

from datetime import date
from typing import Any

import streamlit as st

from agent import SmartAdvisorAgent


APP_TITLE = "Idealista Smart Advisor"
FEATURE_VALUATION = "Property Valuation"
FEATURE_COMPARISON = "Compare With Neighborhood"
FEATURE_NEIGHBORHOOD = "Neighborhood Intelligence"
FEATURE_CHAT = "Chat Smart Advisor"


def get_agent() -> SmartAdvisorAgent:
    """Create or reuse the Smart Advisor agent."""
    if "agent" not in st.session_state:
        st.session_state.agent = SmartAdvisorAgent()

    return st.session_state.agent


def format_currency(value: float) -> str:
    """Format numeric values as euros for display."""
    return f"€ {value:,.0f}".replace(",", ".")


def format_percent(value: float) -> str:
    """Format percentage values for display."""
    return f"{value:.2f}%"


def inject_css() -> None:
    """Inject the custom visual system."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #FFFFFF;
            --panel: #F7F8FA;
            --ink: #171717;
            --muted: #667085;
            --line: #E7E9EE;
            --accent: #FFF15A;
            --accent-soft: #FFFBD1;
            --ai: #EEF6FF;
            --ai-border: #CFE5FF;
            --shadow: 0 18px 50px rgba(16, 24, 40, 0.08);
            --radius: 8px;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--bg);
            color: var(--ink);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0);
        }

        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--line);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 32px;
            padding-bottom: 56px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }

        .hero {
            padding: 28px 0 24px 0;
            border-bottom: 1px solid var(--line);
            margin-bottom: 28px;
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 18px;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 8px;
            background: var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #111111;
            font-weight: 900;
            box-shadow: 0 10px 26px rgba(255, 241, 90, 0.35);
        }

        .brand-copy {
            color: var(--muted);
            font-size: 14px;
            line-height: 1.2;
        }

        .hero-title {
            font-size: 46px;
            line-height: 1.03;
            font-weight: 780;
            margin: 0;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 18px;
            margin-top: 12px;
            max-width: 740px;
        }

        .card {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 24px;
            margin-bottom: 18px;
        }

        .soft-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 20px;
            margin-bottom: 16px;
        }

        .card-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 18px;
        }

        .eyebrow {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            font-weight: 760;
            letter-spacing: .08em;
        }

        .section-title {
            font-size: 22px;
            font-weight: 760;
            margin: 0;
        }

        .field-group {
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 14px 14px 4px 14px;
            margin-bottom: 14px;
            background: #FFFFFF;
        }

        .field-label {
            color: var(--muted);
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .result-price {
            font-size: 48px;
            line-height: 1;
            font-weight: 820;
            color: var(--ink);
            margin: 8px 0 10px 0;
        }

        .result-hero {
            background:
                linear-gradient(135deg, rgba(255, 241, 90, 0.22) 0%, rgba(255,255,255,0) 42%),
                #FFFFFF;
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 34px;
            margin: 28px 0;
            text-align: center;
        }

        .hero-price {
            font-size: 68px;
            line-height: 1;
            font-weight: 850;
            color: var(--ink);
            margin: 14px 0 16px 0;
        }

        .valuation-date {
            color: var(--muted);
            font-size: 13px;
            margin-top: 8px;
        }

        .result-caption {
            color: var(--muted);
            font-size: 14px;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 22px;
        }

        .kpi-grid-wide {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin-top: 26px;
        }

        .kpi-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 16px;
        }

        .kpi-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 720;
            text-transform: uppercase;
            letter-spacing: .06em;
        }

        .kpi-value {
            color: var(--ink);
            font-size: 22px;
            font-weight: 760;
            margin-top: 8px;
        }

        .summary-card {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 24px;
            margin: 22px 0;
        }

        .summary-text {
            color: #344054;
            font-size: 17px;
            line-height: 1.65;
            margin-top: 10px;
        }

        .insight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-top: 18px;
        }

        .insight-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 18px;
        }

        .insight-value {
            font-size: 24px;
            font-weight: 780;
            color: var(--ink);
            margin-top: 8px;
        }

        .ai-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid var(--ai-border);
            background: var(--ai);
            border-radius: 999px;
            color: #175CD3;
            font-size: 13px;
            font-weight: 760;
            padding: 8px 12px;
        }

        .chat-panel {
            background: linear-gradient(180deg, #FFFFFF 0%, #F9FAFB 100%);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 24px;
            margin-top: 34px;
        }

        .advisor-focus {
            border: 1px solid var(--ai-border);
            background:
                linear-gradient(135deg, rgba(238, 246, 255, 0.95) 0%, rgba(255,255,255,1) 42%),
                #FFFFFF;
            padding: 30px;
        }

        .chat-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }

        .advisor-avatar {
            width: 38px;
            height: 38px;
            border-radius: 999px;
            background: var(--ai);
            border: 1px solid var(--ai-border);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #175CD3;
            font-weight: 850;
        }

        .chat-bubble {
            border-radius: 8px;
            padding: 14px 16px;
            margin: 10px 0;
            max-width: 860px;
        }

        .chat-user {
            background: #111111;
            color: #FFFFFF;
            margin-left: auto;
        }

        .chat-ai {
            background: var(--ai);
            border: 1px solid var(--ai-border);
            color: #111111;
        }

        .example-chip {
            display: inline-block;
            border: 1px solid var(--line);
            background: #FFFFFF;
            border-radius: 999px;
            color: var(--muted);
            font-size: 13px;
            padding: 8px 12px;
            margin: 4px 6px 4px 0;
        }

        .flow-steps {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 4px 0 28px 0;
        }

        .flow-step {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 14px;
            position: relative;
        }

        .flow-step.active {
            border-color: var(--ai-border);
            background: var(--ai);
        }

        .flow-step.done {
            border-color: #E6DB38;
            background: #FFFDE8;
        }

        .flow-number {
            width: 24px;
            height: 24px;
            border-radius: 999px;
            background: var(--panel);
            color: var(--muted);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .flow-step.active .flow-number {
            background: #175CD3;
            color: #FFFFFF;
        }

        .flow-step.done .flow-number {
            background: var(--accent);
            color: #111111;
        }

        .flow-title {
            font-size: 14px;
            font-weight: 760;
            color: var(--ink);
        }

        .flow-copy {
            font-size: 12px;
            color: var(--muted);
            margin-top: 4px;
        }

        .suggestion-label {
            color: var(--muted);
            font-size: 13px;
            font-weight: 760;
            margin: 18px 0 8px 0;
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] button {
            background: #111111;
            color: #FFFFFF;
            border: 0;
            border-radius: 8px;
            min-height: 46px;
            font-weight: 760;
            transition: all .16s ease;
        }

        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background: #000000;
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(16, 24, 40, 0.12);
        }

        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .05em;
            font-weight: 760;
        }

        input, textarea, [data-baseweb="select"] {
            border-radius: 8px;
        }

        @media (max-width: 980px) {
            .kpi-grid-wide,
            .insight-grid,
            .flow-steps {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .hero-price {
                font-size: 48px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the premium application header."""
    st.markdown(
        """
        <section class="hero">
            <div class="brand-row">
                <div class="brand-mark">IA</div>
                <div>
                    <div class="eyebrow">AI Real Estate Advisor</div>
                    <div class="brand-copy">Intelligent Property Valuation powered by Machine Learning</div>
                </div>
            </div>
            <h1 class="hero-title">Idealista Smart Advisor</h1>
            <p class="hero-subtitle">
                A premium market intelligence platform for property valuation,
                neighborhood analytics and AI-assisted real estate decisions.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_flow_steps(current_step: int) -> None:
    """Render the valuation journey as a visual progress bar."""
    steps = [
        ("Input data", "Property features"),
        ("Valuation", "ML estimate"),
        ("Market insights", "Dataset context"),
        ("AI Advisor", "Conversation"),
    ]
    cards = []
    for index, (title, copy) in enumerate(steps, start=1):
        state = "done" if index < current_step else "active" if index == current_step else ""
        cards.append(
            f"""
            <div class="flow-step {state}">
                <div class="flow-number">{index}</div>
                <div class="flow-title">{title}</div>
                <div class="flow-copy">{copy}</div>
            </div>
            """
        )

    st.markdown(
        f'<div class="flow-steps">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def open_card(title: str, eyebrow: str = "") -> None:
    """Open a visual card with title markup."""
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">
                <div>
                    <div class="eyebrow">{eyebrow}</div>
                    <h2 class="section-title">{title}</h2>
                </div>
                <span class="ai-pill">AI-ready</span>
            </div>
        """,
        unsafe_allow_html=True,
    )


def close_card() -> None:
    """Close a visual card."""
    st.markdown("</div>", unsafe_allow_html=True)


def show_error(message: str | None) -> None:
    """Show a user-friendly backend error."""
    if not message:
        message = "No se ha podido completar la consulta."

    if "model" in message.lower() or "predict" in message.lower():
        st.warning(
            "El modelo de valoracion todavia no esta disponible. "
            "La interfaz esta preparada para integrarlo sin cambios visuales."
        )
        return

    st.error(message)


def show_empty_result() -> None:
    """Render an elegant empty result state."""
    open_card("Valuation Output", "Live estimate")
    st.markdown(
        """
        <div class="result-caption">Complete the property form and run the analysis.</div>
        <div class="result-price">€ --</div>
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Confidence</div><div class="kpi-value">Pending</div></div>
            <div class="kpi-card"><div class="kpi-label">Zone</div><div class="kpi-value">--</div></div>
            <div class="kpi-card"><div class="kpi-label">Estimated €/m2</div><div class="kpi-value">--</div></div>
            <div class="kpi-card"><div class="kpi-label">Area Score</div><div class="kpi-value">--</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    close_card()


def show_result_ready() -> None:
    """Render a compact state when the main result is shown below."""
    open_card("Valuation Ready", "Output generated")
    st.markdown(
        """
        <div class="result-caption">The full valuation report is displayed below.</div>
        <div class="result-price">Ready</div>
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Report</div><div class="kpi-value">Generated</div></div>
            <div class="kpi-card"><div class="kpi-label">Advisor</div><div class="kpi-value">Online</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    close_card()


def get_market_context(
    agent: SmartAdvisorAgent,
    input_data: dict[str, Any],
    estimated_price: float,
) -> dict[str, Any] | None:
    """Get market context through SmartAdvisorAgent only."""
    neighborhood = input_data.get("LOCATIONNAME")
    if not neighborhood:
        return None

    response = agent.process_request(
        {
            "intent": "property_comparison",
            "neighborhood": neighborhood,
            "property_price": estimated_price,
        }
    )
    if not response.get("success"):
        return None

    return response.get("data", {})


def build_property_summary(
    input_data: dict[str, Any],
    market_context: dict[str, Any] | None,
) -> str:
    """Build a short automatic summary from agent-provided data."""
    neighborhood = input_data.get("LOCATIONNAME", "la zona seleccionada")
    area = input_data.get("CONSTRUCTEDAREA", "la superficie indicada")
    rooms = input_data.get("ROOMNUMBER", "varias")
    bathrooms = input_data.get("BATHNUMBER", "varios")

    if market_context:
        classification = market_context.get("classification", "En precio").lower()
        percentage = market_context.get("percentage_difference", 0)
        return (
            f"La vivienda en {neighborhood} presenta una valoracion {classification} "
            f"respecto a la media del barrio. La estimacion combina una superficie "
            f"de {area} m2, {rooms} habitaciones, {bathrooms} banos y el equipamiento "
            f"declarado. La diferencia frente al mercado local es de "
            f"{format_percent(float(percentage))}."
        )

    return (
        f"La vivienda en {neighborhood} ha sido valorada con el modelo MVP a partir "
        f"de {area} m2, {rooms} habitaciones, {bathrooms} banos y sus amenities. "
        "El resultado esta preparado para enriquecerse con el modelo definitivo."
    )


def render_property_summary(
    input_data: dict[str, Any],
    market_context: dict[str, Any] | None,
) -> None:
    """Render the automatic property summary."""
    summary = build_property_summary(input_data, market_context)
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="eyebrow">Automatic narrative</div>
            <h2 class="section-title">Property Summary</h2>
            <div class="summary-text">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_insights(
    estimated_price: float,
    unit_price: float,
    market_context: dict[str, Any] | None,
) -> None:
    """Render market context using only data returned by SmartAdvisorAgent."""
    if not market_context:
        return

    stats = market_context.get("neighborhood_stats", {})
    avg_price = float(market_context.get("neighborhood_average_price", 0) or 0)
    percentage = float(market_context.get("percentage_difference", 0) or 0)
    classification = market_context.get("classification", "N/A")
    property_count = stats.get("property_count", 0)
    avg_unit_price = float(stats.get("average_unit_price", 0) or 0)
    avg_area = float(stats.get("average_area", 0) or 0)

    st.markdown(
        f"""
        <div class="summary-card">
            <div class="eyebrow">Dataset context</div>
            <h2 class="section-title">Market Insights</h2>
            <div class="insight-grid">
                <div class="insight-card">
                    <div class="kpi-label">Neighborhood average</div>
                    <div class="insight-value">{format_currency(avg_price)}</div>
                </div>
                <div class="insight-card">
                    <div class="kpi-label">Position vs average</div>
                    <div class="insight-value">{format_percent(percentage)}</div>
                </div>
                <div class="insight-card">
                    <div class="kpi-label">Market status</div>
                    <div class="insight-value">{classification}</div>
                </div>
                <div class="insight-card">
                    <div class="kpi-label">Neighborhood €/m2</div>
                    <div class="insight-value">{format_currency(avg_unit_price)}</div>
                </div>
                <div class="insight-card">
                    <div class="kpi-label">Properties analyzed</div>
                    <div class="insight-value">{property_count}</div>
                </div>
                <div class="insight-card">
                    <div class="kpi-label">Zone score</div>
                    <div class="insight-value">MVP-ready</div>
                </div>
            </div>
            <div class="result-caption" style="margin-top:14px;">
                Estimated property €/m2: {format_currency(unit_price)} · Average area in the zone: {avg_area:.2f} m2
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_valuation_result(
    data: dict[str, Any],
    market_context: dict[str, Any] | None = None,
) -> None:
    """Render valuation results returned by the agent."""
    estimated_price = float(data["estimated_price"])
    input_data = data.get("input", {})
    area = float(input_data.get("CONSTRUCTEDAREA", 0) or 0)
    unit_price = estimated_price / area if area > 0 else 0
    neighborhood = input_data.get("LOCATIONNAME", "N/A")
    st.markdown(
        f"""
        <section class="result-hero">
            <div class="eyebrow">Estimated Market Value</div>
            <div class="hero-price">{format_currency(estimated_price)}</div>
            <div class="result-caption">
                Machine Learning valuation for {neighborhood}
            </div>
            <div class="valuation-date">Valuation date: {date.today().strftime("%d/%m/%Y")}</div>
            <div class="kpi-grid-wide">
                <div class="kpi-card"><div class="kpi-label">Estimated €/m2</div><div class="kpi-value">{format_currency(unit_price)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Neighborhood</div><div class="kpi-value">{neighborhood}</div></div>
                <div class="kpi-card"><div class="kpi-label">Confidence</div><div class="kpi-value">MVP model</div></div>
                <div class="kpi-card"><div class="kpi-label">Bedrooms</div><div class="kpi-value">{input_data.get("ROOMNUMBER", "N/A")}</div></div>
                <div class="kpi-card"><div class="kpi-label">Bathrooms</div><div class="kpi-value">{input_data.get("BATHNUMBER", "N/A")}</div></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_property_summary(input_data, market_context)
    render_market_insights(estimated_price, unit_price, market_context)

    with st.expander("Variables sent to the model"):
        st.json(input_data)


def show_valuation_preview(data: dict[str, Any]) -> None:
    """Render compact valuation output for generic response rendering."""
    estimated_price = float(data["estimated_price"])
    input_data = data.get("input", {})
    area = float(input_data.get("CONSTRUCTEDAREA", 0) or 0)
    unit_price = estimated_price / area if area > 0 else 0
    neighborhood = input_data.get("LOCATIONNAME", "N/A")

    open_card("Valuation Output", "Live estimate")
    st.markdown(
        f"""
        <div class="result-caption">Estimated market value</div>
        <div class="result-price">{format_currency(estimated_price)}</div>
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Confidence</div><div class="kpi-value">MVP model</div></div>
            <div class="kpi-card"><div class="kpi-label">Zone</div><div class="kpi-value">{neighborhood}</div></div>
            <div class="kpi-card"><div class="kpi-label">Estimated €/m2</div><div class="kpi-value">{format_currency(unit_price)}</div></div>
            <div class="kpi-card"><div class="kpi-label">Area Score</div><div class="kpi-value">Data-ready</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    close_card()


def show_neighborhood_stats(data: dict[str, Any]) -> None:
    """Render neighborhood statistics returned by the agent."""
    open_card(f"{data['neighborhood']} Market Snapshot", "Neighborhood intelligence")
    col1, col2, col3 = st.columns(3)
    col1.metric("Properties", f"{data['property_count']:,}".replace(",", "."))
    col2.metric("Average price", format_currency(data["average_price"]))
    col3.metric("Median price", format_currency(data["median_price"]))

    col4, col5, col6 = st.columns(3)
    col4.metric("Average €/m2", format_currency(data["average_unit_price"]))
    col5.metric("Average area", f"{data['average_area']:.2f} m2")
    col6.metric(
        "Price range",
        f"{format_currency(data['min_price'])} - {format_currency(data['max_price'])}",
    )
    close_card()


def show_property_comparison(data: dict[str, Any]) -> None:
    """Render property comparison results returned by the agent."""
    open_card("Property vs Neighborhood", "Benchmark")
    col1, col2, col3 = st.columns(3)
    col1.metric("Property price", format_currency(data["property_price"]))
    col2.metric("Neighborhood average", format_currency(data["neighborhood_average_price"]))
    col3.metric(
        "Difference",
        format_currency(data["absolute_difference"]),
        format_percent(data["percentage_difference"]),
    )

    classification = data["classification"]
    if classification == "Infravalorada":
        st.success(f"Classification: {classification}")
    elif classification == "Sobrevalorada":
        st.warning(f"Classification: {classification}")
    else:
        st.info(f"Classification: {classification}")
    close_card()


def show_neighborhood_comparison(data: dict[str, Any]) -> None:
    """Render neighborhood comparison results returned by the agent."""
    first = data["first_neighborhood"]
    second = data["second_neighborhood"]
    price_difference = data["average_price_difference"]
    unit_price_difference = data["average_unit_price_difference"]

    open_card("Neighborhood Comparison", "Market intelligence")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(first["neighborhood"], format_currency(first["average_price"]))
        st.metric("Average €/m2", format_currency(first["average_unit_price"]))
        st.metric("Properties", f"{first['property_count']:,}".replace(",", "."))

    with col2:
        st.metric(second["neighborhood"], format_currency(second["average_price"]))
        st.metric("Average €/m2", format_currency(second["average_unit_price"]))
        st.metric("Properties", f"{second['property_count']:,}".replace(",", "."))

    st.info(
        "Average price difference: "
        f"{format_currency(price_difference['absolute_difference'])} "
        f"({format_percent(price_difference['percentage_difference'])}). "
        "Average €/m2 difference: "
        f"{format_currency(unit_price_difference['absolute_difference'])} "
        f"({format_percent(unit_price_difference['percentage_difference'])})."
    )
    close_card()


def show_budget_recommendations(data: dict[str, Any]) -> None:
    """Render budget recommendation results returned by the agent."""
    open_card("Budget Recommendations", "AI market assistant")
    st.metric("Budget", format_currency(data["budget"]))

    recommendations = data.get("recommendations", [])
    if not recommendations:
        st.warning("No neighborhoods match that budget.")
        close_card()
        return

    for recommendation in recommendations:
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Neighborhood", recommendation["neighborhood"])
            col2.metric("Average price", format_currency(recommendation["average_price"]))
            col3.metric("Budget gap", format_currency(recommendation["budget_gap"]))
            st.caption(
                f"{recommendation['property_count']} properties analyzed | "
                f"{format_currency(recommendation['average_unit_price'])}/m2 | "
                f"average area {recommendation['average_area']:.2f} m2"
            )
    close_card()


def render_response(response: dict[str, Any]) -> None:
    """Render a structured response from SmartAdvisorAgent."""
    message = response.get("message")
    if message:
        st.markdown(f'<div class="chat-bubble chat-ai">{message}</div>', unsafe_allow_html=True)

    if not response.get("success"):
        show_error(response.get("error"))
        return

    intent = response.get("intent")
    data = response.get("data", {})

    if intent in {"neighborhood_stats", "neighborhood_analysis"}:
        show_neighborhood_stats(data)
    elif intent == "property_comparison":
        show_property_comparison(data)
    elif intent == "neighborhood_comparison":
        show_neighborhood_comparison(data)
    elif intent == "budget_recommendation":
        show_budget_recommendations(data)
    elif intent == "valuation":
        show_valuation_preview(data)
    else:
        st.json(response)


def render_property_features(agent: SmartAdvisorAgent) -> None:
    """Render the main valuation workspace."""
    left, right = st.columns([0.92, 1.08], gap="large")

    with left:
        open_card("Property Features", "Input workspace")
        with st.form("valuation_form"):
            st.markdown('<div class="field-group"><div class="field-label">Property / Location</div>', unsafe_allow_html=True)
            neighborhood = st.text_input("Neighborhood", placeholder="Palacio")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="field-group"><div class="field-label">Size</div>', unsafe_allow_html=True)
            area = st.number_input("Constructed area (m2)", min_value=1.0, value=80.0, step=1.0)
            st.markdown("</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="field-group"><div class="field-label">Rooms</div>', unsafe_allow_html=True)
                rooms = st.number_input("Bedrooms", min_value=0, value=2, step=1)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="field-group"><div class="field-label">Bathrooms</div>', unsafe_allow_html=True)
                bathrooms = st.number_input("Bathrooms", min_value=0, value=1, step=1)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="field-group"><div class="field-label">Amenities</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                has_lift = st.checkbox("Lift", value=True)
            with c2:
                has_terrace = st.checkbox("Terrace")
            with c3:
                has_parking = st.checkbox("Parking")
            st.markdown("</div>", unsafe_allow_html=True)

            submitted = st.form_submit_button("Analyze", use_container_width=True)
        close_card()

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
            response = agent.process_request(request)
            st.session_state.last_valuation_response = response
            st.session_state.pop("last_advisor_question", None)
            st.session_state.pop("last_advisor_response", None)

            market_context = None
            if response.get("success"):
                market_context = get_market_context(
                    agent=agent,
                    input_data=response["data"].get("input", {}),
                    estimated_price=float(response["data"]["estimated_price"]),
                )
            st.session_state.last_market_context = market_context

    with right:
        response = st.session_state.get("last_valuation_response")
        if response and response.get("success"):
            show_result_ready()
        elif response:
            render_response(response)
        else:
            show_empty_result()

    response = st.session_state.get("last_valuation_response")
    if response and response.get("success"):
        show_valuation_result(
            response["data"],
            st.session_state.get("last_market_context"),
        )
        render_property_advisor_chat(agent, response)


def render_comparison_form(agent: SmartAdvisorAgent) -> None:
    """Render the property comparison form."""
    open_card("Compare With Neighborhood", "Benchmark tool")
    with st.form("comparison_form"):
        neighborhood = st.text_input("Neighborhood", placeholder="Palacio")
        property_price = st.number_input(
            "Property price",
            min_value=1.0,
            step=1000.0,
        )
        submitted = st.form_submit_button("Analyze", use_container_width=True)

    if submitted:
        request = {
            "intent": "property_comparison",
            "neighborhood": neighborhood,
            "property_price": property_price,
        }
        render_response(agent.process_request(request))
    close_card()


def render_neighborhood_form(agent: SmartAdvisorAgent) -> None:
    """Render the neighborhood analysis form."""
    open_card("Neighborhood Intelligence", "Market analytics")
    with st.form("neighborhood_form"):
        neighborhood = st.text_input("Neighborhood", placeholder="Palacio")
        submitted = st.form_submit_button("Analyze", use_container_width=True)

    if submitted:
        request = {
            "intent": "neighborhood_stats",
            "neighborhood": neighborhood,
        }
        render_response(agent.process_request(request))
    close_card()


def render_chat_mode(agent: SmartAdvisorAgent) -> None:
    """Render a natural-language interaction mode."""
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Conversational AI layer</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Chat Smart Advisor</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-bubble chat-ai">Ask for valuations, market analysis, neighborhood comparisons or budget recommendations.</div>',
        unsafe_allow_html=True,
    )

    examples = [
        "Valora una vivienda en Palacio de 80 m2 con 2 habitaciones y 1 baño con ascensor",
        "Analiza el barrio Palacio",
        "Compara Palacio y Sol",
        "Tengo 300000 euros, ¿en qué barrios puedo comprar?",
    ]
    st.markdown(
        "".join([f'<span class="example-chip">{example}</span>' for example in examples]),
        unsafe_allow_html=True,
    )

    with st.form("chat_form"):
        message = st.text_area(
            "Your query",
            placeholder="Analiza el barrio Palacio",
            height=120,
        )
        submitted = st.form_submit_button("Send to Smart Advisor", use_container_width=True)

    if submitted:
        if not message.strip():
            st.warning("Write a query so Smart Advisor can help.")
        else:
            st.markdown(f'<div class="chat-bubble chat-user">{message}</div>', unsafe_allow_html=True)
            response = agent.process_message(message)
            render_response(response)

            with st.expander("Structured response"):
                st.json(response)

    st.markdown("</div>", unsafe_allow_html=True)


def render_property_advisor_chat(
    agent: SmartAdvisorAgent,
    valuation_response: dict[str, Any],
) -> None:
    """Render the contextual advisor chat after a valuation."""
    input_data = valuation_response.get("data", {}).get("input", {})
    neighborhood = input_data.get("LOCATIONNAME", "esta zona")
    suggestions = [
        f"Analiza el barrio {neighborhood}",
        "Compara Palacio y Sol",
        "Tengo 300000 euros, ¿en qué barrios puedo comprar?",
        f"Valora una vivienda en {neighborhood} de 90 m2 con 3 habitaciones y 2 baños con ascensor",
        f"Analiza la zona {neighborhood}",
        "Tengo 450000 euros, ¿en qué barrios puedo comprar?",
    ]

    st.markdown('<div class="chat-panel advisor-focus">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="chat-header">
            <div class="advisor-avatar">AI</div>
            <div>
                <div class="eyebrow">Conversational intelligence</div>
                <h2 class="section-title">AI Property Advisor</h2>
            </div>
        </div>
        <div class="chat-bubble chat-ai">
            La valoración ya está lista. Puedes preguntarme por qué se ha obtenido
            este precio, cómo influye cada característica de la vivienda o solicitar
            información adicional sobre el mercado.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="suggestion-label">Suggested questions</div>', unsafe_allow_html=True)
    suggestion_cols = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        with suggestion_cols[index % 2]:
            if st.button(suggestion, key=f"advisor_suggestion_{index}", use_container_width=True):
                st.session_state.last_advisor_question = suggestion
                st.session_state.last_advisor_response = agent.process_message(suggestion)

    if st.session_state.get("last_advisor_question") and st.session_state.get("last_advisor_response"):
        st.markdown(
            f'<div class="chat-bubble chat-user">{st.session_state.last_advisor_question}</div>',
            unsafe_allow_html=True,
        )
        render_response(st.session_state.last_advisor_response)

    st.markdown(
        "".join([f'<span class="example-chip">{suggestion}</span>' for suggestion in suggestions[:4]]),
        unsafe_allow_html=True,
    )

    with st.form("valuation_chat_form"):
        message = st.text_area(
            "Ask the advisor",
            placeholder=f"Analiza el barrio {neighborhood}",
            height=110,
        )
        submitted = st.form_submit_button("Ask AI Property Advisor", use_container_width=True)

    if submitted:
        if not message.strip():
            st.warning("Write a question so the advisor can help.")
        else:
            st.markdown(f'<div class="chat-bubble chat-user">{message}</div>', unsafe_allow_html=True)
            response = agent.process_message(message)
            st.session_state.last_advisor_question = message
            st.session_state.last_advisor_response = response
            render_response(response)

            with st.expander("Structured response"):
                st.json(response)

    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar() -> str:
    """Render the navigation sidebar."""
    with st.sidebar:
        st.markdown("### Idealista")
        st.markdown("#### Smart Advisor")
        st.caption("TFM · Data Science & Machine Learning")
        st.divider()
        selected_feature = st.radio(
            "Workspace",
            [
                FEATURE_VALUATION,
                FEATURE_COMPARISON,
                FEATURE_NEIGHBORHOOD,
                FEATURE_CHAT,
            ],
        )
        st.divider()
        st.markdown('<span class="ai-pill">ML backend online</span>', unsafe_allow_html=True)
        st.caption("Minimal SaaS interface for a production-ready demo.")

    return selected_feature


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🏠",
        layout="wide",
    )
    inject_css()
    agent = get_agent()
    selected_feature = render_sidebar()
    render_header()

    if selected_feature == FEATURE_VALUATION:
        current_step = 4 if st.session_state.get("last_valuation_response", {}).get("success") else 1
        render_flow_steps(current_step)
        render_property_features(agent)
    elif selected_feature == FEATURE_COMPARISON:
        render_comparison_form(agent)
        render_chat_mode(agent)
    elif selected_feature == FEATURE_NEIGHBORHOOD:
        render_neighborhood_form(agent)
        render_chat_mode(agent)
    elif selected_feature == FEATURE_CHAT:
        render_chat_mode(agent)


if __name__ == "__main__":
    main()
