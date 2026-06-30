import streamlit as st
from agent import SmartAdvisorAgent

# -----------------------------
# Configuración de la página
# -----------------------------

st.set_page_config(
    page_title="Idealista Smart Advisor",
    page_icon="🏠",
    layout="wide"
)

# Crear el agente una única vez
if "agent" not in st.session_state:
    st.session_state.agent = SmartAdvisorAgent()

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("🏠 Idealista")

    st.markdown("## Smart Advisor")

    st.divider()

    st.markdown("### Funcionalidades")

    st.markdown("""
- 🏠 Valoración de viviendas

- 📍 Inteligencia de barrios

- ⚖️ Comparador de zonas

- 💰 Oportunidades de inversión
""")

    st.divider()

    st.caption("TFM · Máster Data Science & Machine Learning · The Valley")

# -----------------------------
# Título principal
# -----------------------------

st.title("🏠 Idealista Smart Advisor")

st.subheader(
    "Tu asesor inmobiliario inteligente basado en Machine Learning"
)

st.write("")

st.info(
    """
Hola 👋

Soy **Idealista Smart Advisor**.

Puedo ayudarte a:

🏠 Valorar una vivienda

📍 Analizar cualquier barrio

⚖️ Comparar zonas de Madrid

💰 Detectar oportunidades de inversión

Escríbeme lo que necesitas y te ayudaré paso a paso.
"""
)

# -----------------------------
# Acciones rápidas
# -----------------------------

st.markdown("## ¿Qué quieres hacer?")

col1, col2 = st.columns(2)

with col1:

    st.button("🏠 Valorar una vivienda", use_container_width=True)

    st.button("📍 Analizar un barrio", use_container_width=True)

with col2:

    st.button("⚖️ Comparar barrios", use_container_width=True)

    st.button("💰 Buscar oportunidades", use_container_width=True)

st.divider()

# -----------------------------
# Historial del chat
# -----------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# -----------------------------
# Caja de conversación
# -----------------------------

prompt = st.chat_input(
    "Escribe aquí tu consulta..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    respuesta = st.session_state.agent.process_message(prompt)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta
        }
    )

    with st.chat_message("assistant"):

        st.markdown(respuesta)