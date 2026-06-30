"""
Idealista Smart Advisor
Conversational Agent
"""


class SmartAdvisorAgent:

    def __init__(self):
        """Inicializa el estado de la conversación."""

        self.reset()

    def reset(self):
        """Reinicia la conversación."""

        self.intent = None

        self.data = {
            "neighborhood": None,
            "area": None,
            "rooms": None,
            "bathrooms": None,
            "quality": None,
            "price": None,
        }

        self.state = "IDLE"

    def process_message(self, message: str):
        """
        Procesa un mensaje del usuario.
        """

        message = message.lower()

        # --------------------------
        # Detección de intención
        # --------------------------

        if self.state == "IDLE":

            if "valorar" in message or "precio" in message:

                self.intent = "valuation"

                self.state = "WAITING_NEIGHBORHOOD"

                return (
                    "Perfecto. 🏠\n\n"
                    "Voy a ayudarte a estimar el precio de una vivienda.\n\n"
                    "¿En qué barrio se encuentra?"
                )

            elif "barrio" in message:

                return (
                    "Perfecto.\n\n"
                    "Próximamente analizaré cualquier barrio."
                )

            elif "compar" in message:

                return (
                    "Perfecto.\n\n"
                    "Próximamente podré comparar dos barrios."
                )

            elif "invers" in message:

                return (
                    "Perfecto.\n\n"
                    "Próximamente analizaré oportunidades de inversión."
                )

            else:

                return (
                    "No he entendido tu solicitud.\n\n"
                    "Puedes pedirme valorar una vivienda, analizar un barrio, "
                    "comparar zonas o detectar oportunidades de inversión."
                )

        # --------------------------
        # Flujo valoración
        # --------------------------

        if self.state == "WAITING_NEIGHBORHOOD":

            self.data["neighborhood"] = message.title()

            self.state = "WAITING_AREA"

            return (
                f"Perfecto, la vivienda está en **{self.data['neighborhood']}**.\n\n"
                "¿Cuál es la superficie construida en m²?"
            )

        if self.state == "WAITING_AREA":

            self.data["area"] = message

            self.state = "WAITING_ROOMS"

            return "¿Cuántas habitaciones tiene?"

        if self.state == "WAITING_ROOMS":

            self.data["rooms"] = message

            self.state = "WAITING_BATHROOMS"

            return "¿Cuántos baños tiene?"

        if self.state == "WAITING_BATHROOMS":

            self.data["bathrooms"] = message

            self.state = "FINISHED"

            return (
                "✅ Ya tengo toda la información.\n\n"
                "En el siguiente Sprint conectaré el modelo de Machine Learning."
            )

        return "No sé cómo continuar la conversación."