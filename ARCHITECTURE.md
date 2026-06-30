# ARCHITECTURE

# Idealista Smart Advisor

## System Architecture

Idealista Smart Advisor is designed as a modular AI-powered application.

The system separates the user interface, conversational agent, business logic and machine learning model.

This architecture allows each component to evolve independently.

---

# High Level Architecture

                User

                  │

                  ▼

        Streamlit Interface

                  │

                  ▼

      Conversational AI Agent

                  │

      Intent Detection Engine

                  │

      ┌───────────┼────────────┬─────────────┬────────────┐

      ▼           ▼            ▼             ▼

 Property     Neighborhood   Comparison   Investment
 Valuation    Intelligence

      │

      ▼

 Business Services

      │

      ▼

 Machine Learning Model

      │

      ▼

 Response Generator

      │

      ▼

 Streamlit UI

---

# Application Layers

## Layer 1

Presentation Layer

Responsible for:

- Streamlit interface
- User interaction
- Conversation history
- Displaying charts
- Displaying predictions

Files:

app.py

---

## Layer 2

Conversational Agent

Responsible for:

- Detecting user intent
- Maintaining conversation state
- Asking for missing information
- Calling the appropriate business service

Files:

agent.py

---

## Layer 3

Business Logic

Contains all application functionality.

Services:

predict.py

advisor.py

future modules:

comparison.py

investment.py

---

## Layer 4

Machine Learning

Responsible for:

- Loading trained model
- Feature preprocessing
- Prediction
- SHAP explanations

Resources:

model/model.pkl

model/features.pkl

---

# Conversation Flow

User writes a message.

↓

The Agent analyzes the message.

↓

The Agent detects the user's intention.

↓

The Agent determines which information is still missing.

↓

If required:

Ask another question.

↓

Otherwise:

Execute the correct business service.

↓

Generate response.

↓

Display result.

---

# Intent Detection

The system currently supports four intentions.

1.

Property valuation

Examples:

"I want to value a house"

"Estimate this property"

"How much is this apartment worth?"

↓

predict_price()

---

2.

Neighborhood analysis

Examples:

"Analyze Chamartín"

"Tell me about Salamanca"

↓

analyze_neighborhood()

---

3.

Neighborhood comparison

Examples:

"Compare Chamartín and Retiro"

↓

compare_neighborhoods()

---

4.

Investment opportunity

Examples:

"Is this apartment overpriced?"

↓

investment_score()

---

# Conversation States

The conversational agent works using conversation states.

Possible states:

IDLE

WAITING_FOR_NEIGHBORHOOD

WAITING_FOR_AREA

WAITING_FOR_ROOMS

WAITING_FOR_BATHROOMS

WAITING_FOR_PRICE

PREDICTING

SHOWING_RESULTS

This allows the assistant to simulate a natural conversation.

---

# Business Services

Each service performs one specific task.

predict_price()

↓

Estimate property price.

------------------------

analyze_neighborhood()

↓

Generate neighborhood insights.

------------------------

compare_neighborhoods()

↓

Generate comparison.

------------------------

investment_score()

↓

Detect investment opportunities.

---

# Project Structure

TFM_Idealista_TheValley_DataScience_11/

│

├── app.py

├── agent.py

├── advisor.py

├── predict.py

├── utils.py

├── config.py

│

├── data/

├── model/

├── docs/

├── images/

├── .streamlit/

---

# Future Improvements

Future versions may include:

- Real LLM integration
- Voice interaction
- Map visualization
- RAG with market reports
- Authentication
- User history
- Recommendation engine

The current MVP has been intentionally designed to support these future extensions without requiring architectural changes.