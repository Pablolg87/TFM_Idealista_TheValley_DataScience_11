# PROJECT CONTEXT

# Idealista Smart Advisor

## Project Overview

This project is the Master's Final Project (TFM) for the Master in Data Science & Machine Learning at The Valley.

The objective is to build an AI-powered Real Estate Assistant capable of helping users make better property decisions using Machine Learning.

The application will be developed as an MVP using Streamlit.

The final product should simulate an intelligent conversational assistant specialized in the real estate market.

No external LLM APIs (OpenAI, Claude, Gemini, etc.) will be used.

The conversational behaviour will be implemented through Python logic, state management and specialized tools.

---

# Technologies

- Python
- Streamlit
- Pandas
- Scikit-Learn
- SHAP
- Joblib
- Plotly

---

# Dataset

The cleaned dataset is stored in Google Drive.

During development a demo dataset may be used.

Later it will be replaced by the final cleaned dataset without modifying the application architecture.

---

# Machine Learning Model

The application will use a trained regression model capable of predicting housing prices.

The model will be exported as:

- modelo.pkl

The list of features will be stored as:

- feature_columns.pkl

---

# Product Vision

The application should look like a professional product that could be integrated into Idealista.

The user experience should be inspired by Idealista's website.

The interface should be clean, modern and simple.

---

# Conversational Agent

The application simulates an AI Real Estate Expert.

The user interacts only through conversation.

The agent detects the user's intention and decides which internal tool should be executed.

The conversation should feel natural.

---

# Agent Tools

The assistant has four capabilities.

## 1. Property Valuation

Estimate the market value of a property.

Inputs:

- neighborhood
- constructed area
- rooms
- bathrooms
- cadastral quality
- additional variables

Output:

- estimated price
- confidence interval
- SHAP explanation
- natural language explanation

---

## 2. Neighborhood Intelligence

Analyze any neighborhood.

Outputs:

- average price
- average price per square meter
- neighborhood score
- homogeneity
- variability
- automatic conclusion

---

## 3. Neighborhood Comparison

Compare two neighborhoods.

Outputs:

- average price
- €/m²
- homogeneity
- neighborhood score
- comparison summary

---

## 4. Investment Opportunity Detector

Compare:

Predicted Price

vs

Listing Price

Outputs:

- opportunity score
- price difference
- recommendation

---

# Architecture

User

↓

Streamlit

↓

Conversational Agent

↓

Intent Detection

↓

Business Tool

↓

Machine Learning Model

↓

Response

---

# Development Rules

- Write clean and modular code.
- Keep business logic separated from UI.
- Use reusable functions.
- Document important functions.
- Avoid hardcoded values.
- Make the project easy to maintain.

---

# Current Development Stage

The project starts from scratch.

The first objective is building a functional MVP.

The final model will be integrated later.

The application architecture should remain unchanged.