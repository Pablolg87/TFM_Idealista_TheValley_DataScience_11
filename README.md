# TFM_Idealista_TheValley_DataScience_11
Intelligent Real Estate Valuation System developed as the Master's Final Project in Data Science &amp; Machine Learning at The Valley.

# 🏡 AI Real Estate Advisor

> **PropTech platform for real estate valuation and neighborhood intelligence powered by Machine Learning and Conversational AI.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

# 📌 Project Overview

AI Real Estate Advisor is the final Master's project developed at **The Valley Business & Tech School**.

The project combines:

- Machine Learning
- Data Science
- Conversational AI
- Business Intelligence
- Streamlit

to build an interactive decision-support platform for residential property valuation in Madrid.

Instead of presenting only a predictive model, the objective was to transform it into a complete PropTech solution capable of assisting users throughout different stages of a real estate decision.

---

# 🚀 Main Features

## 🤖 AI Property Advisor

A conversational assistant collects the property characteristics using natural language.

After the conversation, the application automatically generates:

- Estimated property value
- Estimated €/m²
- Feature importance
- Model confidence indicators
- Interactive dashboard

---

## 📊 Neighborhood Intelligence

Compare two Madrid neighborhoods using real market information.

Includes:

- Average prices
- Median prices
- Average €/m²
- Property characteristics
- Amenities
- Accessibility
- Visual comparison dashboards

---

## 💼 Investment Simulator

A recommendation engine that identifies the most interesting neighborhoods according to:

- Budget
- Minimum surface
- Number of bedrooms
- Investment priorities

The simulator provides:

- Top 5 recommended neighborhoods
- Opportunity ranking
- Opportunity score
- AI-generated recommendations
- Interactive charts

---

# 🧠 Machine Learning Model

Final selected model:

| Model | Random Forest |
|--------|---------------|
| Target | LOG_PRICE |
| RMSE | ~127,700 € |
| MAPE | 15.31% |
| Pseudo R² | 0.9448 |

The model was trained using a cleaned dataset of residential properties in Madrid.

---

# 📈 Dataset

The project uses residential property data from Madrid including:

- Sale price
- Surface
- Bedrooms
- Bathrooms
- District
- Neighborhood
- Amenities
- Accessibility
- Geographic indicators

After preprocessing, feature engineering and exploratory analysis, the dataset was used to train several predictive models.

---

# 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Altair
- Joblib

---

# 📂 Project Structure

```
├── app.py
├── dataset.csv
├── model.pkl
├── requirements.txt
├── config.py
├── predict.py
├── agent.py
├── assets/
└── README.md
```

---

# ▶️ Run locally

Clone the repository:

```bash
git clone https://github.com/Pablog87/TFM_Idealista_TheValley_DataScience_11.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# 🎯 Academic Objectives

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Regression
- Model Comparison
- Model Explainability
- Business Intelligence
- Conversational User Experience
- Streamlit Application Development

---

# 📸 Application

The application currently includes three functional modules:

✅ AI Property Advisor

✅ Neighborhood Intelligence

✅ Investment Simulator

---

# 🚧 Future Improvements

- SHAP local explanations
- Interactive Madrid map
- Rental yield estimation
- Market trend forecasting
- LLM-powered property reports
- User authentication
- Cloud deployment

---

# ⭐ Acknowledgements

Developed as the Final Master's Project (TFM) for the Master's in Data Science & Machine Learning at **The Valley Business & Tech School**.
