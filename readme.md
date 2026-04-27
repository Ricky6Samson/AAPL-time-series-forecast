# 📈 AAPL Stock Return Forecasting (ML Time Series Project)

## 🔥 Overview
Forecasting next-day Apple (AAPL) stock returns using machine learning with engineered technical indicators and walk-forward validation.

---

## 🎯 Goal
Predict short-term stock returns using historical price data and evaluate real-world predictive power.

---

## ⚙️ Approach
- Feature Engineering: lags, moving averages, volatility
- Models: Linear Regression, XGBoost
- Validation: Walk-forward (time-series correct split)
- Forecasting: Recursive multi-step prediction

---

## 📊 Results

| Model              | MAE     | RMSE    |
|-------------------|---------|---------|
| Linear Regression | 0.0087  | 0.0123  |
| XGBoost           | 0.0107  | 0.0149  |


---

## 📊 Key Insights

* Stock returns are highly unpredictable, with near-zero correlation between predicted and actual values, confirming weak short-term directional signal.
* Linear Regression performed better than XGBoost in MAE, showing that simple linear relationships capture most of the useful signal in engineered features.
* XGBoost captured more variability but introduced noise, leading to higher error despite being a more complex model.
* Low MAE (~0.8–1%) does not imply profitability, as even small prediction errors can lead to incorrect directional trading decisions.
* Walk-forward validation provides a realistic evaluation of model performance by simulating real-time prediction conditions and avoiding data leakage.
---
```
## 📁 Project Structure

aapl-time-series-forecast/
│
├── data/
│ └── aapl.csv
│
├── notebooks/
│ └── analysis.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── split.py
│ ├── train.py
│ └── forecast.py
│
├── outputs/
│ ├── predictions.csv
│ └── metrics.json
│
├── app/
│ └── streamlit_app.py
│
├── requirements.txt
└── README.md

```
---

## 🛠️ Tech Stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · Streamlit

---

## 🚀 Output
- predictions.csv
- metrics.json
- Streamlit dashboard (interactive forecasting tool)
