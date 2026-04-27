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

* Stock returns are very hard to predict, and the model predictions do not closely match actual values.
* Linear Regression performed slightly better than XGBoost in terms of error, showing that simple models can work well for this problem.
* XGBoost captured more variation in the data but also made more mistakes compared to Linear Regression.
* Even though the error values are low, it does not mean the model can be used for profitable trading.
* Walk-forward validation gave a more realistic evaluation since it simulates how predictions would work in real-time.
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
