import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="AAPL Forecasting", layout="centered")

st.title("📈 AAPL Stock Return Forecasting")
st.write("Simple ML model to predict next-day stock returns using Linear Regression and XGBoost.")

df = pd.read_csv("outputs/predictions.csv")

with open('outputs/metrics.json', "r") as f:
    metrics = json.load(f)


st.subheader("📊 Model Performance")

st.write("**Linear Regression**")
st.write(f"MAE: {metrics['lr_mae']}")
st.write(f"RMSE: {metrics['lr_rmse']}")

st.write("**XGBoost**")
st.write(f"MAE: {metrics['xgb_mae']}")
st.write(f"RMSE: {metrics['xgb_rmse']}")


st.subheader("📄 Data Preview")
st.dataframe(df.head())

st.subheader("📈 Actual vs Predicted")

fig, ax = plt.subplots()

ax.plot(df["actual"], label="Actual")
ax.plot(df["lr_pred"], label="Linear Regression")
ax.plot(df["xgb_pred"], label="XGBoost")

ax.legend()
st.pyplot(fig)

# -----------------------
# INSIGHT
# -----------------------
st.subheader("🧠 Insight")

st.write("""
The models show that stock returns are highly noisy and difficult to predict.
Linear Regression performs slightly better than XGBoost in this case.
""")
