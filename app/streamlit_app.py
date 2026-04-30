import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="AAPL Forecasting", layout="centered")

st.title("📈 AAPL Stock Return Forecasting")
st.write("Predict next-day stock returns using Linear Regression and XGBoost.")

# Load data (IMPORTANT: use relative paths for deployment)
df = pd.read_csv("outputs/predictions.csv")

with open("outputs/metrics.json", "r") as f:
    metrics = json.load(f)

# -----------------------
# MODEL SELECTION
# -----------------------
model_choice = st.selectbox(
    "Select Model",
    ["Linear Regression", "XGBoost"]
)

# -----------------------
# METRICS
# -----------------------
st.subheader("📊 Model Performance")

if model_choice == "Linear Regression":
    st.metric("MAE", metrics['lr_mae'])
    st.metric("RMSE", metrics['lr_rmse'])
    pred_col = "lr_pred"
else:
    st.metric("MAE", metrics['xgb_mae'])
    st.metric("RMSE", metrics['xgb_rmse'])
    pred_col = "xgb_pred"

# -----------------------
# DATA PREVIEW
# -----------------------
with st.expander("📄 Data Preview"):
    st.dataframe(df.head(10))

# -----------------------
# CHART
# -----------------------
st.subheader("📈 Actual vs Predicted")

show_actual = st.checkbox("Show Actual Values", value=True)

fig, ax = plt.subplots()

if show_actual:
    ax.plot(df["actual"], label="Actual")

ax.plot(df[pred_col], label=model_choice)

ax.legend()
st.pyplot(fig)

# -----------------------
# INSIGHT
# -----------------------
st.subheader("🧠 Insight")

st.write("""
Stock returns are highly noisy and difficult to predict.
Linear Regression slightly outperforms XGBoost in this setup.
Even with low error metrics, predictive power for trading remains limited.
""")
