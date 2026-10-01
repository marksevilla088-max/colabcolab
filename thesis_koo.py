# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

st.set_page_config(page_title="Thesis - Sales Forecasting", layout="wide")
st.title("📊 Sales Forecasting Thesis")
st.write("Sta. Cruz, Laguna Sales Forecasting (2020–2024)")


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(
        base_dir,
        "Sta_Cruz_Laguna_Sales_Forecasting_Dataset_2020_2024_v2.xlsx"
    )
    df = pd.read_excel(file_path, sheet_name="Sales_Records")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


df = load_data()

st.subheader("1. Raw Data")
st.dataframe(df.head())
st.write(f"Shape: {df.shape}")


# =========================
# DAILY SALES
# =========================
daily_sales = (
    df.groupby("Date")["Quantity_Sold"]
    .sum()
    .reset_index()
    .rename(columns={"Quantity_Sold": "Daily_Demand"})
)

st.subheader("2. Daily Sales Demand Trend")
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(daily_sales["Date"], daily_sales["Daily_Demand"])
ax.set_title("Daily Sales Demand Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Quantity Sold")
ax.grid()
st.pyplot(fig)

st.write(daily_sales["Daily_Demand"].describe())

daily_sales2 = df.groupby("Day_Type")["Quantity_Sold"].mean()
st.write("Average by Day Type:")
st.dataframe(daily_sales2)

monthly_sales = df.groupby("Month")["Quantity_Sold"].sum()
st.write("Monthly Sales:")
st.dataframe(monthly_sales)


# =========================
# MONTHLY TREND
# =========================
daily_sales = daily_sales.set_index("Date")
daily_sales.index = pd.DatetimeIndex(daily_sales.index, freq="D")
monthly = daily_sales.resample("M").sum()

st.subheader("3. Monthly Sales Demand Trend")
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(monthly.index, monthly["Daily_Demand"])
ax.set_title("Monthly Sales Demand Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Demand")
ax.grid()
st.pyplot(fig)


# =========================
# TRAIN / TEST SPLIT
# =========================
train_size = int(len(daily_sales) * 0.8)
train = daily_sales.iloc[:train_size]
test = daily_sales.iloc[train_size:]

st.subheader("4. Train / Test Split")
st.write(f"Training Data: {train.shape}")
st.write(f"Testing Data: {test.shape}")


# =========================
# ARIMA
# =========================
st.subheader("5. ARIMA Forecast")
with st.spinner("Training ARIMA model..."):
    arima_model = ARIMA(train["Daily_Demand"], order=(5, 1, 2))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=len(test))

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(test.index, test["Daily_Demand"], label="Actual")
ax.plot(test.index, arima_forecast, label="ARIMA Forecast")
ax.set_title("ARIMA Forecast vs Actual Sales Demand")
ax.set_xlabel("Date")
ax.set_ylabel("Demand")
ax.legend()
ax.grid()
st.pyplot(fig)

arima_mae = mean_absolute_error(test["Daily_Demand"], arima_forecast)
arima_mape = (
    abs((test["Daily_Demand"] - arima_forecast) / test["Daily_Demand"]).mean()
) * 100
st.write(f"**ARIMA MAE:** {arima_mae:.2f}")
st.write(f"**ARIMA MAPE:** {arima_mape:.2f}%")


# =========================
# SARIMA
# =========================
st.subheader("6. SARIMA Forecast")
with st.spinner("Training SARIMA model..."):
    sarima_model = SARIMAX(
        train["Daily_Demand"],
        order=(5, 1, 2),
        seasonal_order=(1, 1, 1, 7)
    )
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.forecast(steps=len(test))

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(test.index, test["Daily_Demand"], label="Actual")
ax.plot(test.index, sarima_forecast, label="SARIMA Forecast")
ax.set_title("SARIMA Forecast vs Actual")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Demand")
ax.legend()
ax.grid()
st.pyplot(fig)

sarima_mae = mean_absolute_error(test["Daily_Demand"], sarima_forecast)
sarima_mape = (
    abs((test["Daily_Demand"] - sarima_forecast) / test["Daily_Demand"]).mean()
) * 100
st.write(f"**SARIMA MAE:** {sarima_mae:.2f}")
st.write(f"**SARIMA MAPE:** {sarima_mape:.2f}%")


# =========================
# PROPHET
# =========================
st.subheader("7. Prophet Forecast")
prophet_data = daily_sales.reset_index()
prophet_data.columns = ["ds", "y"]

prophet_train = prophet_data.iloc[:1461]
prophet_test = prophet_data.iloc[1461:]

with st.spinner("Training Prophet model..."):
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    prophet_model.fit(prophet_train)
    future = prophet_model.make_future_dataframe(periods=len(prophet_test))
    prophet_prediction = prophet_model.predict(future)

prophet_forecast = prophet_prediction.iloc[-len(prophet_test):]

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(prophet_test["ds"], prophet_test["y"], label="Actual")
ax.plot(prophet_forecast["ds"], prophet_forecast["yhat"], label="Prophet Forecast")
ax.set_title("Prophet Forecast vs Actual Sales Demand")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Demand")
ax.legend()
ax.grid()
st.pyplot(fig)

prophet_mae = mean_absolute_error(prophet_test["y"], prophet_forecast["yhat"])
prophet_mape = (
    abs((prophet_test["y"] - prophet_forecast["yhat"]) / prophet_test["y"]).mean()
) * 100
st.write(f"**Prophet MAE:** {prophet_mae:.2f}")
st.write(f"**Prophet MAPE:** {prophet_mape:.2f}%")


# =========================
# RANDOM FOREST
# =========================
st.subheader("8. Random Forest Forecast")
rf_data = daily_sales.reset_index()
rf_data["Year"] = rf_data["Date"].dt.year
rf_data["Month"] = rf_data["Date"].dt.month
rf_data["Day"] = rf_data["Date"].dt.day
rf_data["Day_of_Week"] = rf_data["Date"].dt.dayofweek
rf_data["Is_Weekend"] = rf_data["Day_of_Week"].isin([5, 6]).astype(int)
rf_data["Lag_1"] = rf_data["Daily_Demand"].shift(1)
rf_data["Lag_7"] = rf_data["Daily_Demand"].shift(7)
rf_data["Rolling_7"] = rf_data["Daily_Demand"].rolling(window=7).mean()
rf_data = rf_data.dropna()

features = [
    "Year", "Month", "Day", "Day_of_Week",
    "Is_Weekend", "Lag_1", "Lag_7", "Rolling_7"
]

train_rf = rf_data[rf_data["Date"] < "2024-01-01"]
test_rf = rf_data[rf_data["Date"] >= "2024-01-01"]

X_train = train_rf[features]
y_train = train_rf["Daily_Demand"]
X_test = test_rf[features]
y_test = test_rf["Daily_Demand"]

with st.spinner("Training Random Forest model..."):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_prediction = rf_model.predict(X_test)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(test_rf["Date"], y_test, label="Actual")
ax.plot(test_rf["Date"], rf_prediction, label="Random Forest Forecast")
ax.set_title("Random Forest Forecast vs Actual Sales Demand")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Demand")
ax.legend()
ax.grid()
st.pyplot(fig)

rf_mae = mean_absolute_error(y_test, rf_prediction)
rf_mape = (abs((y_test - rf_prediction) / y_test).mean()) * 100
st.write(f"**Random Forest MAE:** {rf_mae:.2f}")
st.write(f"**Random Forest MAPE:** {rf_mape:.2f}%")


# =========================
# SUMMARY
# =========================
st.subheader("9. Model Comparison (MAE)")
summary = pd.DataFrame({
    "Model": ["ARIMA", "SARIMA", "Prophet", "Random Forest"],
    "MAE": [arima_mae, sarima_mae, prophet_mae, rf_mae],
    "MAPE (%)": [arima_mape, sarima_mape, prophet_mape, rf_mape]
})
st.dataframe(summary)
