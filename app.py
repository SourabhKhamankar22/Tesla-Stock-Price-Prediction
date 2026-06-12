import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

df = load_data()

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_gru_model():
    model = load_model(
        "models/best_gru_model.keras",
        compile=False
    )
    return model

model = load_gru_model()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📈 Tesla Stock Predictor")

st.sidebar.success("Best Model: GRU")

st.sidebar.info("""
Forecast Options

• Next Day Prediction

• Next 5 Days Forecast

• Next 10 Days Forecast

Model Performance

R² = 0.9686
""")

# =====================================================
# HEADER
# =====================================================

st.title(" Tesla Stock Price Prediction using GRU")

st.markdown("""
### Deep Learning Based Stock Forecasting

This project predicts Tesla stock closing prices using a
GRU (Gated Recurrent Unit) neural network trained on
historical Tesla stock market data.

The GRU model was selected as the final deployment model
after comparing it with SimpleRNN and LSTM architectures.
""")

# =====================================================
# MODEL COMPARISON
# =====================================================

st.subheader("🏆 Model Comparison")

comparison_df = pd.DataFrame({
    "Model": ["SimpleRNN", "LSTM", "GRU"],
    "MAE": [11.94, 0.0130, 0.0105],
    "RMSE": [18.30, 0.0198, 0.0173],
    "R² Score": [0.9389, 0.9591, 0.9686]
})

st.dataframe(
    comparison_df,
    use_container_width=True
)

st.success(
    "GRU achieved the highest R² Score and lowest prediction error, therefore it was selected as the final model."
)

# =====================================================
# DATA PREPARATION
# =====================================================

close_data = df[["Close"]]

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(close_data)

WINDOW_SIZE = 60

# =====================================================
# DATASET PREVIEW
# =====================================================

with st.expander("📂 View Dataset Preview"):
    st.dataframe(df.tail(10))

# =====================================================
# LATEST PRICE + PREDICTION
# =====================================================

last_60_days = scaled_data[-WINDOW_SIZE:]

X_input = np.array([last_60_days])

next_day_scaled = model.predict(
    X_input,
    verbose=0
)

next_day_price = scaler.inverse_transform(
    next_day_scaled
)[0][0]

latest_price = float(df["Close"].iloc[-1])

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Latest Tesla Close Price",
        f"${latest_price:.2f}"
    )

with col2:
    st.metric(
        "Predicted Next Day Price",
        f"${next_day_price:.2f}"
    )

# =====================================================
# FORECAST FUNCTION
# =====================================================

def forecast_future(model, scaled_series, days):

    temp_input = list(
        scaled_series[-WINDOW_SIZE:].flatten()
    )

    predictions = []

    for _ in range(days):

        x_input = np.array(
            temp_input[-WINDOW_SIZE:]
        )

        x_input = x_input.reshape(
            1,
            WINDOW_SIZE,
            1
        )

        pred = model.predict(
            x_input,
            verbose=0
        )

        predictions.append(pred[0][0])

        temp_input.append(pred[0][0])

    predictions = np.array(
        predictions
    ).reshape(-1, 1)

    return scaler.inverse_transform(
        predictions
    )

# =====================================================
# FORECAST SECTION
# =====================================================

st.subheader("🔮 Future Stock Forecast")

forecast_days = st.selectbox(
    "Select Forecast Horizon",
    [1, 5, 10]
)

future_predictions = forecast_future(
    model,
    scaled_data,
    forecast_days
)

forecast_df = pd.DataFrame({
    "Day": np.arange(
        1,
        forecast_days + 1
    ),
    "Predicted Close Price ($)": np.round(
        future_predictions.flatten(),
        2
    )
})

st.dataframe(
    forecast_df,
    use_container_width=True
)

# =====================================================
# FORECAST CHART
# =====================================================

st.subheader("📊 Forecast Visualization")

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    forecast_df["Day"],
    forecast_df["Predicted Close Price ($)"],
    marker="o"
)

ax.set_title(
    f"Tesla Stock Forecast for Next {forecast_days} Days"
)

ax.set_xlabel("Forecast Day")

ax.set_ylabel("Predicted Close Price ($)")

ax.grid(True)

st.pyplot(fig)

# =====================================================
# HISTORICAL PRICE CHART
# =====================================================

st.subheader("📉 Historical Tesla Stock Prices")

fig2, ax2 = plt.subplots(
    figsize=(12, 5)
)

ax2.plot(
    df["Date"],
    df["Close"]
)

ax2.set_title(
    "Historical Tesla Closing Price Trend"
)

ax2.set_xlabel("Date")

ax2.set_ylabel("Closing Price ($)")

ax2.grid(True)

st.pyplot(fig2)

# =====================================================
# PROJECT SUMMARY
# =====================================================

st.subheader("📄 Project Summary")

st.markdown("""
### Objective

Predict Tesla stock closing prices using Deep Learning models and compare their performance.

### Models Implemented

- SimpleRNN
- LSTM
- GRU

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Best Model

GRU

- MAE = 0.0105
- RMSE = 0.0173
- R² = 0.9686

### Deployment

The deployed application forecasts Tesla stock closing prices for:

- Next 1 Day
- Next 5 Days
- Next 10 Days
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Tesla Stock Price Prediction using Deep Learning | Streamlit Deployment Project"
)