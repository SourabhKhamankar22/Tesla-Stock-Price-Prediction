# 🚀 Tesla Stock Price Prediction using Deep Learning

## 📌 Project Overview

This project predicts Tesla (TSLA) stock closing prices using Deep Learning models. Historical Tesla stock data was analyzed through Exploratory Data Analysis (EDA), preprocessed, transformed into time-series sequences, and used to train multiple neural network architectures.

Three Deep Learning models were developed and compared:

* SimpleRNN
* LSTM (Long Short-Term Memory)
* GRU (Gated Recurrent Unit)

After evaluation, the GRU model achieved the best performance and was selected for deployment using Streamlit.

---

# 🎯 Problem Statement

Stock markets are highly volatile and influenced by numerous factors including investor sentiment, economic conditions, company performance, and global events.

The goal of this project is to:

* Analyze Tesla stock market data
* Identify trends and patterns
* Predict future Tesla closing prices
* Compare Deep Learning architectures
* Deploy an interactive forecasting application

---

# 🏢 Business Applications

This solution can be useful for:

### Investment Firms

Forecast future stock movements for portfolio optimization.

### Traders

Generate buy/sell decision support.

### Financial Analysts

Analyze market trends and stock behavior.

### Research

Study Deep Learning applications in financial forecasting.

### FinTech Applications

Integrate predictive analytics into trading platforms.

---

# 📂 Dataset Information

Dataset: Tesla Historical Stock Market Data

Features:

| Feature   | Description            |
| --------- | ---------------------- |
| Date      | Trading Date           |
| Open      | Opening Price          |
| High      | Highest Price          |
| Low       | Lowest Price           |
| Close     | Closing Price          |
| Adj Close | Adjusted Closing Price |
| Volume    | Trading Volume         |

Target Variable:

Close Price

Dataset Size:

* 2416 Records
* Daily Historical Tesla Stock Data

---

# 🔍 Exploratory Data Analysis (EDA)

Several visualizations were performed to understand stock behavior.

## 1. Tesla Closing Price Trend

Observation:

* Strong long-term growth trend.
* Significant market volatility.
* Sharp price increases after 2013.

## 2. Trading Volume Analysis

Observation:

* Trading volume spikes correspond to major market events.
* Increased investor activity during high-growth periods.

## 3. Distribution Analysis

Observation:

* Stock prices are positively skewed.
* Presence of high-value observations.

## 4. Moving Average Analysis

Indicators Used:

* 30-Day Moving Average
* 100-Day Moving Average

Observation:

* Moving averages smooth noise.
* Help identify long-term trends.

## 5. Correlation Analysis

Observation:

* Open, High, Low, Close are highly correlated.
* Volume has weaker correlation with prices.

---

# ⚙️ Data Preprocessing

## Data Cleaning

Performed:

* Missing value detection
* Duplicate value detection
* Data type conversion
* Date formatting

Results:

* No duplicate records found.
* No major missing values in original dataset.

## Missing Value Handling

Generated missing values occurred due to:

* MA30
* MA100
* Daily_Return

Strategy Used:

dropna()

Reason:

Initial rows lacked sufficient historical observations.

## Outlier Handling

Outliers were retained.

Reason:

In financial time-series forecasting, extreme price movements contain valuable market information.

---

# 🛠 Feature Engineering

Additional features created:

### Daily Return

Daily_Return = Percentage Change in Closing Price

### Moving Averages

MA30 = 30-Day Moving Average

MA100 = 100-Day Moving Average

### Time-Series Sequence Generation

Sliding Window Approach:

Sequence Length = 60 Days

Meaning:

Previous 60 trading days are used to predict the next trading day.

---

# 🤖 Deep Learning Models

## 1. SimpleRNN

Architecture:

* SimpleRNN Layer
* Dropout Layer
* Dense Layer

Advantages:

* Simple architecture
* Fast training

Limitations:

* Vanishing gradient problem
* Poor long-term memory

---

## 2. LSTM

Architecture:

* LSTM Layer
* Dropout Layer
* Dense Layer

Advantages:

* Captures long-term dependencies
* Better memory retention

---

## 3. GRU

Architecture:

* GRU Layer
* Dropout Layer
* Dense Layer

Advantages:

* Faster than LSTM
* Fewer parameters
* Better computational efficiency

---

# 🔧 Hyperparameter Tuning

Technique:

Random Search

Parameters Tuned:

* Number of Units
* Dropout Rate

Best Configuration:

Units = 32

Dropout = 0.1

---

# 📊 Model Evaluation Metrics

Metrics Used:

### MAE

Mean Absolute Error

Measures average prediction error.

### RMSE

Root Mean Squared Error

Penalizes large errors.

### R² Score

Measures explained variance.

Higher is better.

---

# 📈 Model Comparison

| Model     | MAE    | RMSE   | R² Score |
| --------- | ------ | ------ | -------- |
| SimpleRNN | 11.94  | 18.30  | 0.9389   |
| LSTM      | 0.0130 | 0.0198 | 0.9591   |
| GRU       | 0.0105 | 0.0173 | 0.9686   |

🏆 Best Model: GRU

Reasons:

* Lowest MAE
* Lowest RMSE
* Highest R² Score
* Stable training performance
* Faster computation

---

# 🔍 Key Insights

### Insight 1

Tesla stock demonstrates strong long-term upward growth.

### Insight 2

Trading volume increases during major market events.

### Insight 3

Moving averages effectively capture market trends.

### Insight 4

Deep Learning significantly improves forecasting capability.

### Insight 5

GRU outperforms both SimpleRNN and LSTM.

### Insight 6

Historical stock behavior contains sufficient temporal patterns for future forecasting.

---

# 🚀 Streamlit Deployment

The final GRU model was deployed using Streamlit.

Features:

✅ Dataset Preview

✅ Model Comparison

✅ Historical Price Visualization

✅ Next Day Forecast

✅ Next 5 Days Forecast

✅ Next 10 Days Forecast

✅ Forecast Visualization

---

# 🖥 Streamlit Screens

## Home Page

Displays:

* Project Summary
* Best Model
* Evaluation Metrics

## Model Comparison Section

Shows:

* MAE
* RMSE
* R² Score

for all models.

## Forecast Module

Allows users to:

* Select forecast horizon
* Generate predictions
* Visualize future trends

---

# 📁 Project Structure

Tesla-Stock-Prediction/

│

├── data/

│ └── Tesla.csv

│

├── notebooks/

│ └── Tesla_Stock_Prediction.ipynb

│

├── models/

│ ├── rnn_model.h5

│ ├── lstm_model.h5

│ └── gru_model.h5

│

├── streamlit_app.py

├── requirements.txt

├── README.md

└── assets/

---

# 💻 Installation

Clone Repository

git clone <https://github.com/SourabhKhamankar22/Tesla-Stock-Price-Prediction>

cd Tesla-Stock-Prediction

Install Dependencies

pip install -r requirements.txt

Run Streamlit

streamlit run streamlit_app.py

---

# 📦 Requirements

Python 3.10+

TensorFlow

NumPy

Pandas

Matplotlib

Seaborn

Scikit-Learn

Streamlit

Keras

---

# Future Enhancements

* Real-Time Yahoo Finance API Integration
* News Sentiment Analysis
* Twitter/X Sentiment Analysis
* Transformer Models
* Multi-Stock Forecasting
* Portfolio Prediction System

---

# Conclusion

This project successfully demonstrates the use of Deep Learning for stock price forecasting. Multiple neural network architectures were compared, and GRU achieved the best predictive performance with an R² Score of 0.9686.

The deployed Streamlit application provides an interactive interface for forecasting Tesla stock prices and demonstrates the practical application of Deep Learning in financial analytics.
