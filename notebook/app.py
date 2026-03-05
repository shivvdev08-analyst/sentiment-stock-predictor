import streamlit as st
from transformers import pipeline
import pickle
import numpy as np
import yfinance as yf

st.set_page_config(page_title="AI Stock Predictor", layout="centered")

st.title("📊 AI Stock Prediction using Sentiment")

st.write("Analyze trader tweets and predict possible stock movement.")

# load models
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
model = pickle.load(open("final_stock_model.pkl", "rb"))

# stock selection
ticker = st.text_input("Enter Stock Symbol (Example: TSLA, AAPL, TATASTEEL.NS)", "TSLA")

# get stock data
stock = yf.Ticker(ticker)
data = stock.history(period="5d")

if not data.empty:
    
    current_price = data["Close"][-1]

    st.subheader("Live Stock Price")
    st.write(f"{ticker} Current Price: $", round(current_price,2))

    st.line_chart(data["Close"])

# tweet input
st.subheader("Enter Trader Tweet")

tweet = st.text_input("Example: Tesla stock looks bullish today")

if tweet:

    result = finbert(tweet)[0]

    sentiment = result["label"]
    score = result["score"]

    st.subheader("Sentiment Analysis")

    st.write("Sentiment:", sentiment)
    st.write("Confidence:", round(score,2))

    if sentiment == "positive":
        sentiment_score = 1
    elif sentiment == "negative":
        sentiment_score = -1
    else:
        sentiment_score = 0

    # example feature vector
    features = np.array([[0.002,0.003,-0.001,0.004,120000,
                          0.001,0.002,-0.001,0.001,
                          sentiment_score,
                          0.002,0.003,0.001,0.002,
                          0.0005,0.001,3]])

    prediction = model.predict(features)[0]

    prob = model.predict_proba(features)[0]

    st.subheader("Prediction Confidence")

    st.write("UP Probability:", round(prob[1]*100,2), "%")
    st.write("DOWN Probability:", round(prob[0]*100,2), "%")

    if prediction == 1:
        st.success(f"📈 Stock Likely to Go UP ({round(prob[1]*100,2)}%)")
    else:
        st.error(f"📉 Stock Likely to Go DOWN ({round(prob[0]*100,2)}%)")