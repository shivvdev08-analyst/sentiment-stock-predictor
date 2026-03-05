# AI Stock Prediction using Sentiment Analysis

This project predicts stock movement using trader sentiment and historical stock data.

## Project Overview

The system analyzes trader tweets and predicts whether a stock is likely to go **UP 📈 or DOWN 📉**.

It combines:

- FinBERT sentiment analysis
- Machine Learning stock prediction
- Yahoo Finance market data
- Streamlit interactive dashboard

## Technologies Used

Python  
Scikit-learn  
XGBoost  
Transformers (FinBERT)  
Streamlit  
Yahoo Finance API (yfinance)

## How the System Works

Trader Tweet  
↓  
Sentiment Analysis (FinBERT)  
↓  
Sentiment Score  
↓  
Stock Prediction Model (XGBoost)  
↓  
Prediction Dashboard (Streamlit)

## How to Run the Project

Clone the repository:

git clone https://github.com/YOUR_USERNAME/sentiment-stock-predictor.git


Go to the project folder:


cd sentiment-stock-predictor


Install dependencies:


pip install -r requirements.txt


Run the application:


streamlit run notebook/app.py


## Example

Input:


Tesla stock looks bullish today


Output:


Sentiment: Positive
Prediction: Stock Likely to Go UP


## Author

Shivam Vinod Chaudhari