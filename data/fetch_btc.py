import yfinance as yf
import pandas as pd

def fetch_btc_data():
    df = yf.download("BTC-USD", interval="1m", period="1d")
    df = df[['Close']].dropna()
    df.columns = ['price']
    return df
