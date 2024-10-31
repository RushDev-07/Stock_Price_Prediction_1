import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start, end,interval):
    data = yf.download(ticker, start="2014-01-01", end="2024-01-01", interval='1d')
    return data
