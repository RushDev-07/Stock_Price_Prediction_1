import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start):
    data = yf.download(ticker, start, interval = '1d')
    return data
