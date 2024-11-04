import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start, end, interval = '1d')
    return data
