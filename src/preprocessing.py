import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    data = data.fillna(method='ffill').fillna(method='bfill')
    data.columns = ['Open', 'High', 'Low', 'Close','Adj Close', 'Volume']
    scaler = MinMaxScaler()
    data['Volume'] = scaler.fit_transform(data[['Volume']])
    return data
