import pandas as pd
from src.config import TEST_SIZE

def split_data(data):
    train_size = int(len(data) * (1 - TEST_SIZE))
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data
