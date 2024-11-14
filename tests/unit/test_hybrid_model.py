import pytest
from src.hybrid_model import HybridModel
import numpy as np

def test_hybrid_model_initialization():
    model = HybridModel(input_size=10, hidden_size=50, num_layers=2, dropout_rate=0.2, epochs=1, batch_size=32)
    assert model is not None, "Hybrid model should initialize successfully"

def test_hybrid_model_training():
    train_features = np.random.rand(100, 10)
    train_target = np.random.rand(100, 1)
    model = HybridModel(input_size=10, hidden_size=50, num_layers=2, dropout_rate=0.2, epochs=1, batch_size=32)
    model.train(train_features, train_target)
