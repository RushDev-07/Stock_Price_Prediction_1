import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Take output from last time step
        return out

class HybridModel:
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, n_dense_units=25, dropout_rate=0.2, epochs=50, batch_size=32, learning_rate=0.001):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_dense_units = n_dense_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size=1, dropout_rate=dropout_rate).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def preprocess_data(self, features, target, time_steps=60):
        scaled_features = self.scaler.fit_transform(features)
        scaled_target = self.scaler.fit_transform(target.values.reshape(-1, 1))

        X, y = [], []
        for i in range(time_steps, len(scaled_features)):
            X.append(scaled_features[i - time_steps:i])
            y.append(scaled_target[i])
        
        X, y = np.array(X), np.array(y)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def train(self, train_features, train_target):
        train_loader = self.preprocess_data(train_features, train_target)
        best_loss = float('inf')
        patience = 10
        wait = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.view(-1, X_batch.size(1), X_batch.size(2))  # Ensure correct shape: (batch_size, time_steps, input_size)
                
                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}")
            
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered.")
                    break

    def predict(self, test_features):
        self.model.eval()
        test_loader = self.preprocess_data(test_features, test_features.iloc[:, 0])
        predictions = []

        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.view(-1, X_batch.size(1), X_batch.size(2))  # Ensure correct shape
                pred = self.model(X_batch).cpu().numpy()
                predictions.extend(pred)

        predictions = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions).flatten()
