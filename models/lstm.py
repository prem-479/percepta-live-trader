"""
models/lstm.py
LSTM model for sequence prediction
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM-based binary classifier for price direction prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 2)  # Binary classification
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            logits: (batch_size, 2)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and classifier
        out = self.dropout(last_output)
        logits = self.fc(out)
        
        return logits


class LSTMRegressor(nn.Module):
    """LSTM-based regressor for price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMRegressor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)  # Regression output
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            predictions: (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        predictions = self.fc(out)
        return predictions