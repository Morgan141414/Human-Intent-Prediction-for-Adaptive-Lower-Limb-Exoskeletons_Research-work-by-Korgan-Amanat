import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

class RandomForestBaseline:
    """RandomForestBaseline: feature extraction + sklearn RandomForest."""
    def __init__(self, n_estimators=100, random_state=42):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        
    def extract_features(self, X):
        # Extract mean, std, max, min for each channel
        # X shape: (batch, seq_len, channels)
        features = np.hstack([
            np.mean(X, axis=1),
            np.std(X, axis=1),
            np.max(X, axis=1),
            np.min(X, axis=1)
        ])
        return features
        
    def fit(self, X, y):
        features = self.extract_features(X)
        self.rf.fit(features, y)
        
    def predict(self, X):
        features = self.extract_features(X)
        return self.rf.predict(features)

class LSTMModel(nn.Module):
    """LSTMModel: 2-layer LSTM with FC head."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take last time step
        out = self.fc(out)
        return out

class ConvLSTMAttention(nn.Module):
    """ConvLSTMAttention: Conv1D blocks -> Bidirectional LSTM -> Temporal Attention -> FC head."""
    def __init__(self, input_dim, hidden_dim=64, num_classes=5):
        super(ConvLSTMAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2) # Back to (batch, seq_len, features)
        
        lstm_out, _ = self.lstm(x)
        
        # Temporal Attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        out = self.fc(context)
        return out

class TemporalTransformer(nn.Module):
    """TemporalTransformer: Positional encoding -> 3 Transformer layers -> [CLS] token -> FC head."""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, num_classes=5, max_seq_len=200):
        super(TemporalTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_encoder[:, :seq_len + 1, :]
        
        x = self.transformer(x)
        cls_out = x[:, 0, :]
        out = self.fc(cls_out)
        return out
