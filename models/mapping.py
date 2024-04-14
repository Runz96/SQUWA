import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Mapping(nn.Module):
    """Mapping subnet"""
    def __init__(self, input_dim: int = 1, output_dim: int = 20):
        super(Mapping, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(64 * 16, output_dim)
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.seq(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
