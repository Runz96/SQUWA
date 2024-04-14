import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Attention(nn.Module):
    """Attention subnet"""

    def __init__(self, input_dim: int = 1, output_dim: int = 9):
        super(Attention, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv1d(input_dim, 48, kernel_size=119, stride=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=0),
            # nn.Dropout(0.4),

            nn.Conv1d(48, 96, kernel_size=3, stride=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=0),
            # nn.Dropout(0.4),
            # nn.Conv1d(32, 64, 3, 1),
            # nn.BatchNorm1d(64),
            nn.Conv1d(96, 192, kernel_size=3, stride=1),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=0),
            # nn.Dropout(0.4),
            
            nn.Conv1d(192, 384, kernel_size=3, stride=1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=0),
        )

        self.fc = nn.Linear(384, output_dim)
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: Tensor):
        x = self.seq(x)
        x = F.avg_pool1d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x
