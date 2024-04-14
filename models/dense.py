import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    def __init__(
        self,
        sqi_dim: int, 
        hidden_dim: int
    ):
        super(Dense, self).__init__()
        self.sqi_dim = sqi_dim
        self.hidden_dim = hidden_dim

        # Linear transformations for query, key, and value
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(1, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(0.4)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.sqi_dim, self.hidden_dim)) # Assuming max sequence length is 1000
      

    def scaled_dot_product_attention(self, query, key, value):
        # Calculate the dot products (scores)
        scores = torch.bmm(query, key.permute(0, 2, 1)) / (self.hidden_dim ** 0.5)
        # Apply softmax to get the weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Multiply by values
        output = torch.bmm(attn_weights, value)
        return output, attn_weights
    
    def forward(self, hidden_states, sqi_sequence):
        # Apply positional encoding
        hidden_states = hidden_states + self.positional_encoding[:, :self.sqi_dim, :]

        # Transform hidden_states and sqi_sequence to query, key, and value
        query = self.query(hidden_states)
        key = self.key(sqi_sequence.permute(0, 2, 1))
        value = self.value(hidden_states)

        # Compute attention
        output, attn_weights = self.scaled_dot_product_attention(query, key, value)
        context_vector = torch.mean(output, dim=1)
        return context_vector, attn_weights
