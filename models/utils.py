import torch
import torch.nn as nn

# The ConditionalModel acts as a switch or a gate for another PyTorch model/module.
class ConditionalModel(nn.Module):
    def __init__(self, model, state):
        super(ConditionalModel, self).__init__()
        self.model = model
        self.state = state

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs) if self.state else None
