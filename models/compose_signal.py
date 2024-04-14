from torch import nn, Tensor
import torch
import torch.nn.functional as F
from typing import Tuple

from .attention import Attention
from omegaconf import DictConfig
from .utils import ConditionalModel


class ConvNetPath(nn.Module):
    """three size of kernal for conv net for one path"""

    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        kernel_sizes: Tuple[int, int, int],
        stride: int = 1,
        config: DictConfig = None,
    ):
        super(ConvNetPath, self).__init__()

        small_gate = nn.Sequential(
            # nn.Conv1d(in_channels, 64, kernel_sizes[0], stride, padding="same"),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.Conv1d(in_channels, 1,  1, stride, padding="same"),
            # nn.BatchNorm1d(1),
            # nn.ReLU(),
            nn.Conv1d(in_channels, intermediate_channels, kernel_sizes[0], stride, padding="same"),
            nn.BatchNorm1d(intermediate_channels),
            nn.ReLU(),
            nn.Conv1d(intermediate_channels, 1, 3, stride, padding="same"),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            # nn.Conv1d(32, out_channels, kernel_sizes[0], stride, padding="same"),
            # nn.BatchNorm1d(out_channels),
            # nn.ReLU(),
        )

        medium_gate = nn.Sequential(
            # nn.Conv1d(in_channels, 64, kernel_sizes[1], stride, padding="same"),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.Conv1d(in_channels, 1,  1, stride, padding="same"),
            # nn.BatchNorm1d(1),
            # nn.ReLU(),
            nn.Conv1d(in_channels, intermediate_channels, kernel_sizes[1], stride, padding="same"),
            nn.BatchNorm1d(intermediate_channels),
            nn.ReLU(),
            nn.Conv1d(intermediate_channels, 1, 3, stride, padding="same"),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            # nn.Conv1d(32, out_channels, kernel_sizes[1], stride, padding="same"),
            # nn.BatchNorm1d(out_channels),
            # nn.ReLU(),
        )

        large_gate = nn.Sequential(
            # nn.Conv1d(in_channels, 64, kernel_sizes[2], stride, padding="same"),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.Conv1d(in_channels, 1,  1, stride, padding="same"),
            # nn.BatchNorm1d(1),
            # nn.ReLU(),
            nn.Conv1d(in_channels, intermediate_channels, kernel_sizes[2], stride, padding="same"),
            nn.BatchNorm1d(intermediate_channels),
            nn.ReLU(),
            nn.Conv1d(intermediate_channels, 1, 3, stride, padding="same"),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            # nn.Conv1d(32, out_channels, kernel_sizes[2], stride, padding="same"),
            # nn.BatchNorm1d(out_channels),
            # nn.ReLU(),
        )

        self.small_gate = ConditionalModel(small_gate, config.small_gate)
        self.medium_gate = ConditionalModel(medium_gate, config.medium_gate)
        self.large_gate = ConditionalModel(large_gate, config.large_gate)
        self.out_dim = config.small_gate + config.medium_gate + config.large_gate

    def forward(self, x: Tensor):
        x1 = self.small_gate(x)
        x2 = self.medium_gate(x)
        x3 = self.large_gate(x)
        return [i for i in (x1, x2, x3) if i != None]


class CompositeSignalGeneratorModule(nn.Module):
    """generate composite signal for each path"""

    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        kernel_sizes: Tuple[int, int, int],
        stride: int = 1,
        config: DictConfig = None,
    ):
        super(CompositeSignalGeneratorModule, self).__init__()
        self.conv_net_path_ppg = ConvNetPath(
            in_channels,
            intermediate_channels,
            kernel_sizes,
            stride,
            config.inception_module.raw_signal,
        )
        self.conv_net_path_d = ConvNetPath(
            in_channels,
            intermediate_channels,
            kernel_sizes,
            stride,
            config.inception_module.first_derivative,
        )
        self.conv_net_path_dd = ConvNetPath(
            in_channels,
            intermediate_channels,
            kernel_sizes,
            stride,
            config.inception_module.second_derivative,
        )

        attn_out_dim = (
            self.conv_net_path_ppg.out_dim
            + self.conv_net_path_d.out_dim
            + self.conv_net_path_dd.out_dim
        )
        attn = Attention(output_dim=attn_out_dim)
        self.attention_subnet = ConditionalModel(attn, config.attention_module)

    def forward(self, x: Tensor):
        x_raw = self.conv_net_path_ppg(x[:, 0])
        x_first = self.conv_net_path_d(x[:, 1])
        x_second = self.conv_net_path_dd(x[:, 2])
        xx = torch.stack([*x_raw, *x_first, *x_second], dim=1)
        w = self.attention_subnet(x[:, 0])
        if w == None:
            w = torch.ones((xx.shape[0], self.attention_subnet.model.out_dim))
        
        weighted_tensor = xx * w.unsqueeze(-1).unsqueeze(-1)
        output = weighted_tensor.sum(dim=1) 

        # print("Weights: ", w)

        # output = xx.squeeze(2)

        return output
