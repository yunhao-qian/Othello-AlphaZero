"""Neural network proposed in the AlphaGo Zero paper, extended to support general board
games."""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolutional block at the beginning of the network."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block in the residual tower."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding="same"
        )
        self.norm1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding="same"
        )
        self.norm2 = nn.BatchNorm2d(num_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + skip
        x = self.relu2(x)
        return x


class PolicyHead(nn.Module):
    """Policy head predicting the probability distribution of actions."""

    def __init__(self, in_channels: int, num_grids: int, num_actions: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 2, 1)
        self.norm = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(2 * num_grids, num_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        x = self.softmax(x)
        return x


class ValueHead(nn.Module):
    """Value head predicting the action-value of the current position."""

    def __init__(self, in_channels: int, num_grids: int, hidden_size: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.norm = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(num_grids, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu1(x)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        x = x.squeeze(1)
        x = self.tanh(x)
        return x


class AlphaZeroResNet(nn.Module):
    """Neural network proposed in the AlphaGo Zero paper, extended to support general
    board games."""

    def __init__(
        self,
        in_channels: int,
        num_grids: int,
        num_actions: int,
        feature_channels: int = 256,
        num_residual_blocks: int = 19,
        value_head_hidden_size: int = 256,
    ) -> None:
        super().__init__()

        self.conv_block = ConvBlock(in_channels, feature_channels)
        self.residual_blocks = nn.Sequential(
            *(ResidualBlock(feature_channels) for _ in range(num_residual_blocks))
        )
        self.policy_head = PolicyHead(feature_channels, num_grids, num_actions)
        self.value_head = ValueHead(feature_channels, num_grids, value_head_hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_block(x)
        x = self.residual_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
