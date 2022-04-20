"""
Common layers, modules for time series forecasting neural networks.
"""

import torch
import torch.nn.functional as F
from torch import nn


# periodic activations
def time_activation(tau, f, w, b, w0, b0):
    """

    Args:
        tau (int): time index
        f (def): activation function
        w (torch.Tensor):
        b (torch.Tensor):
        w0 (torch.Tensor):
        b0 (torch.Tensor):

    Returns:
    """
    v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    """
    sine activation.
    """

    def __init__(self, out_features):
        super(SineActivation, self).__init__()

        self.out_features = out_features

        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.w = nn.parameter.Parameter(torch.randn(1, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(1, out_features - 1))

        self.f = torch.sin

    def forward(self, tau):
        return time_activation(tau, self.f, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    """
    cosine activation.
    """

    def __init__(self, out_features):
        super(CosineActivation, self).__init__()

        self.out_features = out_features

        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.w = nn.parameter.Parameter(torch.randn(1, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(1, out_features - 1))

        self.f = torch.cos

    def forward(self, tau):
        return time_activation(tau, self.f, self.w, self.b, self.w0, self.b0)


class T2V(nn.Module):
    def __init__(self, activation: str, hidden_dim: int, embedding_size: int):
        """
        Time embedding layer to capture periodic and non-periodic pattern.
        Args:
            activation (str): type of periodic function ['cos'/'sin']
            hidden_dim (int): dimension of hidden layer which is output of periodic activations
            embedding_size (int): dimension of output of time2vec embedding which is output of linear layer
        """
        super().__init__()

        self.activation = activation
        self.hidden_dim = hidden_dim
        self.embedding_size = embedding_size

        if activation == "sin":
            self.l1 = SineActivation(hidden_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(hidden_dim)
        else:
            raise ValueError(
                f"'{activation}' is not supported, only 'sin' and 'cos' are valid."
            )

        self.fc = nn.Linear(hidden_dim, embedding_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.fc(x)
        return x
