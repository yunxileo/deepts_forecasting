"""
Implementations of Convolutional Component, Recurrent Component,
Recurrent-skip Component, Temporal Attention Layer and Autoregressive Component.
"""

import torch
from torch import nn


class Convolutional(nn.Module):
    """
    Examples
    >>> x = torch.rand(32, 1, 14, 10)
    >>> cov = Convolutional(62, 2, 10)
    >>> out = cov(x)
    >>> print(out.shape)
    """

    def __init__(
        self, out_channels: int = None, kernel_size: int = None, input_size: int = None
    ) -> None:
        super(Convolutional, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(kernel_size, input_size),
            stride=1,
        )

    def forward(self, x):
        return self.conv(x)


class Recurrent(nn.Module):
    def __init__(
        self, input_size: int = None, hidden_size: int = None, rnn_type: str = "GRU"
    ) -> None:
        super(Recurrent, self).__init__()
        if rnn_type == "GRU":
            self.rnn = nn.RNN(input_size, hidden_size)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size)
        else:
            raise ValueError(f"{rnn_type} is not supported, please check your input!")

    def forward(self, x):
        return self.rnn(x)


class RecurrentSkip(nn.Module):
    def __init__(
        self, input_size: int = None, hidden_size: int = None, rnn_type: str = "GRU"
    ) -> None:
        """

        Args:
            input_size (int):
        """
        super(RecurrentSkip, self).__init__()
        if rnn_type == "GRU":
            self.rnn = nn.RNN(input_size, hidden_size)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size)
        else:
            raise ValueError(f"{rnn_type} is not supported, please check your input!")

    def forward(self, x):
        return self.rnn(x)


class Autoregressive(nn.Module):
    def __init__(self, window_length: int = None) -> None:
        super(Autoregressive, self).__init__()
        self.ar = nn.Linear(window_length, 1)

    def forward(self, x):
        return self.ar(x)
