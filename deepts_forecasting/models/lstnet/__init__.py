"""
implementations of complete structure of LSTNet.
"""
from typing import Union

import pandas as pd
import torch
import torch.nn.functional as F
from deepts_forecasting.models.lstnet.sub_modules import (
    Autoregressive,
    Convolutional,
    Recurrent,
    RecurrentSkip,
)
from torch import nn


class LSTNet(nn.Module):
    """
    >>> x = torch.rand(64, 28, 10)
    >>> model = LSTNet()
    >>> out = model(x)
    """

    def __init__(
        self,
        input_length: int = 28,
        input_size: int = 10,
        kernel_size: int = 7,
        conv_out_size: int = 20,
        hidden_size_rnn: int = 50,
        hidden_size_skip: int = 7,
        ar_window_length: int = 7,
        skip_steps: int = 7,
        num_heads: int = 1,
        short_mem_type: str = "GRU",
        long_mem_type: str = "GRU",
        dropout: float = 0.1,
        device: Union[str, torch.device] = "cpu",
    ):
        """

        Args:
            input_length (int): length of input time series steps
            input_size (int): number of input time series
            kernel_size (int): one of dimensions of CNN filter while another dimension is fixed as `input_size`
            conv_out_size (int): size of output of CNN layer
            hidden_size_rnn (int): size of hidden state of RNN
            hidden_size_skip (int): size of hidden state of skip RNN
            ar_window_length (int): length of window of Autoregressive Component
            skip_steps (int): skip steps of skip RNN
            num_heads (int): number of heads of multihead-attention layer
            short_mem_type (str): type of short-term memory. GRU/LSTM
            long_mem_type (str): type of long-term-memory. GRU/LSTM/Attention
            dropout (float): dropout
        """
        if ar_window_length > input_length:
            raise ValueError("ar_window_length cannot be larger than input_length!")

        super(LSTNet, self).__init__()
        self.input_length = input_length
        self.input_size = input_size
        self.conv_out_size = conv_out_size
        self.hidden_size_skip = hidden_size_skip
        self.ar_window_length = ar_window_length
        self.short_mem_type = short_mem_type
        self.long_mem_type = long_mem_type
        self.skip_times = int((input_length - kernel_size) / skip_steps)
        self.skip_steps = skip_steps
        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Convolutional Component
        self.conv = Convolutional(
            out_channels=conv_out_size, kernel_size=kernel_size, input_size=input_size
        )

        # Recurrent Component
        self.rnn = Recurrent(
            input_size=conv_out_size,
            hidden_size=hidden_size_rnn,
            rnn_type=short_mem_type,
        )

        # Recurrent-skip Component or Temporal Attention Layer
        if long_mem_type == "GRU" or long_mem_type == "LSTM":
            assert skip_steps >= 0, "skip_steps cannot be smaller than zero!"
            self.rnn_skip = RecurrentSkip(
                input_size=conv_out_size,
                hidden_size=hidden_size_skip,
                rnn_type=long_mem_type,
            )
            self.linear = nn.Linear(
                in_features=hidden_size_rnn + skip_steps * hidden_size_skip,
                out_features=self.input_size,
            )
        elif long_mem_type == "Attention":
            if num_heads < 0:
                raise ValueError("num_heads cannot be smaller than zero!")
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size_rnn, num_heads=num_heads
            )
            self.linear = nn.Linear(
                in_features=hidden_size_rnn * 2, out_features=input_size
            )
        else:
            raise ValueError(
                "Only GRU, LSTM, and Attention mechanism are supported for long-term memory."
            )

        # Autoregressive Component

        self.ar = Autoregressive(window_length=ar_window_length)

    def forward(self, x: pd.DataFrame):
        """

        Args:
            x (pd.DataFrame): input [batch_size, input_length, input_size]
        """
        batch_size = x.size(0)

        # CNN
        # reshape tensor to [batch_size, channel, input_seq_length, num_features]
        c = x.view(-1, 1, self.input_length, self.input_size)
        c = self.conv(c)
        c = F.relu(c)
        # output size of conv is [batch_size, num_filter, H, 1]
        # H = input_length - kernel_size + 1
        c = self.dropout(c)
        # squeeze the 3-rd dim
        c = torch.squeeze(c, 3)  # [batch_size, num_filter, H]

        # Short-term memory: RNN
        r = c.permute(2, 0, 1).contiguous()  # reshape c [H, batch, num_filter]
        if self.short_mem_type == "GRU":
            output, h_n = self.rnn(r)
        else:
            output, (h_n, c_n) = self.rnn(r)
        h_n = self.dropout(torch.squeeze(h_n, 0))  # [batch_size, hidden_size_rnn]

        # Long-term memory: SKIP-RNN/Attention
        s = c[:, :, int(-self.skip_times * self.skip_steps) :].contiguous()
        s = s.view(batch_size, self.conv_out_size, self.skip_times, self.skip_steps)
        s = s.permute(2, 0, 3, 1).contiguous()
        s = s.view(self.skip_times, batch_size * self.skip_steps, self.conv_out_size)
        if self.long_mem_type == "GRU":
            _, skip_hn = self.rnn_skip(s)
            skip_hn = skip_hn.view(batch_size, self.skip_steps * self.hidden_size_skip)
            skip_hn = self.dropout(skip_hn)
            r = torch.cat((h_n, skip_hn), axis=1)
        elif self.long_mem_type == "LSTM":
            _, (skip_hn, skip_cn) = self.rnn_skip(s)
            skip_hn = skip_hn.view(batch_size, self.skip_steps * self.hidden_size_skip)
            skip_hn = self.dropout(skip_hn)
            r = torch.cat((h_n, skip_hn), axis=1)
        elif self.long_mem_type == "Attention":
            a, _ = self.attention(output, output, output)
            a = a.permute(1, 0, 2)[:, -1, :]
            r = torch.cat((a, h_n), 1)
        else:
            raise (ValueError("long_mem_type not supported"))

        # combine output of RNN layer and skip-RNN/Attention layer
        res = self.linear(r)

        # Autoregressive
        z = x[:, -self.ar_window_length :, :]
        z = z.permute(0, 2, 1).contiguous().view(-1, self.ar_window_length)
        z = self.ar(z)
        z = z.view(-1, self.input_size)
        res = res + z

        return res
