import torch
import torch.nn as nn
import torch.nn.functional as F
from deepts_forecasting.models.informer.attention import (
    AttentionLayer,
    FullAttention,
    ProbAttention,
)


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class InformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=128,
        d_ff=512,
        factor=5,
        n_heads=4,
        dropout=0.1,
        activation="relu",
    ):
        """

        Args:
            d_model:
            d_ff:
            factor:
            n_heads:
            dropout:
            activation:

        Examples::
            >>> encoder_layer = InformerEncoderLayer(d_model=128, d_ff=512, n_heads=8)
            >>> src = torch.rand(10, 32, 128)
            >>> out = encoder_layer(src)
        """
        super(InformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        prob_attention = ProbAttention(
            False, factor, attention_dropout=dropout, output_attention=False
        )
        self.attention = AttentionLayer(prob_attention, d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class InformerEncoder(nn.Module):
    def __init__(self, encoder_layer, conv_layer=None, num_layers=1, norm_layer=None):
        """

        Args:
            encoder_layer:
            conv_layer:
            num_layers:
            norm_layer:

        Examples::
            >>> layer = InformerEncoderLayer(d_model=128, d_ff=512, n_heads=8)
            >>> layer_2 = ConvLayer(c_in=128)
            >>> informer_encoder = InformerEncoder(encoder_layer=layer, conv_layer=layer_2, num_layers=2)
            >>> src = torch.rand(10, 32, 128)
            >>> out = informer_encoder(src)
        """
        super(InformerEncoder, self).__init__()
        # self.encoder_layers = _get_clones(encoder_layer, num_layers)
        self.encoder_layers = nn.ModuleList(
            [encoder_layer for num in range(num_layers)]
        )
        self.conv_layers = (
            nn.ModuleList([conv_layer for num in range(num_layers - 1)])
            if conv_layer is not None
            else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for encoder_layer, conv_layer in zip(self.encoder_layers, self.conv_layers):
                x, attn = encoder_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.encoder_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.encoder_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x


class InformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model, d_ff=None, n_heads=4, dropout=0.1, activation="relu", factor=5
    ):
        super(InformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        prob_attention = FullAttention(
            False, factor, attention_dropout=dropout, output_attention=False
        )
        full_attention = FullAttention(
            False, factor, attention_dropout=dropout, output_attention=False
        )
        self.self_attention = AttentionLayer(prob_attention, d_model, n_heads)
        self.cross_attention = AttentionLayer(full_attention, d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        )

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class InformerDecoder(nn.Module):
    def __init__(self, layer, num_layers=1, norm_layer=None):
        super(InformerDecoder, self).__init__()
        self.layers = nn.ModuleList([layer for num in range(num_layers)])
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x
