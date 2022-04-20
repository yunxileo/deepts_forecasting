import torch
from deepts_forecasting.models.base_model import BaseModel
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


# Cell
class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        c_out: int = 1,
        n_head: int = 1,
        dim_feedforward=128,
        dropout=0.1,
        activation="relu",
        n_layers=1,
    ):
        """
        Args:
            d_model: the number of features (aka variables, dimensions, channels) in the time series dataset
            c_out: the number of target classes
            d_model: total dimension of the model.
            n_head:  parallel attention heads.
            dim_feedforward: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            n_layers: the number of sub-encoder-layers in the encoder.

        Examples::
            >>> transformer_model = TransformerModel(d_model=512, n_head=8, n_layers=6)
            >>> src = torch.rand((10, 32, 512))
            >>> out = transformer_model(src)
        """
        super(TransformerModel, self).__init__()
        # TransformerEncoderLayer is made up of self-attn and feedforward network
        encoder_layer = TransformerEncoderLayer(
            d_model,
            n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )  # (batch_size,seq_length,d_model)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, n_layers, norm=encoder_norm
        )  # (batch_size,seq_length,d_model)
        self.outlinear = nn.Linear(d_model, c_out)  # (batch_size,seq_length,c_out)

    def forward(self, x):
        x = self.transformer_encoder(x)
        out = self.outlinear(x)
        return out
