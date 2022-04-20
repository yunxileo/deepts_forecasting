import math
from abc import ABC
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from deepts_forecasting.models.base_model import BaseModelWithCovariates
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

# def _create_transformer_inputs(data):
#     # '_TimeSeriesSequentialDataset' stores time series in the
#     # (batch_size, input_chunk_length, input_size) format. PyTorch's nn.Transformer
#     # module needs it the (input_chunk_length, batch_size, input_size) format.
#     # Therefore, the first two dimensions need to be swapped.
#     src = data.permute(1, 0, 2)
#     tgt = src[-1:, :, :]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, maxlen: int = 500):
        """An implementation of positional encoding as described in 'Attention is All you Need' by Vaswani et al. (2017)

        Args:
            d_model: the number of expected features in the transformer encoder/decoder inputs.
            dropout: Fraction of neurons affected by Dropout (default=0.1).
            maxlen: The dimensionality of the computed positional encoding array

        Examples::
        >>> model = PositionalEncoding(d_model=64, dropout=0.1, maxlen=128)
        >>> src = torch.rand((16, 128, 64))
        >>> out = model(src)

        Note: A full example to apply nn.Transformer module for the word language model is available in
        https://github.com/pytorch/examples/tree/master/word_language_model

        """
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0).transpose(0, 1)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, x):
        # module needs it the (batch_size, seq_length, input_size) format
        x = x + self.pos_embedding[: x.size(0), :]
        return x


class TransformerModel(BaseModelWithCovariates, ABC):
    def __init__(
        self,
        dim_feedforward: int = 128,
        n_head: int = 1,
        n_layers: int = 2,
        d_model: int = 64,
        max_prediction_length: int = None,
        max_encoder_length: int = None,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_categoricals: List[str] = [],
        x_reals: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        output_size: Union[int, List[int]] = 1,
        dropout=0.1,
        activation="relu",
        loss=nn.L1Loss(),
        **kwargs
    ):
        self.embeddings = None
        self.save_hyperparameters()
        # store loss function separately as it is a module
        super().__init__(loss=loss, **kwargs)
        encoder_cont_size = len(self.hparams.x_reals)
        decoder_cont_size = len(
            self.hparams.time_varying_reals_decoder + self.hparams.static_reals
        )
        cat_size = sum([size[1] for size in self.hparams.embedding_sizes.values()])
        encoder_input_size = encoder_cont_size + cat_size
        decoder_input_size = decoder_cont_size + cat_size

        self.encoder_input_linear = nn.Linear(encoder_input_size, d_model)
        self.decoder_input_linear = nn.Linear(decoder_input_size, d_model)
        self.encoder_positional_encoding = PositionalEncoding(
            d_model, dropout, self.hparams.max_encoder_length
        )
        self.decoder_positional_encoding = PositionalEncoding(
            d_model, dropout, self.hparams.max_prediction_length
        )
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

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_head, dim_feedforward, dropout, activation
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, n_layers, decoder_norm
        )

        # (54,7)
        # self.dense_layer = nn.Linear(encoder_input_size, self.hparams.max_prediction_length*14)
        # # (54*14,1)
        self.out_linear = nn.Linear(
            d_model, self.hparams.output_size
        )  # (batch_size,seq_length,output_size)

        self.build_embeddings()

    def build_embeddings(self):
        self.embeddings = nn.ModuleDict()
        for name in self.hparams.embedding_sizes.keys():
            embedding_size = self.hparams.embedding_sizes[name][1]
            # convert to list to become mutable
            self.hparams.embedding_sizes[name] = list(
                self.hparams.embedding_sizes[name]
            )
            self.hparams.embedding_sizes[name][1] = embedding_size
            self.embeddings[name] = nn.Embedding(
                self.hparams.embedding_sizes[name][0],
                embedding_size,
            )

    @property
    def decoder_reals_positions(self) -> List[int]:
        return [
            self.hparams.x_reals.index(name)
            for name in self.reals
            if name in self.decoder_variables + self.static_variables
        ]

    def construct_input_vector(self, x_cat, x_cont):

        # create input vector
        if len(self.hparams.x_categoricals) > 0:
            input_vectors = {}
            for name, emb in self.embeddings.items():
                input_vectors[name] = emb(
                    x_cat[..., self.hparams.x_categoricals.index(name)]
                )

            flat_embeddings = torch.cat([emb for emb in input_vectors.values()], dim=-1)

            input_vector = flat_embeddings

        if len(self.hparams.x_reals) > 0:
            input_vector = x_cont

        if len(self.hparams.x_reals) > 0 and len(self.hparams.x_categoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)
        return input_vector

    def encode(self, x: Dict[str, torch.Tensor]):
        """
        Encode sequence into hidden state
        """
        # '_TimeSeriesDataset' stores time series in the
        # (batch_size, seq_length, input_size) format. PyTorch's nn.Transformer
        # module needs it the (seq_length, batch_size, input_size) format.
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])

        src = input_vector.permute(1, 0, 2)
        src = self.encoder_input_linear(src)
        src = self.encoder_positional_encoding(src)
        memory = self.transformer_encoder(src)
        return memory

    def decode(
        self,
        x: Dict[str, torch.Tensor],
        memory: torch.Tensor,
    ):
        network_decoder_input = x["decoder_cont"][..., self.decoder_reals_positions]
        decoder_input_vector = self.construct_input_vector(
            x["decoder_cat"], network_decoder_input
        )

        tgt = decoder_input_vector.permute(1, 0, 2)
        tgt = self.decoder_input_linear(tgt)
        tgt = self.decoder_positional_encoding(tgt)
        out = self.transformer_decoder(tgt, memory)
        return out

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        memory = self.encode(x)
        out = self.decode(x, memory)
        out = self.out_linear(out)
        prediction = out.permute(1, 0, 2)
        return prediction
