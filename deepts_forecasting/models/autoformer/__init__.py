import math
from abc import ABC
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from deepts_forecasting.models.autoformer.sub_module import (
    AutoCorrelation,
    AutoCorrelationLayer,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    MyLayerNorm,
    SeriesDecompose,
)
from deepts_forecasting.models.base_model import BaseModelWithCovariates


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class Autoformer(BaseModelWithCovariates, ABC):
    def __init__(
        self,
        d_model=64,
        d_ff=256,
        n_heads=4,
        num_layers=2,
        factor=5,
        moving_avg=21,
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
        super(Autoformer, self).__init__(loss=loss, **kwargs)
        # Decomp
        kernel_size = moving_avg
        self.decomp = SeriesDecompose(kernel_size)
        encoder_cont_size = len(self.hparams.x_reals)
        decoder_cont_size = len(
            self.hparams.time_varying_reals_decoder + self.hparams.static_reals
        )
        cat_size = sum([size[1] for size in self.hparams.embedding_sizes.values()])
        encoder_input_size = encoder_cont_size + cat_size
        decoder_input_size = decoder_cont_size + cat_size
        self.build_embeddings()
        self.encoder_input_linear = TokenEmbedding(encoder_input_size, d_model)
        self.decoder_input_linear = TokenEmbedding(decoder_input_size, d_model)
        self.out_linear = nn.Linear(
            d_model, self.hparams.output_size
        )  # (batch_size,seq_length,output_size)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for num in range(num_layers)
            ],
            norm_layer=MyLayerNorm(d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model=d_model,
                    c_out=d_model,
                    d_ff=d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for num in range(num_layers)
            ],
            norm_layer=MyLayerNorm(d_model),
            projection=None,
        )

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

        seasonal_init, trend_init = self.decomp(x["encoder_cont"])
        src = self.encoder_input_linear(input_vector)
        memory = self.encoder(src)
        return memory, seasonal_init, trend_init

    def decode(
        self,
        x: Dict[str, torch.Tensor],
        memory: torch.Tensor,
        seasonal_init: torch.Tensor,
        trend_init: torch.Tensor,
    ):

        # decomp init
        mean = (
            torch.mean(x["encoder_cont"], dim=1)
            .unsqueeze(1)
            .repeat(1, self.hparams.max_prediction_length, 1)
        )
        zeros = torch.zeros(
            [
                seasonal_init.shape[0],
                self.hparams.max_prediction_length,
                seasonal_init.shape[2],
            ]
        )

        # make decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.hparams.max_prediction_length :, :], mean], dim=1
        )
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.hparams.max_prediction_length :, :], zeros], dim=1
        )
        trend_init = TokenEmbedding(trend_init.shape[2], self.hparams.d_model)(
            trend_init
        )
        # # decoder input_1
        # cont_decoder_input = x["decoder_cont"][..., self.decoder_reals_positions]
        # decoder_input_vector = self.construct_input_vector(x["decoder_cat"],
        #                                                    cont_decoder_input)
        # dec_input_1 = self.decoder_input_linear(decoder_input_vector)

        # decoder input_2
        dec_input_2 = TokenEmbedding(seasonal_init.shape[2], self.hparams.d_model)(
            seasonal_init
        )

        # # concat input_1 and input_2
        # decoder_input = torch.cat([dec_input_1, dec_input_2], dim=2)
        decoder_input = dec_input_2
        projection = nn.Linear(decoder_input.shape[2], self.hparams.d_model, bias=True)
        tgt = projection(decoder_input)
        seasonal_part, trend_part = self.decoder(
            tgt, memory, x_mask=None, cross_mask=None, trend=trend_init
        )
        dec_out = trend_part + seasonal_part
        return dec_out

    def forward(self, x, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        memory, seasonal_init, trend_init = self.encode(x)
        out = self.decode(x, memory, seasonal_init, trend_init)
        out = self.out_linear(out)
        return out[:, -self.hparams.max_prediction_length :, :]
