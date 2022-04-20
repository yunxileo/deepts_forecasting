from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import torch
from deepts_forecasting.metrics import MAE
from deepts_forecasting.models.base_model import BaseModelWithCovariates
from torch import nn
from torch.nn import GRU, LSTM, RNN

HiddenState = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


def get_rnn(cell_type: Union[Type[RNN], str]) -> Type[RNN]:
    """
    Get LSTM or GRU.

    Args:
        cell_type (Union[RNN, str]): "LSTM" or "GRU"

    Returns:
        Type[RNN]: returns GRU or LSTM RNN module
    """
    if isinstance(cell_type, RNN):
        rnn = cell_type
    elif cell_type == "LSTM":
        rnn = LSTM
    elif cell_type == "GRU":
        rnn = GRU
    else:
        raise ValueError(
            f"RNN type {cell_type} is not supported. supported: [LSTM, GRU]"
        )
    return rnn


class Seq2SeqNetwork(BaseModelWithCovariates, ABC):
    def __init__(
        self,
        hidden_size: int = 10,
        rnn_layers: int = 2,
        max_prediction_length: int = 7,
        max_encoder_length: int = 14,
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
        # loss: MAE = None,
        loss=nn.L1Loss(),
        cell_type: str = "LSTM",
        **kwargs,
    ):
        if loss is None:
            loss = MAE()
        self.save_hyperparameters()
        # store loss function separately as it is a module
        super().__init__(loss=loss, **kwargs)
        rnn_class = get_rnn(cell_type)
        self.dense_layer = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)
        encoder_cont_size = len(self.hparams.x_reals)
        decoder_cont_size = len(
            self.hparams.time_varying_reals_decoder + self.hparams.static_reals
        )
        cat_size = sum([size[1] for size in self.hparams.embedding_sizes.values()])
        encoder_input_size = encoder_cont_size + cat_size
        decoder_input_size = decoder_cont_size + cat_size

        self.encode_rnn = rnn_class(
            input_size=encoder_input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            batch_first=True,
        )

        self.decode_rnn = rnn_class(
            input_size=decoder_input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            batch_first=True,
        )

        self.build_embeddings()

    @property
    def decoder_reals_positions(self) -> List[int]:
        return [
            self.hparams.x_reals.index(name)
            for name in self.reals
            if name in self.decoder_variables + self.static_variables
        ]

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
        # encode using rnn
        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
        encoder_output, hidden_state = self.encode_rnn(input_vector)
        return encoder_output, hidden_state

    def decode(
        self,
        x: Dict[str, torch.Tensor],
        hidden_state: HiddenState,
        # target_scale: torch.Tensor
    ):
        # decoder_lengths = x["decoder_lengths"]
        network_decoder_input = x["decoder_cont"][..., self.decoder_reals_positions]
        decoder_input_vector = self.construct_input_vector(
            x["decoder_cat"], network_decoder_input
        )
        decoder_output, hidden_state = self.decode_rnn(
            decoder_input_vector, hidden_state
        )
        # output = self.transform_output(decoder_output,
        #                                target_scale=target_scale)
        return decoder_output

    def forward(
        self, x: Dict[str, torch.Tensor], n_samples: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward network
        """
        _, hidden_state = self.encode(x)
        output = self.decode(
            x,
            # target_scale=x["target_scale"],
            hidden_state=hidden_state,
        )

        prediction = self.dense_layer(output)
        return prediction
