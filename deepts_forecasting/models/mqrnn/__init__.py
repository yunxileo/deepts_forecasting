from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from deepts_forecasting.models.base_model import BaseModelWithCovariates
from torch import nn


class QuantileLoss(nn.Module):
    """source: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629"""

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """preds: tensor of shape (batch, prediction_length, num_quantiles)
        target: tensor of shape (batch, prediction_length,1)
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):

            errors = target.squeeze(-1) - preds[:, :, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class MQRNN(BaseModelWithCovariates):
    """
    Global decoder for output of encoder LSTM
    """

    def __init__(
        self,
        hidden_size: int,
        rnn_layers: int,
        max_prediction_length: int,
        max_encoder_length: int,
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
        quantiles=[0.25, 0.5, 0.75],
        loss=QuantileLoss(quantiles=[0.25, 0.5, 0.75]),
        **kwargs,
    ):

        self.embeddings = None
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]
        if loss is None:
            loss = QuantileLoss(quantiles=[0.25, 0.5, 0.75])
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(loss=loss, **kwargs)

        encoder_cont_size = len(
            self.hparams.time_varying_reals_encoder + self.hparams.static_reals
        )
        decoder_cont_size = len(
            self.hparams.time_varying_reals_decoder + self.hparams.static_reals
        )
        cat_size = sum([size[1] for size in self.hparams.embedding_sizes.values()])
        encoder_input_size = encoder_cont_size + cat_size
        decoder_input_size = decoder_cont_size + cat_size
        encoder_hidden_size = self.hparams.hidden_size * (
            self.hparams.max_prediction_length + 1
        )

        self.global_mlp = nn.Linear(
            decoder_input_size + self.hparams.hidden_size, encoder_hidden_size
        )
        # print(self.global_mlp) # Linear(in_features=42, out_features=35, bias=True)
        self.local_mlp = nn.Linear(
            decoder_input_size
            + 2 * self.hparams.hidden_size * self.hparams.max_prediction_length,
            len(quantiles),
        )
        self.encode_rnn = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            batch_first=True,
        )
        self.build_embeddings()

    @staticmethod
    def get_embedding_size(n: int, max_size: int = 100) -> int:
        """
        Determine empirically good embedding sizes (formula taken from fastai).

        Args:
            n (int): number of classes
            max_size (int, optional): maximum embedding size. Defaults to 100.

        Returns:
            int: embedding size
        """
        if n > 2:
            return min(round(1.6 * n**0.56), max_size)
        else:
            return 1

    @property
    def decoder_reals_positions(self):
        return [
            self.hparams.x_reals.index(name)
            for name in self.hparams.x_reals
            if name
            in self.hparams.time_varying_reals_decoder + self.hparams.static_reals
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

    def encode(self, x):
        """
        Encode sequence into hidden state
        """
        # encode using rnn

        input_vector = self.construct_input_vector(x["encoder_cat"], x["encoder_cont"])
        # print("input_vector shape is:", input_vector.shape)
        encoder_output, (hidden_state, _) = self.encode_rnn(
            input_vector,
        )  # second ouput is not needed (hidden state)
        return encoder_output, hidden_state

    def global_decode(
        self,
        x,
        hidden_state,
    ):

        network_decoder_input = x["decoder_cont"][..., self.decoder_reals_positions]

        # print("network_decoder_input shape is:", network_decoder_input.shape)
        decoder_input_vector = self.construct_input_vector(
            x["decoder_cat"], network_decoder_input
        )

        # print("decoder_input_vector shape is:", decoder_input_vector.shape) # torch.Size([2, 6, 37])
        global_input = torch.cat([decoder_input_vector, hidden_state], dim=2)
        # print("global_input_vector shape is:", global_input.size()) # torch.Size([2, 6, 72])
        global_output = self.global_mlp(global_input)
        # print("global_output shape is:", global_output.size())
        return global_output, decoder_input_vector

    def forward(self, x):

        _, hidden_state = self.encode(x)
        hidden_state = hidden_state[-1, :, :]
        hidden_state = hidden_state.unsqueeze(1)
        # print("hidden_state shape is ", hidden_state.size()) # torch.Size([2, 1, 5])
        h_t = hidden_state.expand(-1, self.hparams.max_prediction_length, -1)
        context, decoder_input = self.global_decode(x, h_t)
        # print('context shape is:', context.size())
        context = context.view(
            -1,
            self.hparams.max_prediction_length + 1,
            self.hparams.hidden_size * self.hparams.max_prediction_length,
        )

        c_alpha = context[:, -1, :]  # batch_size*hidden_size
        # print('c_alpha shape is:', c_alpha.size())
        c_t = context[:, :-1, :]
        # print('c_t shape is:', c_t.size())
        y = []
        for i in range(self.hparams.max_prediction_length):
            c_i = c_t[:, i, :]
            # print("c_i shape is:", c_i.size())
            x_i = decoder_input[:, i, :]
            local_input = torch.cat([c_i, c_alpha, x_i], dim=1)  # batch_size*
            out = self.local_mlp(local_input)  # batch_size*num_quantiles
            y.append(out.unsqueeze(1))
        output_tensor = torch.cat(
            y, dim=1
        )  # batch_size*prediction_length*num_quantiles
        return output_tensor
