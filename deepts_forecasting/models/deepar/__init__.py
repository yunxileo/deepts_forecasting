from abc import ABC
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from deepts_forecasting.models.base_model import BaseModelWithCovariates
from torch import distributions, nn


def gaussian_sample(mu, sigma):
    distribution = torch.distributions.normal.Normal(mu, sigma)
    samples = distribution.sample((100,))  # n_sample*batch_size*sequence_length
    prediction = samples.mean(dim=0)
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(-1)
    return prediction  # batch_size*sequence_length


class NormalDistributionLoss(nn.Module):
    """
    DistributionLoss base class.

    Class should be inherited for all distribution losses, i.e. if a network predicts
    the parameters of a probability distribution, DistributionLoss can be used to
    score those parameters and calculate loss for given true values.

    Define two class attributes in a child class:

    Attributes:
        distribution_class (distributions.Distribution): torch probability distribution
        distribution_arguments (List[str]): list of parameter names for the distribution

    Further, implement the methods :py:meth:`~map_x_to_distribution` and :py:meth:`~rescale_parameters`.
    """

    def __init__(self):
        """
        Initialize metric

        Args:
            name (str): metric name. Defaults to class name.
            quantiles (List[float], optional): quantiles for probability range.
                Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
            reduction (str, optional): Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
        """
        super().__init__()
        self.distribution = distributions.Normal

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        """
        Map the a tensor of parameters to a probability distribution.

        Args:
            x (torch.Tensor): parameters for probability distribution. Last dimension will index the parameters

        Returns:
            distributions.Distribution: torch probability distribution as defined in the
                class attribute ``distribution_class``
        """
        return self.distribution(loc=x[..., 0], scale=x[..., 1])

    def forward(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        if len(y_actual.shape) == 3:
            y_actual = y_actual.squeeze(dim=-1)
        distribution = self.map_x_to_distribution(y_pred)
        loss = -distribution.log_prob(y_actual)
        return torch.mean(loss)

    def predict(self, y_pred, n_samples: int) -> torch.Tensor:
        """
        Sample from distribution.

        Args:
            y_pred: prediction output of network (shape batch_size x sequence_length x n_paramters)
            n_samples (int): number of samples to draw

        Returns:
            torch.Tensor: tensor with samples  (shape batch_size x sequence_length)
        """
        dist = self.map_x_to_distribution(y_pred)
        samples = dist.sample((n_samples,))
        if samples.ndim == 3:
            samples = samples.permute(1, 2, 0)
        elif samples.ndim == 2:
            samples = samples.transpose(0, 1)
        return samples.mean(dim=0)  # batch_size x sequence_length


class DeepAR(BaseModelWithCovariates, ABC):
    """ """

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
        loss=NormalDistributionLoss(),
        **kwargs,
    ):
        self.embeddings = None
        if loss is None:
            loss = NormalDistributionLoss()
        # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(loss=loss, **kwargs)

        self.mu_layer = nn.Linear(self.hparams.hidden_size, 1)
        self.sigma_layer = nn.Linear(self.hparams.hidden_size, 1)
        self.activation = nn.Softplus()

        encoder_cont_size = len(
            self.hparams.time_varying_reals_encoder + self.hparams.static_reals
        )
        decoder_cont_size = len(
            self.hparams.time_varying_reals_decoder + self.hparams.static_reals
        )
        cat_size = sum([size[1] for size in self.hparams.embedding_sizes.values()])
        encoder_input_size = encoder_cont_size + cat_size
        decoder_input_size = decoder_cont_size + cat_size

        self.encode_rnn = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            batch_first=True,
        )

        self.decode_rnn = nn.LSTM(
            input_size=decoder_input_size,
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
        encoder_output, hidden_state = self.encode_rnn(
            input_vector,
        )  # second output is not needed (hidden state)
        return encoder_output, hidden_state

    def decode(
        self,
        x,
        hidden_state,
    ):

        network_decoder_input = x["decoder_cont"][..., self.decoder_reals_positions]
        # print("network_decoder_input shape is:", network_decoder_input.shape)
        decoder_input_vector = self.construct_input_vector(
            x["decoder_cat"], network_decoder_input
        )
        # print("decoder_input_vector shape is:", decoder_input_vector.shape)
        decoder_output, hidden_state = self.decode_rnn(
            decoder_input_vector, hidden_state
        )

        return decoder_output

    def forward(self, x, n_samples: int = None):
        """
        Forward network
        """
        _, hidden_state = self.encode(x)
        output = self.decode(
            x,
            hidden_state=hidden_state,
        )
        mu = self.mu_layer(output).squeeze()
        sigma = self.sigma_layer(output)
        sigma = self.activation(sigma).squeeze()
        prediction = torch.stack([mu, sigma], dim=-1)
        # return relevant part
        return prediction

    def predict(self, data, **kwargs):
        dataloader = data
        # prepare model
        self.eval()  # no dropout, etc. no gradients
        outputs = []
        with torch.no_grad():
            for x, _ in dataloader:
                # make prediction
                out = self(x, **kwargs)  # raw output is dictionary
                output = gaussian_sample(out[..., 0], out[..., 1])
                output = self.transform_output(
                    prediction=output, target_scale=x["target_scale"]
                )
                outputs.append(output)
        return torch.cat(outputs)
