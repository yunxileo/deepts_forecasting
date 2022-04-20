import math
from abc import ABC
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepts_forecasting.models.base_model import BaseModel
from deepts_forecasting.utils.data import TimeSeriesDataSet


class ResidualBlock(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_fn,
        weight_norm: bool,
        nr_blocks_below: int,
        num_layers: int,
        input_size: int,
        target_size: int,
    ):
        """PyTorch module implementing a residual block module used in `_TCNModule`.

        Parameters
        ----------
        num_filters
            The number of filters in a convolutional layer of the TCN.
        kernel_size
            The size of every kernel in a convolutional layer.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        dropout_fn
            The dropout function to be applied to every convolutional layer.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        nr_blocks_below
            The number of residual blocks before the current one.
        num_layers
            The number of convolutional layers.
        input_size
            The dimensionality of the input time series of the whole network.
        target_size
            The dimensionality of the output time series of the whole network.

        Inputs
        ------
        x of shape `(batch_size, in_dimension, input_chunk_length)`
            Tensor containing the features of the input sequence.
            in_dimension is equal to `input_size` if this is the first residual block,
            in all other cases it is equal to `num_filters`.

        Outputs
        -------
        y of shape `(batch_size, out_dimension, input_chunk_length)`
            Tensor containing the output sequence of the residual block.
            out_dimension is equal to `output_size` if this is the last residual block,
            in all other cases it is equal to `num_filters`.
        """
        super().__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=output_dim,
            kernel_size=kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.weight_norm(
                self.conv1
            ), nn.utils.weight_norm(self.conv2)

        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        residual = x

        # first step
        left_padding = (self.dilation_base**self.nr_blocks_below) * (
            self.kernel_size - 1
        )
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(F.relu(self.conv1(x)))

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x)
        x = self.dropout_fn(x)

        # add residual
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x = x + residual

        return x


class TCNModel(BaseModel, ABC):
    def __init__(
        self,
        prediction_length: int,
        input_length: int,
        reals: List,
        kernel_size: int,
        num_filters: int,
        num_layers: Optional[int],
        dilation_base: int,
        weight_norm: bool,
        target_size: int,
        dropout: float,
        **kwargs
    ):

        """PyTorch module implementing a dilated TCN module used in `TCNModel`.


         Parameters
         ----------
         input_size
             The dimensionality of the input time series.
         target_size
             The dimensionality of the output time series.
        kernel_size
             The size of every kernel in a convolutional layer.
         num_filters
             The number of filters in a convolutional layer of the TCN.
         num_layers
             The number of convolutional layers.
         weight_norm
             Boolean value indicating whether to use weight normalization.
         dilation_base
             The base of the exponent that will determine the dilation on every level.
         dropout
             The dropout rate for every convolutional layer.
         **kwargs
             all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

         Inputs
         ------
         x of shape `(batch_size, input_length, input_size)`
             Tensor containing the features of the input sequence.

         Outputs
         -------
         y of shape `(batch_size, input_length, target_size, nr_params)`
             Tensor containing the predictions of the next 'output_chunk_length' points in the last
             'output_chunk_length' entries of the tensor. The entries before contain the data points
             leading up to the first prediction, all in chronological order.
        """

        super().__init__(**kwargs)

        # required for all modules -> saves hparams for checkpoints
        self.save_hyperparameters()
        # Defining parameters

        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.target_size = target_size
        self.dilation_base = dilation_base
        self.dropout = nn.Dropout(p=dropout)
        cont_size = len(self.hparams.reals)
        self.input_size = cont_size
        self.out_linear = nn.Linear(
            self.hparams.input_length * self.target_size, self.hparams.prediction_length
        )  # (batch_size,seq_length,output_size)

        # If num_layers is not passed, compute number of layers needed for full history coverage
        if num_layers is None and dilation_base > 1:
            num_layers = math.ceil(
                math.log(
                    (self.input_chunk_length - 1)
                    * (dilation_base - 1)
                    / (kernel_size - 1)
                    / 2
                    + 1,
                    dilation_base,
                )
            )
            # logger.info("Number of layers chosen: " + str(num_layers))
        elif num_layers is None:
            num_layers = math.ceil(
                (self.input_chunk_length - 1) / (kernel_size - 1) / 2
            )
            # logger.info("Number of layers chosen: " + str(num_layers))
        self.num_layers = num_layers

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = ResidualBlock(
                num_filters,
                kernel_size,
                dilation_base,
                self.dropout,
                weight_norm,
                i,
                num_layers,
                self.input_size,
                target_size,
            )
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)

    def forward(self, x: Dict[str, torch.Tensor]):
        # data is of size (batch_size, input_length, input_size)
        x = x["encoder_cont"]
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        for res_block in self.res_blocks_list:
            x = res_block(x)

        x = x.transpose(1, 2)
        x = x.view(batch_size, self.hparams.input_length * self.target_size)

        y = self.out_linear(x)
        y = y.reshape(batch_size, self.hparams.prediction_length, 1)

        return y

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        """
        Convenience function to create network from :py:class`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Args:
            dataset (TimeSeriesDataSet): dataset where sole predictor is the target.
            **kwargs: additional arguments to be passed to ``__init__`` method.

        Returns:
            NBeats
        """
        new_kwargs = {
            "prediction_length": dataset.max_prediction_length,
            "input_length": dataset.max_encoder_length,
            "reals": dataset.reals,
        }
        new_kwargs.update(kwargs)

        # validate arguments
        assert (
            dataset.min_encoder_length == dataset.max_encoder_length
        ), "only fixed encoder length is allowed, but min_encoder_length != max_encoder_length"

        assert (
            dataset.max_prediction_length == dataset.min_prediction_length
        ), "only fixed prediction length is allowed, but max_prediction_length != min_prediction_length"

        # initialize class
        return super().from_dataset(dataset, **new_kwargs)
