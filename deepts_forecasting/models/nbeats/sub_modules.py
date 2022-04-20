"""
Implementations of Stacks, Blocks, Sublayers, etc. for N-BEATS network.
"""

from enum import Enum
from typing import List, NewType, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GType(Enum):
    GENERIC = 1
    TREND = 2
    SEASONALITY = 3
    EXOGENOUS = 4


GTypes = NewType("GTypes", _GType)


class TrendBasis(nn.Module):
    """
    Examples
    >>> trend = TrendBasis(3,4)
    >>> x = torch.rand(2,4,3)
    >>> out = trend(x)
    """

    def __init__(self, expansion_coefficient_dim, seq_length):
        super().__init__()

        # basis is of size (expansion_coefficient_dim, target_length)
        basis = torch.stack(
            [
                (torch.arange(seq_length) / seq_length) ** i
                for i in range(expansion_coefficient_dim)
            ],
            dim=1,
        ).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class SeasonalityBasis(nn.Module):
    def __init__(self, seq_length):
        super().__init__()
        half_minus_one = int(seq_length / 2 - 1)
        cos_vectors = [
            torch.cos(torch.arange(seq_length) / seq_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]

        sin_vectors = [
            torch.sin(torch.arange(seq_length) / seq_length * 2 * np.pi * i)
            for i in range(1, half_minus_one + 1)
        ]

        # basis is of size (2 * int(seq_length / 2 - 1) + 1, seq_length)
        basis = torch.stack(
            [torch.ones(seq_length)] + cos_vectors + sin_vectors, dim=1
        ).T

        self.basis = nn.Parameter(basis, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x, self.basis)


class _Block(nn.Module):
    """
    Examples
    >>> block = _Block(3, 6, 4, 1, 3, 10, 5, _GType.EXOGENOUS)
    >>> x = torch.rand(2,10,1)
    >>> e_x = torch.rand(2,15,6)
    >>> out = block(x, e_x)

        Outputs
        -------
        backcast of shape `(batch_size, input_chunk_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        forecast of shape `(batch_size, output_chunk_length, nr_params)`
            Tensor containing the forward forecast by `g`

    """

    def __init__(
        self,
        num_layers: int,
        exogenous_dim: int,
        layer_width: int,
        nr_params: int,
        expansion_coefficient_dim: int,
        input_chunk_length: int,
        target_length: int,
        g_type: GTypes,
    ):
        """PyTorch module implementing the basic building block of the N-BEATS architecture.

        The blocks produce outputs of size (target_length, nr_params); i.e.
        "one vector per parameter". The parameters are predicted only for forecast outputs.
        Backcast outputs are in the original "domain".

        Parameters
        ----------
        num_layers
            The number of fully connected layers preceding the final forking layers.
        layer_width
            The number of neurons that make up each fully connected layer.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used)
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Used in the generic architecture and the trend module of the interpretable architecture, where it determines
            the degree of the polynomial basis.
        input_chunk_length
            The length of the input sequence fed to the model.
        target_length
            The length of the forecast of the model.
        g_type
            The type of function that is implemented by the waveform generator.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        x_hat of shape `(batch_size, input_chunk_length)`
            Tensor containing the 'backcast' of the block, which represents an approximation of `x`
            given the constraints of the functional space determined by `g`.
        y_hat of shape `(batch_size, output_chunk_length)`
            Tensor containing the forward forecast of the block.

        """
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.target_length = target_length
        self.nr_params = nr_params
        self.g_type = g_type
        self.relu = nn.ReLU()
        self.exogenous_dim = exogenous_dim

        # fully connected stack before fork
        self.linear_layer_stack_list = [nn.Linear(self.input_chunk_length, layer_width)]
        if self.exogenous_dim > 0:
            units = (
                input_chunk_length + target_length
            ) * exogenous_dim + input_chunk_length * 1
            self.linear_layer_stack_list = [nn.Linear(units, layer_width)]
        self.linear_layer_stack_list += [
            nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)
        ]
        self.fc_stack = nn.ModuleList(self.linear_layer_stack_list)

        # Fully connected layer producing forecast/backcast expansion coeffcients (waveform generator parameters).
        # The coefficients are emitted for each parameter of the likelihood.
        if g_type == _GType.SEASONALITY:
            self.backcast_linear_layer = nn.Linear(
                layer_width, 2 * int(input_chunk_length / 2 - 1) + 1
            )
            self.forecast_linear_layer = nn.Linear(
                layer_width, nr_params * (2 * int(target_length / 2 - 1) + 1)
            )
        elif g_type == _GType.EXOGENOUS:
            self.backcast_linear_layer = nn.Linear(layer_width, exogenous_dim)
            self.forecast_linear_layer = nn.Linear(layer_width, exogenous_dim)
        else:
            self.backcast_linear_layer = nn.Linear(
                layer_width, expansion_coefficient_dim
            )
            self.forecast_linear_layer = nn.Linear(
                layer_width, nr_params * expansion_coefficient_dim
            )

        # waveform generator functions
        if g_type == _GType.GENERIC:
            self.backcast_g = nn.Linear(expansion_coefficient_dim, input_chunk_length)
            self.forecast_g = nn.Linear(expansion_coefficient_dim, target_length)
        elif g_type == _GType.TREND:
            self.backcast_g = TrendBasis(expansion_coefficient_dim, input_chunk_length)
            self.forecast_g = TrendBasis(expansion_coefficient_dim, target_length)
        elif g_type == _GType.SEASONALITY:
            self.backcast_g = SeasonalityBasis(input_chunk_length)
            self.forecast_g = SeasonalityBasis(target_length)
        elif g_type == _GType.EXOGENOUS:
            self.backcast_g = torch.einsum
            self.forecast_g = torch.einsum
        else:
            raise (ValueError("g_type not supported"))

    def forward(self, x, exogenous_x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        if exogenous_x is not None and self.exogenous_dim > 0:
            x = torch.cat([x, exogenous_x.reshape(batch_size, -1)], dim=1)
        # assert (x.shape[-1] == 1), "the last dim must be 1"
        # if x.shape[-1] == 1:
        #     x = torch.squeeze(x)
        # fully connected layer stack
        for layer in self.linear_layer_stack_list:
            x = self.relu(layer(x))
        # forked linear layers producing waveform generator parameters
        theta_backcast = self.backcast_linear_layer(x)
        theta_forecast = self.forecast_linear_layer(x)

        # set the expansion coefs in last dimension for the forecasts
        # theta_forecast = theta_forecast.view(batch_size, self.nr_params, -1)
        if exogenous_x is not None and self.exogenous_dim > 0:
            exogenous_x = exogenous_x.permute(0, 2, 1)
            backcast_basis = exogenous_x[:, :, : self.input_chunk_length]
            forecast_basis = exogenous_x[:, :, self.input_chunk_length :]
            x_hat = self.backcast_g("bz,bzs->bs", theta_backcast, backcast_basis)
            y_hat = self.forecast_g("bz,bzs->bs", theta_forecast, forecast_basis)
        # waveform generator applications (project the expansion coefs onto basis vectors)
        else:
            x_hat = self.backcast_g(theta_backcast)
            y_hat = self.forecast_g(theta_forecast)

        # Set the distribution parameters as the last dimension
        # x_hat = x_hat.reshape(batch_size, self.input_chunk_length, -1)
        y_hat = y_hat.reshape(batch_size, self.target_length, self.nr_params)
        return x_hat, y_hat


class _Stack(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        exogenous_dim: int,
        layer_width: int,
        nr_params: int,
        expansion_coefficient_dim: int,
        input_chunk_length: int,
        target_length: int,
        g_type: GTypes,
    ):
        """PyTorch module implementing one stack of the N-BEATS architecture that comprises multiple basic blocks.


               Parameters
               ----------
               num_blocks
                   The number of blocks making up this stack.
               num_layers
                   The number of fully connected layers preceding the final forking layers in each block.
               layer_width
                   The number of neurons that make up each fully connected layer in each block.
               nr_params
                   The number of parameters of the likelihood (or 1 if no likelihood is used)
               expansion_coefficient_dim
                   The dimensionality of the waveform generator parameters, also known as expansion coefficients.
               input_chunk_length
                   The length of the input sequence fed to the model.
               target_length
                   The length of the forecast of the model.
               g_type
                   The function that is implemented by the waveform generators in each block.

               Inputs
               ------
               stack_input of shape `(batch_size, input_chunk_length)`
                   Tensor containing the input sequence.

               Outputs
               -------
               stack_residual of shape `(batch_size, input_chunk_length)`
                   Tensor containing the 'backcast' of the block, which represents an approximation of `x`
                   given the constraints of the functional space determined by `g`.
               stack_forecast of shape `(batch_size, output_chunk_length)`
                   Tensor containing the forward forecast of the stack.
        =
               Examples
               >>> stack = _Stack(3,3,0,4,1,3,10,5,_GType.TREND)
               >>> x = torch.rand(2,10,1)
               >>> e_x = torch.rand(2,15,6)
               >>> out = stack(x, exogenous_x=None)

        """
        super().__init__()

        self.input_chunk_length = input_chunk_length
        self.target_length = target_length
        self.nr_params = nr_params

        block = _Block(
            num_layers,
            exogenous_dim,
            layer_width,
            nr_params,
            expansion_coefficient_dim,
            input_chunk_length,
            target_length,
            g_type,
        )

        self.blocks = nn.ModuleList([block] * num_blocks)

    def forward(self, x, exogenous_x):
        # One forecast vector per parameter in the distribution

        stack_forecast = torch.zeros(
            x.shape[0],
            self.target_length,
            self.nr_params,
            device=x.device,
            dtype=x.dtype,
        )
        for block in self.blocks:
            # pass input through block
            x_hat, y_hat = block(x, exogenous_x)

            # add block forecast to stack forecast
            stack_forecast = stack_forecast + y_hat

            # subtract backcast from input to produce residual
            x = x - x_hat

        stack_residual = x

        return stack_residual, stack_forecast


# class _NBEATSModule(nn.Module):
#     def __init__(
#         self,
#         input_chunk_length: int,
#         target_length: int,
#         nr_params: int,
#         generic_architecture: bool,
#         num_stacks: int,
#         num_blocks: int,
#         num_layers: int,
#         layer_widths: List[int],
#         expansion_coefficient_dim: int,
#         trend_polynomial_degree: int,
#         **kwargs
#     ):
#         """PyTorch module implementing the N-BEATS architecture.
#
#         Parameters
#         ----------
#             Number of output components in the target
#         nr_params
#             The number of parameters of the likelihood (or 1 if no likelihood is used).
#         generic_architecture
#             Boolean value indicating whether the generic architecture of N-BEATS is used.
#             If not, the interpretable architecture outlined in the paper (consisting of one trend
#             and one seasonality stack with appropriate waveform generator functions).
#         num_stacks
#             The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
#         num_blocks
#             The number of blocks making up every stack.
#         num_layers
#             The number of fully connected layers preceding the final forking layers in each block of every stack.
#             Only used if `generic_architecture` is set to `True`.
#         layer_widths
#             Determines the number of neurons that make up each fully connected layer in each block of every stack.
#             If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds
#             to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks
#             with FC layers of the same width.
#         expansion_coefficient_dim
#             The dimensionality of the waveform generator parameters, also known as expansion coefficients.
#             Only used if `generic_architecture` is set to `True`.
#         trend_polynomial_degree
#             The degree of the polynomial used as waveform generator in trend stacks. Only used if
#             `generic_architecture` is set to `False`.
#         **kwargs
#             all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.
#
#         Inputs
#         ------
#         x of shape `(batch_size, input_chunk_length)`
#             Tensor containing the input sequence.
#
#         Outputs
#         -------
#         y of shape `(batch_size, output_chunk_length, target_size/output_dim, nr_params)`
#             Tensor containing the output of the NBEATS module.
#
#         Examples::
#
#         >>> block = _NBEATSModule(10, 5, 1,False ,3,2,3,[16,16], 3, 3)
#         >>> x = torch.rand(2,10,1)
#         >>> out = block(x)
#
#         """
#         super().__init__(**kwargs)
#
#         # required for all modules -> saves hparams for checkpoints
#         #self.save_hyperparameters()
#
#         self.input_chunk_length = input_chunk_length
#         self.target_length = target_length
#         self.nr_params = nr_params
#
#         if generic_architecture:
#             self.stacks_list = [
#                 _Stack(num_blocks, num_layers,, layer_widths[
#                     i], nr_params, expansion_coefficient_dim, self.input_chunk_length, self.target_length, _GType.GENERIC
#                 for i in range(num_stacks)
#             ]
#         else:
#             num_stacks = 2
#             trend_stack = _Stack(num_blocks, num_layers,, layer_widths[
#                 0], nr_params, trend_polynomial_degree + 1, self.input_chunk_length, self.target_length, _GType.TREND
#             seasonality_stack = _Stack(num_blocks, num_layers,, layer_widths[
#                 1], nr_params, -1, self.input_chunk_length, self.target_length, _GType.SEASONALITY
#             self.stacks_list = [trend_stack, seasonality_stack]
#
#         self.stacks = nn.ModuleList(self.stacks_list)
#
#         # setting the last backcast "branch" to be not trainable (without next block/stack, it doesn't need to be
#         # backpropagated). Removing this lines would cause logtensorboard to crash, since no gradient is stored
#         # on this params (the last block backcast is not part of the final output of the net).
#         self.stacks_list[-1].blocks[-1].backcast_linear_layer.requires_grad_(False)
#         self.stacks_list[-1].blocks[-1].backcast_g.requires_grad_(False)
#
#     def forward(self, x):
#
#         # if x1, x2,... y1, y2... is one multivariate ts containing x and y, and a1, a2... one covariate ts
#         # we reshape into x1, y1, a1, x2, y2, a2... etc
#         x = torch.reshape(x, (x.shape[0], self.input_chunk_length, 1))
#         # squeeze last dimension (because model is univariate)
#         x = x.squeeze(dim=2)
#
#         # One vector of length target_length per parameter in the distribution
#         y = torch.zeros(
#             x.shape[0],
#             self.target_length,
#             self.nr_params,
#             device=x.device,
#             dtype=x.dtype,
#         )
#
#         for stack in self.stacks_list:
#             # compute stack output
#             stack_residual, stack_forecast = stack(x)
#
#             # add stack forecast to final output
#             y = y + stack_forecast
#
#             # set current stack residual as input for next stack
#             x = stack_residual
#
#         # In multivariate case, we get a result [x1_param1, x1_param2], [y1_param1, y1_param2], [x2..], [y2..], ...
#         # We want to reshape to original format. We also get rid of the covariates and keep only the target dimensions.
#         # The covariates are by construction added as extra time series on the right side. So we need to get rid of this
#         # right output (keeping only :self.output_dim).
#         y = y.view(
#             y.shape[0], self.target_length, self.nr_params
#         )
#
#         return y
