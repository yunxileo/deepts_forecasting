"""
N-BEATS
-------
"""
from typing import Dict, List, NewType, Tuple, Union

import torch
from deepts_forecasting.models.base_model import BaseModel
from deepts_forecasting.models.nbeats.sub_modules import _GType, _Stack
from deepts_forecasting.utils.data import TimeSeriesDataSet
from torch import nn


class NBEATSModel(BaseModel):
    def __init__(
        self,
        prediction_length: int,
        input_length: int,
        exogenous_dim: int,
        time_varying_known_reals: List,
        reals: List,
        nr_params: int,
        stack_types: List[str],
        num_blocks: int,
        num_layers: int,
        layer_widths: List[int],
        expansion_coefficient_dim: int,
        **kwargs,
    ):
        """
        Initialize NBeats Model - use its :py:meth:`~from_dataset` method if possible.

        Based on the article
        `N-BEATS: Neural basis expansion analysis for interpretable time series
        forecasting <http://arxiv.org/abs/1905.10437>`_. The network has (if used as ensemble) outperformed all
        other methods
        including ensembles of traditional statical methods in the M4 competition. The M4 competition is arguably
        the most important benchmark for univariate time series forecasting.
        Args:
            prediction_length: Length of the prediction. Also known as 'horizon'.
            nr_params
                The number of parameters of the likelihood (or 1 if no likelihood is used).
            num_layers
                The number of fully connected layers preceding the final forking layers in each block of every stack.
                Only used if `generic_architecture` is set to `True`.
            layer_widths
                Determines the number of neurons that make up each fully connected layer in each block of every stack.
                If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds
                to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks
                with FC layers of the same width.
            expansion_coefficient_dim
                If the `generic_architecture` is set to `True`, then the length of the expansion coefficient.
                If type is “T” (trend), then it corresponds to the degree of the polynomial.
                If the type is “S”(seasonal) then this is the minimum period allowed, e.g. 2 for changes every timestep.
                A list of ints of length 1 or ‘num_stacks’. Default value for generic mode: [32] Recommended value for
                interpretable mode: [3]
            stack_types: One of the following values: “generic”, “seasonality" or “trend". A list of strings
                of length 1 or ‘num_stacks’. Default and recommended value
                for generic mode: [“generic”] Recommended value for interpretable mode: [“trend”,”seasonality”]
            **kwargs
                all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, target_size/output_dim, nr_params)`
            Tensor containing the output of the NBEATS module.

        """
        super().__init__(**kwargs)

        # required for all modules -> saves hparams for checkpoints
        self.save_hyperparameters()
        input_chunk_length = self.hparams.input_length
        target_length = self.hparams.prediction_length
        self.nr_params = nr_params
        self.stacks = nn.ModuleList()
        for stack_id, stack_type in enumerate(stack_types):
            if stack_type == "generic":
                stack = _Stack(
                    num_blocks=num_blocks,
                    num_layers=num_layers,
                    exogenous_dim=0,
                    layer_width=layer_widths[stack_id],
                    nr_params=nr_params,
                    expansion_coefficient_dim=expansion_coefficient_dim,
                    input_chunk_length=input_chunk_length,
                    target_length=target_length,
                    g_type=_GType.GENERIC,
                )
            elif stack_type == "trend":
                stack = _Stack(
                    num_blocks=num_blocks,
                    num_layers=num_layers,
                    exogenous_dim=0,
                    layer_width=layer_widths[stack_id],
                    nr_params=nr_params,
                    expansion_coefficient_dim=expansion_coefficient_dim,
                    input_chunk_length=input_chunk_length,
                    target_length=target_length,
                    g_type=_GType.TREND,
                )
            elif stack_type == "seasonality":
                stack = _Stack(
                    num_blocks=num_blocks,
                    num_layers=num_layers,
                    exogenous_dim=0,
                    layer_width=layer_widths[stack_id],
                    nr_params=nr_params,
                    expansion_coefficient_dim=expansion_coefficient_dim,
                    input_chunk_length=input_chunk_length,
                    target_length=target_length,
                    g_type=_GType.SEASONALITY,
                )
            elif stack_type == "exogenous":
                stack = _Stack(
                    num_blocks=num_blocks,
                    num_layers=num_layers,
                    exogenous_dim=self.hparams.exogenous_dim,
                    layer_width=layer_widths[stack_id],
                    nr_params=nr_params,
                    expansion_coefficient_dim=expansion_coefficient_dim,
                    input_chunk_length=input_chunk_length,
                    target_length=target_length,
                    g_type=_GType.EXOGENOUS,
                )
            else:
                raise ValueError(f"Unknown stack type {stack_type}")

            self.stacks.append(stack)

    @property
    def exogenous_reals_positions(self) -> List[int]:
        return [
            self.hparams.reals.index(name)
            for name in self.hparams.time_varying_known_reals
        ]

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # if x1, x2,... y1, y2... is one multivariate ts containing x and y, and a1, a2... one covariate ts
        # we reshape into x1, y1, a1, x2, y2, a2... etc

        input_x = x["encoder_target"][..., 0]
        batch_size = input_x.shape[0]

        exogenous_x = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        exogenous_x = exogenous_x[..., self.exogenous_reals_positions]

        # One vector of length target_length per parameter in the distribution
        y = torch.zeros(
            batch_size,
            self.hparams.prediction_length,
            self.nr_params,
            dtype=input_x.dtype,
        )

        for stack in self.stacks:
            # compute stack output
            stack_residual, stack_forecast = stack(x=input_x, exogenous_x=exogenous_x)

            # add stack forecast to final output
            y = y + stack_forecast
            # set current stack residual as input for next stack
            input_x = stack_residual

        # In multivariate case, we get a result [x1_param1, x1_param2], [y1_param1, y1_param2], [x2..], [y2..], ...
        # We want to reshape to original format. We also get rid of the covariates and keep only the target dimensions.
        # The covariates are by construction added as extra time series on the right side. So we need to get rid of this
        # right output (keeping only :self.output_dim).
        y = y.view(y.shape[0], self.hparams.prediction_length, self.nr_params)
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
            "exogenous_dim": len(dataset.time_varying_known_reals),
            "time_varying_known_reals": dataset.time_varying_known_reals,
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
