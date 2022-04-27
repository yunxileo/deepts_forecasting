"""
Loss functions.
"""

import torch
from typing import Dict, List, Tuple, Union
from torchmetrics import Metric as LightningMetric
from sklearn.base import BaseEstimator


class Metric(LightningMetric):
    """
    Base metric class that has basic functions that can handle predicting quantiles and operate in log space.
    See the `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/latest/metrics.html>`_
    for details of how to implement a new metric

    Other metrics should inherit from this base class
    """

    def __init__(
        self, name: str = None, quantiles: List[float] = None, reduction="mean"
    ):
        """
        Initialize metric

        Args:
            name (str): metric name. Defaults to class name.
            quantiles (List[float], optional): quantiles for probability range. Defaults to None.
            reduction (str, optional): Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
        """
        self.quantiles = quantiles
        self.reduction = reduction
        if name is None:
            name = self.__class__.__name__
        self.name = name
        super().__init__()

    def update(y_pred: torch.Tensor, y_actual: torch.Tensor):
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        """
        Abstract method that calcualtes metric

        Should be overriden in derived classes

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        raise NotImplementedError()

    def rescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        """
        Rescale normalized parameters into the scale required for the output.

        Args:
            parameters (torch.Tensor): normalized parameters (indexed by last dimension)
            target_scale (torch.Tensor): scale of parameters (n_batch_samples x (center, scale))
            encoder (BaseEstimator): original encoder that normalized the target in the first place

        Returns:
            torch.Tensor: parameters in real/not normalized space
        """
        return encoder(dict(prediction=parameters, target_scale=target_scale))

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            if self.quantiles is None:
                assert (
                    y_pred.size(-1) == 1
                ), "Prediction should only have one extra dimension"
                y_pred = y_pred[..., 0]
            else:
                y_pred = y_pred.mean(-1)
        return y_pred

    def to_quantiles(
        self, y_pred: torch.Tensor, quantiles: List[float] = None
    ) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network
            quantiles (List[float], optional): quantiles for probability range. Defaults to quantiles as
                as defined in the class initialization.

        Returns:
            torch.Tensor: prediction quantiles
        """
        if quantiles is None:
            quantiles = self.quantiles

        if y_pred.ndim == 2:
            return y_pred.unsqueeze(-1)
        elif y_pred.ndim == 3:
            if y_pred.size(2) > 1:  # single dimension means all quantiles are the same
                assert quantiles is not None, "quantiles are not defined"
                y_pred = torch.quantile(
                    y_pred, torch.tensor(quantiles, device=y_pred.device), dim=2
                ).permute(1, 2, 0)
            return y_pred
        else:
            raise ValueError(
                f"prediction has 1 or more than 3 dimensions: {y_pred.ndim}"
            )


class MAE(Metric):
    """
    Mean average absolute error.

    Defined as ``(y_pred - target).abs()``
    """

    def loss(self, y_pred, target):
        loss = (self.to_prediction(y_pred) - target).abs()
        return loss
