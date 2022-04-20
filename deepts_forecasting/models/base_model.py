"""
Implementations of base models.

TO-DO:
1. transform_output
2. on_save_checkpoint
3. on_load_checkpoint
4. all of log
"""

import inspect
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from deepts_forecasting.utils.data import TimeSeriesDataSet
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.parsing import get_init_args
from torch import nn
from torch.utils.data import DataLoader


class BaseModel(LightningModule, ABC):
    """ """

    def __init__(
        self,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        learning_rate: Union[float, List[float]] = 1e-3,
        logging_metrics: nn.ModuleList = nn.ModuleList([]),
        loss=nn.MSELoss(),
        output_transformer: Callable = None,
        monotone_constaints={},
    ):
        super().__init__()
        # update hparams
        frame = inspect.currentframe()
        init_args = get_init_args(frame)
        self.save_hyperparameters(
            {name: val for name, val in init_args.items() if name not in self.hparams}
        )

        self.loss = loss
        self.monotone_constaints = monotone_constaints
        if not hasattr(self, "logging_metrics"):
            self.logging_metrics = nn.ModuleList([metric for metric in logging_metrics])
        if not hasattr(self, "output_transformer"):
            self.output_transformer = output_transformer

    def size(self):
        """
        get number of parameters in model

        """
        return sum(p.numel() for p in self.parameters())

    def training_step(self, batch, batch_idx):
        """
        Train on batch
        Args:
            batch:

        Returns:

        """
        x, y = batch
        log, _ = self.step(x, y)
        self.log("train_loss", log["loss"], on_step=False, on_epoch=True, prog_bar=True)
        return log

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log, _ = self.step(x, y)
        self.log("val_loss", log["loss"], on_step=False, on_epoch=True, prog_bar=True)
        return log

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}

    def epoch_end(self, outputs):
        """
        Run at epoch end for training or validation. Can be overriden in models.
        """
        pass

    def step(self, x, y, **kwargs):
        """
        Run for each train/val step
        Returns:
        """
        out = self(x, **kwargs)

        # calculate loss
        loss = self.loss(out, y)

        # log
        self.log_metrics(y, out)
        log = {"loss": loss}

        return log, out

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            idx = y_pred.size(-1) // 2
            y_pred = y_pred[..., idx]
        return y_pred

    def log_metrics(
        self,
        y,
        out,
    ) -> None:
        """
        Log metrics every training/validation step.

        Args:
            x: x as passed to the network by the dataloader
            y: y as passed to the loss function by the dataloader
            out: output of the network
        """
        # logging losses
        y_hat_detached = out.detach()
        y_hat_point_detached = self.to_prediction(y_hat_detached)
        for metric in self.logging_metrics:
            loss_value = metric(y_hat_point_detached, y)
            self.log(
                f"{['val', 'train'][self.training]}_{metric.name}",
                loss_value,
                on_step=self.training,
                on_epoch=True,
            )

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

    @classmethod
    def from_dataset(cls, dataset, **kwargs) -> LightningModule:
        """
        Create model from dataset, i.e. save dataset parameters in model

        This function should be called as ``super().from_dataset()`` in a derived models
        that implement it

        Args:
            dataset (TimeSeriesDataSet): timeseries dataset

        Returns:
            BaseModel: Model that can be trained
        """
        if "output_transformer" not in kwargs:
            kwargs["output_transformer"] = dataset.target_normalizer
        net = cls(**kwargs)
        net.dataset_parameters = dataset.get_parameters()
        return net

    def forward(self, x):
        """
        Network forward pass.

        Args:
            x (Dict[str, torch.Tensor]): network input (x as returned by the dataloader)

        Returns:
            Dict[str, torch.Tensor]: network outputs - includes at entries for ``prediction`` and ``target_scale``
        """
        raise NotImplementedError()

    def transform_output(
        self,
        prediction: Union[torch.Tensor, List[torch.Tensor]],
        target_scale: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Extract prediction from network output and rescale it to real space / de-normalize it.

        Args:
            prediction (Union[torch.Tensor, List[torch.Tensor]]): normalized prediction
            target_scale (Union[torch.Tensor, List[torch.Tensor]]): scale to rescale prediction

        Returns:
            torch.Tensor: rescaled prediction
        """
        out = self.output_transformer(
            dict(prediction=prediction, target_scale=target_scale)
        )
        return out

    def predict(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        batch_size: int = 32,
        num_workers: int = 0,
        return_index: bool = True,
        **kwargs,
    ):
        # convert to dataloader
        if isinstance(data, pd.DataFrame):
            data = TimeSeriesDataSet.from_parameters(
                self.dataset_parameters, data, predict=True
            )
        if isinstance(data, TimeSeriesDataSet):
            dataloader = DataLoader(
                data, batch_size=batch_size, train=False, num_workers=num_workers
            )
        else:
            dataloader = data
        # ensure passed dataloader is correct
        assert isinstance(
            dataloader.dataset, TimeSeriesDataSet
        ), "dataset behind dataloader mut be TimeSeriesDataSet"
        # prepare model
        self.eval()  # no dropout, etc. no gradients
        output = []
        index = []
        with torch.no_grad():
            for x, _ in dataloader:
                # move data to appropriate device
                # if x != self.device:
                #     x = x.to(self.device)
                # make prediction
                out = self(x, **kwargs)  #
                out = self.transform_output(
                    prediction=out, target_scale=x["target_scale"]
                )
                output.append(out)
                if return_index:
                    index.append(dataloader.dataset.x_to_index(x))
        if return_index:
            x_index = pd.concat(index, axis=0, ignore_index=True)
        return torch.cat(output), x_index


class BaseModelWithCovariates(BaseModel):
    """
    Model with additional methods using covariates.

    Assumes the following hyperparameters:

    Args:
        static_categoricals (List[str]): names of static categorical variables
        static_reals (List[str]): names of static continuous variables
        time_varying_categoricals_encoder (List[str]): names of categorical variables for encoder
        time_varying_categoricals_decoder (List[str]): names of categorical variables for decoder
        time_varying_reals_encoder (List[str]): names of continuous variables for encoder
        time_varying_reals_decoder (List[str]): names of continuous variables for decoder
        x_reals (List[str]): order of continuous variables in tensor passed to forward function
        x_categoricals (List[str]): order of categorical variables in tensor passed to forward function
        embedding_sizes (Dict[str, Tuple[int, int]]): dictionary mapping categorical variables to tuple of integers
            where the first integer denotes the number of categorical classes and the second the embedding size
        embedding_labels (Dict[str, List[str]]): dictionary mapping (string) indices to list of categorical labels
        embedding_paddings (List[str]): names of categorical variables for which label 0 is always mapped to an
             embedding vector filled with zeros
        categorical_groups (Dict[str, List[str]]): dictionary of categorical variables that are grouped together and
            can also take multiple values simultaneously (e.g. holiday during octoberfest). They should be implemented
            as bag of embeddings
    """

    @property
    def reals(self) -> List[str]:
        """List of all continuous variables in model"""
        return list(
            dict.fromkeys(
                self.hparams.static_reals
                + self.hparams.time_varying_reals_encoder
                + self.hparams.time_varying_reals_decoder
            )
        )

    @property
    def categoricals(self) -> List[str]:
        """List of all categorical variables in model"""
        return list(
            dict.fromkeys(
                self.hparams.static_categoricals
                + self.hparams.time_varying_categoricals_encoder
                + self.hparams.time_varying_categoricals_decoder
            )
        )

    @property
    def static_variables(self) -> List[str]:
        """List of all static variables in model"""
        return self.hparams.static_categoricals + self.hparams.static_reals

    @property
    def encoder_variables(self) -> List[str]:
        """List of all encoder variables in model (excluding static variables)"""
        return (
            self.hparams.time_varying_categoricals_encoder
            + self.hparams.time_varying_reals_encoder
        )

    @property
    def decoder_variables(self) -> List[str]:
        """List of all decoder variables in model (excluding static variables)"""
        return (
            self.hparams.time_varying_categoricals_decoder
            + self.hparams.time_varying_reals_decoder
        )

    @property
    def categorical_groups_mapping(self) -> Dict[str, str]:
        """Mapping of categorical variables to categorical groups"""
        groups = {}
        for group_name, sublist in self.hparams.categorical_groups.items():
            groups.update({name: group_name for name in sublist})
        return groups

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ) -> LightningModule:
        """
        Create model from dataset and set parameters related to covariates.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            LightningModule
        """
        # assert fixed encoder and decoder length for the moment
        if allowed_encoder_known_variable_names is None:
            allowed_encoder_known_variable_names = (
                dataset.time_varying_known_categoricals
                + dataset.time_varying_known_reals
            )

        # embeddings
        embedding_labels = {
            name: encoder.classes_
            for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals
        }
        # determine embedding sizes based on heuristic
        embedding_sizes = {
            name: (len(encoder.classes_), cls.get_embedding_size(len(encoder.classes_)))
            for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals
        }
        embedding_sizes.update(kwargs.get("embedding_sizes", {}))
        kwargs.setdefault("embedding_sizes", embedding_sizes)

        new_kwargs = dict(
            max_encoder_length=dataset.max_encoder_length,
            max_prediction_length=dataset.max_prediction_length,
            static_categoricals=dataset.static_categoricals,
            time_varying_categoricals_encoder=[
                name
                for name in dataset.time_varying_known_categoricals
                if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_categoricals,
            time_varying_categoricals_decoder=dataset.time_varying_known_categoricals,
            static_reals=dataset.static_reals,
            time_varying_reals_encoder=[
                name
                for name in dataset.time_varying_known_reals
                if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_reals,
            time_varying_reals_decoder=dataset.time_varying_known_reals,
            x_reals=dataset.reals,
            x_categoricals=dataset.flat_categoricals,
            embedding_labels=embedding_labels,
            categorical_groups=dataset.variable_groups,
        )
        new_kwargs.update(kwargs)
        return super().from_dataset(dataset, **new_kwargs)
