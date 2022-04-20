"""
reference:https://github.com/gautham20/pytorch-ts/blob/master/torch_utils/trainer.py
"""

import pathlib
from copy import deepcopy
from logging import Logger
from typing import Dict, Optional, Type, Union

import numpy as np
import torch
from deepts_forecasting.logging import get_logger, raise_log
from deepts_forecasting.models import BaseModel
from deepts_forecasting.utils import save_dict
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        logger: Union[Logger, bool] = True,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        model_name: str = None,
        model: BaseModel = None,
        optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = {},
        lr_scheduler_cls: torch.optim.lr_scheduler._LRScheduler = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        loss_fn: nn.modules.loss._Loss = nn.L1Loss(),
        additional_metric_fns: Dict = None,
        scheduler_batch_step: bool = False,
        device: str = "auto",
        **kwargs,
    ):
        """

        Args:
            name:
            model:
            optimizer:
            loss_fn:
            scheduler:
            device:
        """
        # initialize device type
        self._init_device(device)

        # configure logger
        self.configure_logger(logger)
        # log interval
        self.log_interval = log_interval
        self.log_val_interval = log_val_interval

        # model
        self.model_name = model_name
        self.model = model.to(self.device)

        # loss function
        self.loss_fn = loss_fn

        # initialize optimizer and lr_scheduler
        self._init_optimizer(cls=optimizer_cls, kws=optimizer_kwargs)
        self._init_lr_scheduler(
            cls=lr_scheduler_cls, kws=lr_scheduler_kwargs
        )  # self.lr_scheduler can be None

        # checkpoint path
        self.checkpoint_path = pathlib.Path(
            kwargs.get("checkpoint_folder", f"./models/{model_name}_chkpts")
        )
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        # checkpoint interval
        self.train_checkpoint_interval = kwargs.get("train_checkpoint_interval", 1)
        # maximum checkpoints
        self.max_checkpoints = kwargs.get("max_checkpoints", 25)

        # a dict to record checkpoints saved in path currently
        # key: epoch; value:valid loss (if calculated)
        self.saved_checkpoints = {}

        self.scheduler_batch_step = scheduler_batch_step

        self.additional_metric_fns = (
            {} if additional_metric_fns is None else additional_metric_fns
        )
        self.additional_metric_fns = self.additional_metric_fns.items()

        self.valid_losses = {}

    def _init_device(self, device):
        """
        Initialize device type.

        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device in ["cpu", "cuda"]:
            self.device = torch.device(device)
        else:
            raise ValueError(
                f"Device should be 'auto', 'cpu' or 'cuda'. '{device}' is not supported."
            )

    def _create_from_cls_and_kwargs(self, cls, kws):
        """

        Args:
            cls:
            kws:

        """
        try:
            return cls(**kws)
        except (TypeError, ValueError) as e:
            raise_log(
                ValueError(
                    "Error when building the optimizer or learning rate scheduler;"
                    "please check the provided class and arguments"
                    "\nclass: {}"
                    "\narguments (kwargs): {}"
                    "\nerror:\n{}".format(cls, kws, e)
                ),
                self.logger,
            )

    def _init_optimizer(self, cls: torch.optim.Optimizer, kws: Optional[Dict]):
        optimizer_kws = deepcopy(kws)
        optimizer_kws["params"] = self.model.parameters()
        self.optimizer = self._create_from_cls_and_kwargs(cls, optimizer_kws)

    def _init_lr_scheduler(
        self, cls: torch.optim.lr_scheduler._LRScheduler, kws: Optional[Dict]
    ):
        if cls is not None:
            lr_sched_kws = deepcopy(kws)
            lr_sched_kws["optimizer"] = self.optimizer
            self.lr_scheduler = self._create_from_cls_and_kwargs(cls, lr_sched_kws)
        else:
            self.lr_scheduler = None

    def configure_logger(self, logger: Union[Logger, bool]):
        """
        Configure logger for logging.

        """
        if logger is True:
            self.logger = get_logger(__name__)
        elif logger is False:
            self.logger = None
        else:
            self.logger = logger

    def refresh_checkpoints(self, checkpoint_path: str = None):
        """
        If checkpoints have existed before `Trainer` class is instantiated, you can use this method to refresh self.saved_checkpoints.

        """
        if checkpoint_path is not None:
            assert (
                self.checkpoint_path == checkpoint_path
            ), "Refresh path is not the same as running path."
        else:
            checkpoint_path = self.checkpoint_path

        for cp in checkpoint_path.glob("checkpoint_*"):
            cp_file = torch.load(cp)
            if "loss" in cp_file:
                self.saved_checkpoints.update({cp_file["epoch"]: cp_file["loss"]})
            else:
                break

    def _get_checkpoints(self, name: str = None):
        """

        Args:
            name (str): model name.

        Returns:

        """
        checkpoints = []
        checkpoint_path = (
            self.checkpoint_path
            if name is None
            else pathlib.Path(f"./models/{name}_chkpts")
        )
        for cp in checkpoint_path.glob("checkpoint_*"):
            checkpoint_name = str(cp).split("/")[-1]
            checkpoint_epoch = int(checkpoint_name.split("_")[-1])
            checkpoints.append(
                (
                    cp,
                    checkpoint_epoch,
                    self.saved_checkpoints.get(checkpoint_epoch),
                    None,
                )
            )
        if len(self.saved_checkpoints) == 0:
            checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
        else:
            checkpoints = sorted(checkpoints, key=lambda x: x[-1], reverse=True)

        return checkpoints

    def _drop_checkpoints(self) -> None:
        """ """
        checkpoints = self._get_checkpoints()
        if len(checkpoints) > self.max_checkpoints:
            if len(self.saved_checkpoints) == 0:
                checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
            else:
                checkpoints = sorted(checkpoints, key=lambda x: x[-1], reverse=True)
            for delete_cp in checkpoints[self.max_checkpoints :]:
                del self.saved_checkpoints[delete_cp[1]]
                delete_cp[0].unlink()
                print(f"removed checkpoint of epoch - {delete_cp[1]}")

    def _save_checkpoint(self, epoch: int, valid_loss: Optional[float] = None):
        """
        Args:
            epoch (int): the number of epoch
            valid_loss (Optional[float]): loss on validation set
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            checkpoint.update({"scheduler_state_dict": self.lr_scheduler.state_dict()})
        if valid_loss:
            checkpoint.update({"loss": valid_loss})
            self.saved_checkpoints.update({epoch: valid_loss})
        torch.save(checkpoint, self.checkpoint_path / f"checkpoint_{epoch}")
        save_dict(self.checkpoint_path, "valid_losses", self.valid_losses)
        print(f"saved checkpoint for epoch {epoch}")
        self._drop_checkpoints()

    def _load_checkpoint(self, epoch=None, only_model=False, name=None):
        """
        Args:
            epoch:
            only_model:
            name:
        """
        if name is None:
            checkpoints = self._get_checkpoints()
        else:
            checkpoints = self._get_checkpoints(name)
        if len(checkpoints) > 0:
            if not epoch:
                checkpoint_config = checkpoints[0]
            else:
                checkpoint_config = list(filter(lambda x: x[1] == epoch, checkpoints))[
                    0
                ]
            checkpoint = torch.load(checkpoint_config[0])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if not only_model:
                if type(self.optimizer) is list:
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(
                            checkpoint["optimizer_state_dict"][i]
                        )
                else:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.lr_scheduler is not None:
                    if type(self.lr_scheduler) is list:
                        for i in range(len(self.lr_scheduler)):
                            self.lr_scheduler[i].load_state_dict(
                                checkpoint["scheduler_state_dict"][i]
                            )
                    else:
                        self.lr_scheduler.load_state_dict(
                            checkpoint["scheduler_state_dict"]
                        )
            print(f'loaded checkpoint for epoch - {checkpoint["epoch"]}')
            return checkpoint["epoch"]
        return None

    def load_best_checkpoint(self):
        """
        TODO: check this method
        """
        if self.valid_losses:
            best_epoch = sorted(self.valid_losses.items(), key=lambda x: x[1])[0][0]
            loaded_epoch = self._load_checkpoint(epoch=best_epoch, only_model=True)
        return loaded_epoch

    def _step_optim(self):
        """ """
        if isinstance(self.optimizer, list):
            for i in range(len(self.optimizer)):
                self.optimizer[i].step()
                self.optimizer[i].zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _step_scheduler(self, valid_loss=None):
        """ """
        if isinstance(self.lr_scheduler, list):
            for i in range(len(self.lr_scheduler)):
                if self.lr_scheduler[i].__class__.__name__ == "ReduceLROnPlateau":
                    self.lr_scheduler[i].step(valid_loss)
                else:
                    self.lr_scheduler[i].step()
        else:
            if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.lr_scheduler.step(valid_loss)
            else:
                self.lr_scheduler.step()

    def _loss_batch(self, xb, yb, optimize, additional_metrics=None):
        """
        Calculate loss on a batch.

        Args:
            xb: input data on a batch
            yb: target data on a batch
            optimize: optimizer
            additional_metrics (dict):

        """
        if isinstance(xb, dict):
            xb = {xbi: xb[xbi].to(self.device) for xbi in xb.keys()}
        else:
            raise TypeError(
                "Output of dataloader should be `Tuple[Dict, torch.Tensor]`"
            )
        yb = yb.to(self.device)
        yb = yb.squeeze(-1)
        y_pred = self.model(xb)
        loss = self.loss_fn(y_pred, yb)

        if additional_metrics is not None:
            additional_metrics = {
                name: fn(y_pred, yb) for name, fn in additional_metrics
            }
        if optimize:
            loss.backward()
            self._step_optim()
        loss_value = loss.item()
        del xb
        del yb
        del y_pred
        del loss
        if additional_metrics is not None:
            return loss_value, additional_metrics
        return loss_value

    def evaluate(self, dataloader):
        """ """
        self.model.eval()
        eval_bar = tqdm(dataloader, leave=False)
        with torch.no_grad():
            loss_values = [
                self._loss_batch(xb, yb, False, self.additional_metric_fns)
                for xb, yb in eval_bar
            ]
            if len(loss_values[0]) > 1:
                loss_value = np.mean([lv[0] for lv in loss_values])
                additional_metrics = np.mean([lv[1] for lv in loss_values], axis=0)
                additional_metrics_result = {
                    name: result
                    for (name, fn), result in zip(
                        self.additional_metric_fns, additional_metrics
                    )
                }
                return loss_value, additional_metrics_result
            # eval_bar.set_description("evaluation loss %.2f" % loss_value)
            else:
                loss_value = np.mean(loss_values)
                return loss_value, None

    def predict(self, dataloader):
        """ """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for xb, yb in tqdm(dataloader):
                if isinstance(xb, dict):
                    xb = {xbi: xb[xbi].to(self.device) for xbi in xb.keys()}
                else:
                    raise TypeError(
                        "Output of dataloader should be `Tuple[Dict, torch.Tensor]`"
                    )
                yb = yb.to(self.device)
                y_pred = self.model(xb)
                predictions.append(y_pred.cpu().numpy())
        return np.concatenate(predictions)

    def fit(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader = None,
        resume=False,
        resume_only_model=False,
    ):
        """
        Args:
            epochs (int): number of epochs
            train_dataloader (torch.utils.data.DataLoader): train set
            valid_dataloader (torch.utils.data.DataLoader): validation set
            resume (bool): if train on a given checkpoint
            resume_only_model (bool): if only train on a model of a given checkpoint

        """
        start_epoch = 0
        if resume:
            loaded_epoch = self._load_checkpoint(only_model=resume_only_model)
            if loaded_epoch:
                start_epoch = loaded_epoch
        for i in tqdm(range(start_epoch, start_epoch + epochs), leave=True):
            self.model.train()
            epoch_losses = []
            training_bar = tqdm(train_dataloader, leave=False)
            for it, (xb, yb) in enumerate(training_bar):
                loss = self._loss_batch(xb, yb, True)
                # running_loss += loss
                training_bar.set_description("loss %.4f" % loss)
                epoch_losses.append(loss)

                if self.lr_scheduler is not None and self.scheduler_batch_step:
                    self._step_scheduler()
            print(f"Training loss at epoch {i + 1} - {np.mean(epoch_losses)}")
            if valid_dataloader is not None:
                valid_loss, additional_metrics = self.evaluate(valid_dataloader)
                if additional_metrics is not None:
                    print(additional_metrics)
                print(f"Valid loss at epoch {i + 1} - {valid_loss}")
                self.valid_losses[i + 1] = valid_loss
            if self.lr_scheduler is not None and not self.scheduler_batch_step:
                self._step_scheduler(valid_loss)
            if (i + 1) % self.train_checkpoint_interval == 0:
                self._save_checkpoint(i + 1)

    def plot_prediction(
        self, x: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], idx: int = 0
    ):
        """
        Plot prediction of prediction vs actuals.

        Args:
            x: network input
            pred: network prediction
            idx: index of prediction to plot


        """
        pass
