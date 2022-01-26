"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

import time

import torch
import torch.utils.data
from overrides import overrides
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from lightning_modules.schedulers import (
    cosine_scheduler,
    linear_scheduler,
    multistep_scheduler,
)
from model import get_model
from utils import cfg_to_hparams, get_specific_pl_logger


class BROSModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = get_model(self.cfg)
        self.ignore_index = -100

        self.time_tracker = None

        self.optimizer_types = {
            "sgd": SGD,
            "adam": Adam,
            "adamw": AdamW,
        }

    @overrides
    def setup(self, stage):
        self.time_tracker = time.time()

    @overrides
    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer)
        scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def _get_lr_scheduler(self, optimizer):
        cfg_train = self.cfg.train
        lr_schedule_method = cfg_train.optimizer.lr_schedule.method
        lr_schedule_params = cfg_train.optimizer.lr_schedule.params

        if lr_schedule_method is None:
            scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1)
        elif lr_schedule_method == "step":
            scheduler = multistep_scheduler(optimizer, **lr_schedule_params)
        elif lr_schedule_method == "cosine":
            total_samples = cfg_train.max_epochs * cfg_train.num_samples_per_epoch
            total_batch_size = cfg_train.batch_size * self.trainer.world_size
            max_iter = total_samples / total_batch_size
            scheduler = cosine_scheduler(
                optimizer, training_steps=max_iter, **lr_schedule_params
            )
        elif lr_schedule_method == "linear":
            total_samples = cfg_train.max_epochs * cfg_train.num_samples_per_epoch
            total_batch_size = cfg_train.batch_size * self.trainer.world_size
            max_iter = total_samples / total_batch_size
            scheduler = linear_scheduler(
                optimizer, training_steps=max_iter, **lr_schedule_params
            )
        else:
            raise ValueError(f"Unknown lr_schedule_method={lr_schedule_method}")

        return scheduler

    def _get_optimizer(self):
        opt_cfg = self.cfg.train.optimizer
        method = opt_cfg.method.lower()

        if method not in self.optimizer_types:
            raise ValueError(f"Unknown optimizer method={method}")

        kwargs = dict(opt_cfg.params)
        kwargs["params"] = self.net.parameters()
        optimizer = self.optimizer_types[method](**kwargs)

        return optimizer

    @rank_zero_only
    @overrides
    def on_fit_end(self):
        hparam_dict = cfg_to_hparams(self.cfg, {})
        metric_dict = {"metric/dummy": 0}

        tb_logger = get_specific_pl_logger(self.logger, TensorBoardLogger)

        if tb_logger:
            tb_logger.log_hyperparams(hparam_dict, metric_dict)

    @overrides
    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.tensor(0.0).to(self.device)
        for step_out in training_step_outputs:
            avg_loss += step_out["loss"]

        log_dict = {"train_loss": avg_loss}
        self._log_shell(log_dict, prefix="train ")

    def _log_shell(self, log_info, prefix=""):
        log_info_shell = {}
        for k, v in log_info.items():
            new_v = v
            if type(new_v) is torch.Tensor:
                new_v = new_v.item()
            log_info_shell[k] = new_v

        out_str = prefix.upper()
        if prefix.upper().strip() in ["TRAIN", "VAL"]:
            out_str += f"[epoch: {self.current_epoch}/{self.cfg.train.max_epochs}]"

        if self.training:
            lr = self.trainer._lightning_optimizers[0].param_groups[0]["lr"]
            log_info_shell["lr"] = lr

        for key, value in log_info_shell.items():
            out_str += f" || {key}: {round(value, 5)}"
        out_str += f" || time: {round(time.time() - self.time_tracker, 1)}"
        out_str += " secs."
        self.print(out_str, flush=True)
        self.time_tracker = time.time()
