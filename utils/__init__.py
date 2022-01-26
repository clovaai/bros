"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

import os

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


def get_config(default_conf_file="./configs/default.yaml"):
    cfg = OmegaConf.load(default_conf_file)

    cfg_cli = _get_config_from_cli()
    if "config" in cfg_cli:
        cfg_cli_config = OmegaConf.load(cfg_cli.config)
        cfg = OmegaConf.merge(cfg, cfg_cli_config)
        del cfg_cli["config"]

    cfg = OmegaConf.merge(cfg, cfg_cli)

    _check_config(cfg)
    _update_config(cfg)

    return cfg


def _get_config_from_cli():
    cfg_cli = OmegaConf.from_cli()
    cli_keys = list(cfg_cli.keys())
    for cli_key in cli_keys:
        if "--" in cli_key:
            cfg_cli[cli_key.replace("--", "")] = cfg_cli[cli_key]
            del cfg_cli[cli_key]

    return cfg_cli


def _check_config(cfg):
    assert cfg.dataset in ["funsd", "sroie"]
    if cfg.dataset == "funsd":
        assert cfg.task in ["ee", "el"]
        if cfg.task == "ee":
            assert cfg.model.head in ["bies", "spade"]
        elif cfg.task == "el":
            assert cfg.model.head == "spade_rel"
    elif cfg.dataset == "sroie":
        assert cfg.task == "ee"
        if cfg.task == "ee":
            assert cfg.model.head == "bio"


def _update_config(cfg):
    cfg.save_weight_dir = os.path.join(cfg.workspace, "checkpoints")
    cfg.tensorboard_dir = os.path.join(cfg.workspace, "tensorboard_logs")

    if cfg.dataset == "funsd":
        if cfg.task == "ee":
            if cfg.model.head == "bies":
                cfg.dataset_root_path = "./datasets/funsd"
                cfg.model.n_classes = 13
            elif cfg.model.head == "spade":
                cfg.dataset_root_path = "./datasets/funsd_spade"
                cfg.model.n_classes = 3
        elif cfg.task == "el":
            cfg.dataset_root_path = "./datasets/funsd_spade"
            cfg.model.n_classes = 3
    elif cfg.dataset == "sroie":
        if cfg.task == "ee":
            if cfg.model.head == "bio":
                cfg.dataset_root_path = "./datasets/sroie"
                cfg.model.n_classes = 2 * 4 + 1

    # set per-gpu batch size
    num_devices = torch.cuda.device_count()
    for mode in ["train", "val"]:
        new_batch_size = cfg[mode].batch_size // num_devices
        cfg[mode].batch_size = new_batch_size


def get_callbacks(cfg):
    callbacks = []

    cb = LastestModelCheckpoint(
        dirpath=cfg.save_weight_dir, save_top_k=0, save_last=True
    )
    cb.CHECKPOINT_NAME_LAST = "{epoch}-last"
    cb.FILE_EXTENSION = ".pt"
    callbacks.append(cb)

    return callbacks


class LastestModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_end(self, trainer, pl_module):
        """Save the latest model at every train epoch end."""
        self.save_checkpoint(trainer)


def get_plugins(cfg):
    plugins = []

    if cfg.train.strategy.type == "ddp":
        plugins.append(DDPPlugin())

    return plugins


def get_loggers(cfg):
    loggers = []

    loggers.append(
        TensorBoardLogger(
            cfg.tensorboard_dir, name="", version="", default_hp_metric=False
        )
    )

    return loggers


def cfg_to_hparams(cfg, hparam_dict, parent_str=""):
    for key, val in cfg.items():
        if isinstance(val, DictConfig):
            hparam_dict = cfg_to_hparams(val, hparam_dict, parent_str + key + "__")
        else:
            hparam_dict[parent_str + key] = str(val)
    return hparam_dict


def get_specific_pl_logger(pl_loggers, logger_type):
    for pl_logger in pl_loggers:
        if isinstance(pl_logger, logger_type):
            return pl_logger
    return None


def get_class_names(dataset_root_path):
    class_names_file = os.path.join(dataset_root_path, "class_names.txt")
    class_names = (
        open(class_names_file, "r", encoding="utf-8").read().strip().split("\n")
    )
    return class_names
