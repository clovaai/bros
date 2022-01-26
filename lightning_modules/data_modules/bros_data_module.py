"""
BROS
Copyright 2022-present NAVER Corp.
Apache License v2.0
"""

import time

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.utils.data.dataloader import DataLoader

from lightning_modules.data_modules.bros_dataset import BROSDataset


class BROSDataModule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None
        self.tokenizer = tokenizer
        self.collate_fn = None

        if self.cfg.model.backbone in [
            "naver-clova-ocr/bros-base-uncased",
            "naver-clova-ocr/bros-large-uncased",
        ]:
            self.backbone_type = "bros"
        elif self.cfg.model.backbone in [
            "microsoft/layoutlm-base-uncased",
            "microsoft/layoutlm-large-uncased",
        ]:
            self.backbone_type = "layoutlm"
        else:
            raise ValueError(
                f"Not supported model: self.cfg.model.backbone={self.cfg.model.backbone}"
            )

    @overrides
    def setup(self, stage=None):
        self.train_loader = self._get_train_loader()
        self.val_loader = self._get_val_test_loaders(mode="val")

    @overrides
    def train_dataloader(self):
        return self.train_loader

    @overrides
    def val_dataloader(self):
        return self.val_loader

    def _get_train_loader(self):
        start_time = time.time()

        dataset = BROSDataset(
            self.cfg.dataset,
            self.cfg.task,
            self.backbone_type,
            self.cfg.model.head,
            self.cfg.dataset_root_path,
            self.tokenizer,
            self.cfg.train.max_seq_length,
            mode="train",
        )

        data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )

        elapsed_time = time.time() - start_time
        print(f"Elapsed time for loading training data: {elapsed_time}", flush=True)

        return data_loader

    def _get_val_test_loaders(self, mode):
        dataset = BROSDataset(
            self.cfg.dataset,
            self.cfg.task,
            self.backbone_type,
            self.cfg.model.head,
            self.cfg.dataset_root_path,
            self.tokenizer,
            mode=mode,
        )

        data_loader = DataLoader(
            dataset,
            batch_size=self.cfg[mode].batch_size,
            shuffle=False,
            num_workers=self.cfg[mode].num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return data_loader

    @overrides
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        return batch
