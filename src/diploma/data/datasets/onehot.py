from typing import List, Dict, Optional, Callable, Any, Tuple
import json
from os.path import join

import torch

import torch.utils.data

from diploma.data.datasets.components.onehot import OneHotDataset

import torch
import lightning.pytorch as pl

from pathlib import Path


class OneHotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        splits_path: str,
        samples_dir: str,
        augmentation: bool,
        input_size: Tuple[int, int],
        categories_json: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.splits_path = splits_path
        self.samples_dir = samples_dir
        self.augmentation = augmentation
        self.input_size = input_size
        self.categories = json.loads(Path(categories_json).read_text())

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.splits = json.loads(Path(self.splits_path).read_text())

        self.train_dataset = OneHotDataset(
            self.samples_dir,
            self.splits["train"],
            self.augmentation,
            self.input_size,
            self.categories,
        )

        self.val_dataset = OneHotDataset(
            self.samples_dir,
            self.splits["val"],
            False,
            self.input_size,
            self.categories,
        )

        self.test_dataset = OneHotDataset(
            self.samples_dir,
            self.splits["val"],
            False,
            self.input_size,
            self.categories,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=True,
        )
