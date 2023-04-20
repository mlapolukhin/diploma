import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
import pytorch_lightning as pl

import os
from typing import Dict, Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    model = instantiate(cfg.model)
    optimizer = instantiate(
        cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters())
    )
    model.set_optimizer(optimizer)
    datamodule = instantiate(cfg.data)
    trainer = instantiate(cfg.trainer)

    if cfg.task_type == "train":
        trainer.fit(
            model=model,
            datamodule=datamodule,
        )

    if cfg.task_type == "test":
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
        )


if __name__ == "__main__":
    main()
