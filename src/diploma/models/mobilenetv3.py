import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchmetrics import MetricCollection, Metric
from torchmetrics import ConfusionMatrix
import lightning.pytorch as pl

import os
from typing import Callable, Dict, Any, List

from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from diploma.train.utils import plot_confusion_matrix
import torchvision
import json
from PIL import Image
import numpy as np


class MobileNetV3Module(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        metrics: MetricCollection,
        criterion: Callable,
        categories_json: str,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.categories = json.loads(Path(categories_json).read_text())

        self.metrics = dict()
        self.metrics["training"] = MetricCollection(
            {k: v for (k, v) in metrics["training"].items()}
        )
        self.metrics["validation"] = MetricCollection(
            {k: v for (k, v) in metrics["validation"].items()}
        )
        self.metrics["test"] = MetricCollection(
            {k: v for (k, v) in metrics["test"].items()}
        )
        # just for consistency with Lightning device management
        self._metrics = MetricCollection(
            {split: self.metrics[split] for split in self.metrics}
        )

        self.confusion_matrix = {
            f"cm_{split}": ConfusionMatrix(
                task="multiclass", num_classes=len(self.categories)
            )
            for split in self.metrics
        }
        # just for consistency with Lightning device management
        self._contusion_matrix = MetricCollection(
            {
                f"cm_{split}": self.confusion_matrix[f"cm_{split}"]
                for split in self.metrics
            }
        )

        self.automatic_optimization = False

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: Any):
        self.optimizers().zero_grad()

        output = self.model(batch)
        loss = self.criterion(output, batch)
        loss["total_loss"].backward()

        self.optimizers().step()
        self.optimizers().zero_grad()

        self.log(
            "loss/train",
            loss["total_loss"].item() / len(batch["inputs"]),
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        for loss_name, loss_value in loss["losses"].items():
            self.log(
                f"loss_{loss_name}/train",
                loss_value.item() / len(batch["inputs"]),
                prog_bar=True,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        self.update_metrics(output, batch, "training")

    def validation_step(self, batch, batch_idx):
        result = self._shared_eval_step(batch, batch_idx, "validation")
        self.log(
            "loss/validation",
            result["loss"]["total_loss"].item() / len(batch["inputs"]),
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        for loss_name, loss_value in result["loss"]["losses"].items():
            self.log(
                f"loss_{loss_name}/validation",
                loss_value.item() / len(batch["inputs"]),
                prog_bar=True,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx):
        result = self._shared_eval_step(batch, batch_idx, "test")
        self.log(
            "loss/test",
            result["loss"]["total_loss"].item() / len(batch["inputs"]),
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        for loss_name, loss_value in result["loss"]["losses"].items():
            self.log(
                f"loss_{loss_name}/test",
                loss_value.item() / len(batch["inputs"]),
                prog_bar=True,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )

    def _shared_eval_step(self, batch, batch_idx, split):
        output = self.model(batch)
        loss = self.criterion(output, batch)
        self.update_metrics(output, batch, split)
        if batch_idx == 0:
            self.draw_classification_samples(output, batch, batch_idx, split)
        return {"loss": loss}

    def update_metrics(self, output, batch, split):
        y_true = (
            torch.Tensor([self.categories.index(c) for c in batch["category"]])
            .long()
            .to(batch["inputs"].device)
        )
        y_pred = (
            torch.Tensor([self.categories.index(c) for c in output["category"]])
            .long()
            .to(batch["inputs"].device)
        )
        self.metrics[split](y_pred, y_true)
        self.confusion_matrix[f"cm_{split}"](y_pred, y_true)

    def on_train_epoch_end(self):
        self.draw_confusion_matrix("training")

        for k, m in self.metrics["training"].items():
            self.log(f"{k}/training", m.compute(), sync_dist=True)
            m.reset()

    def on_validation_epoch_end(self):
        self.draw_confusion_matrix("validation")

        for k, m in self.metrics["validation"].items():
            self.log(f"{k}/validation", m.compute(), sync_dist=True)
            m.reset()

    def on_test_epoch_end(self):
        self.draw_confusion_matrix("test")

        for k, m in self.metrics["test"].items():
            self.log(f"{k}/test", m.compute(), sync_dist=True)
            m.reset()

    def draw_confusion_matrix(self, split):
        m = self.confusion_matrix[f"cm_{split}"]
        m_arr = m.compute().detach().cpu().numpy()
        m.reset()

        out_dir = (
            Path(self.trainer.logger.log_dir)
            / "confusion_matrix"
            / f"epoch_{self.current_epoch:04d}"
            / split
        )
        out_dir.mkdir(exist_ok=True, parents=True)

        pd.DataFrame(data=m_arr, index=self.categories, columns=self.categories).to_csv(
            out_dir / "cm.csv", index=True
        )

        plot_confusion_matrix(
            confmat=m_arr,
            class_names=self.categories,
            save_path=out_dir / "cm.png",
            normalize=False,
            cmap="Blues",
            bg_color="white",
        )
        plot_confusion_matrix(
            confmat=m_arr,
            class_names=self.categories,
            save_path=out_dir / "cmn.png",
            normalize=True,
            cmap="Blues",
            bg_color="white",
        )

        self.logger.experiment.add_image(
            "confusion_matrix/" + split,
            torchvision.transforms.ToTensor()(Image.open(out_dir / "cm.png")),
            self.current_epoch,
        )
        self.logger.experiment.add_image(
            "confusion_matrix_normalized/" + split,
            torchvision.transforms.ToTensor()(Image.open(out_dir / "cmn.png")),
            self.current_epoch,
        )

    def draw_classification_samples(self, output, batch, batch_idx, split):
        save_path = (
            Path(self.trainer.logger.log_dir)
            / "classification_samples"
            / f"epoch_{self.current_epoch:04d}"
            / split
        )
        save_path.mkdir(exist_ok=True, parents=True)

        images = (
            batch["inputs"].permute(0, 2, 3, 1).detach().cpu().numpy() * 255
        ).astype("uint8")

        y_true = [self.categories.index(c) for c in batch["category"]]
        y_pred = [self.categories.index(c) for c in output["category"]]

        for i in range(len(y_true)):
            yt = y_true[i]
            yp = y_pred[i]

            stem = f"{self.categories[yp]}_conf_{output['logits'][i, yp].item():.2f}_batch_{batch_idx:06d}_{i:03d}"
            img_path = save_path / self.categories[yt] / f"{stem}.jpg"
            img_path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(images[i]).save(img_path)

            prb_path = save_path / self.categories[yt] / f"{stem}.json"
            prb_path.write_text(
                json.dumps(
                    {
                        class_str: "%.3f" % output["logits"][i, class_i].item()
                        for (class_i, class_str) in enumerate(self.categories)
                    },
                    indent=4,
                    sort_keys=False,
                )
            )
