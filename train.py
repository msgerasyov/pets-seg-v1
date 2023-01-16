"""
Semantic segmentation example using PyTorch Lightning, Optuna, and MLflow.
Heavily inspired by:
https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
"""
import argparse
import os
import random
from typing import Any, Dict, Callable
from typing import Optional

import albumentations as A
import cv2
import mlflow
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional import dice, jaccard_index
from torchvision import datasets
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


class LightningDeepLabV3(pl.LightningModule):
    def __init__(self, opt_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = deeplabv3_resnet50(num_classes=1, weights_backbone=weights)
        self.opt_config = opt_config
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)["out"].squeeze()
        loss = self.criterion(output, target)
        dice_score = dice(output, target.to(torch.long))
        iou = jaccard_index(output, target.to(torch.long), task="binary")
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_dice", dice_score, on_step=False, on_epoch=True)
        self.log("train_iou", iou, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self(data)["out"].squeeze()
        dice_score = dice(output, target.to(torch.long))
        iou = jaccard_index(output, target.to(torch.long), task="binary")
        self.log("val_dice", dice_score, on_step=False, on_epoch=True)
        self.log("val_iou", iou, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        if self.opt_config["opt"] == "Adam":
            return optim.Adam(self.model.parameters(), **self.opt_config["params"])

        return optim.SGD(self.model.parameters(), **self.opt_config["params"])


def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask


class OxfordIIITPetSegmentationDataset(datasets.OxfordIIITPet):
    def __init__(self, root: str, split: str = "trainval", transforms: Optional[Callable] = None,
                 download: bool = False):
        super().__init__(root=root, split=split, target_types="segmentation", download=download)
        self._transforms = transforms

    def __getitem__(self, index: int):
        image, mask = super().__getitem__(index)
        image, mask = np.asarray(image), preprocess_mask(np.asarray(mask))
        if self._transforms is not None:
            transformed = self._transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


class OxfordIIITPetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, train_transforms: Optional[Callable] = None,
                 val_transforms: Optional[Callable] = None, test_transforms: Optional[Callable] = None,
                 download: bool = False, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms
        self.download = download
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.pets_test = OxfordIIITPetSegmentationDataset(self.data_dir, split="test", transforms=self._test_transforms,
                                                          download=self.download)
        pets_train = OxfordIIITPetSegmentationDataset(self.data_dir, split="trainval",
                                                      transforms=self._train_transforms, download=self.download)
        pets_val = OxfordIIITPetSegmentationDataset(self.data_dir, split="trainval",
                                                    transforms=self._val_transforms, download=self.download)
        train_idx, val_idx = train_test_split(np.arange(len(pets_train)), test_size=0.15, random_state=42)
        self.pets_train, self.pets_val = Subset(pets_train, train_idx), Subset(pets_val, val_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.pets_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.pets_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.pets_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )


def sample_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    opt = trial.suggest_categorical("opt", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    if opt == "Adam":
        beta1 = trial.suggest_float("beta1", 0.9, 0.99)
        beta2 = trial.suggest_float("beta2", 0.99, 1.0)
        return {
            "opt": opt,
            "params": {
                "lr": lr,
                "betas": (beta1, beta2)
            }
        }
    momentum = trial.suggest_float("momentum", 0.9, 0.99)
    return {
        "opt": opt,
        "params": {
            "lr": lr,
            "momentum": momentum
        }
    }


def get_transforms():
    train_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=256, min_width=256),
            A.RandomCrop(256, 256),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=256, min_width=256),
            A.CenterCrop(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform, test_transform


def objective(trial: optuna.trial.BaseTrial, config: Dict[str, Any]) -> float:
    experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
    exp_id = experiment.experiment_id if experiment else mlflow.create_experiment(config["mlflow"]["experiment_name"])
    run_name = config["mlflow"].get("run_name", str(trial.number))

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        # Use same code objective to reproduce the best model
        run_id = run.info.run_id
        params = sample_params(trial)
        model = LightningDeepLabV3(params)
        data_dir = config.get("data_dir", os.getcwd())
        train_transforms, val_transforms, test_transforms = get_transforms()
        datamodule = OxfordIIITPetDataModule(data_dir=data_dir, batch_size=config["batch_size"],
                                             train_transforms=train_transforms, val_transforms=val_transforms,
                                             test_transforms=test_transforms, download=True,
                                             num_workers=config["n_workers"])
        logger = MLFlowLogger(experiment_name=config["mlflow"]["experiment_name"], run_name=run_name)
        logger._run_id = run_id
        trainer = pl.Trainer(
            logger=logger,
            enable_checkpointing=False,
            limit_train_batches=config.get("limit_train_batches", 1.0),
            max_epochs=config["epochs"],
            log_every_n_steps=10,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_iou"),
                       EarlyStopping(monitor="val_iou", mode="max", patience=config["patience"])],
        )
        trainer.logger.log_hyperparams(params)
        trainer.fit(model, datamodule=datamodule)
        if config["log_model"]:
            mlflow.pytorch.log_model(model, "model")

        return trainer.callback_metrics["val_iou"].item()


def seed_rngs(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    with open("config.yml", "r") as stream:
        config = yaml.safe_load(stream)
    seed_rngs(config["seed"])
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    pruner = optuna.pruners.MedianPruner(**config["optuna"]["pruner"])
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: objective(trial, config["tune"]), n_trials=config["optuna"]["n_trials"])
    objective(study.best_trial, config["train"])
