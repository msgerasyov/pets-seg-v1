"""
Semantic segmentation example using PyTorch Lightning, Optuna, and MLflow.
Heavily inspired by:
https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
"""
import os
import pathlib
import random
from typing import Dict, Any

import mlflow
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import yaml
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from utils import LightningDeepLabV3, OxfordIIITPetDataModule, get_transforms


def sample_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.9, 0.99)

    return {
        "lr": lr,
        "momentum": momentum
    }


def objective(trial: optuna.trial.BaseTrial, config: Dict[str, Any]) -> float:
    experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
    exp_id = experiment.experiment_id if experiment else mlflow.create_experiment(config["mlflow"]["experiment_name"])
    run_name = config["mlflow"].get("run_name", str(trial.number))

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        # Use same code objective to reproduce the best model
        params = sample_params(trial)
        model = LightningDeepLabV3(lr=params["lr"], momentum=params["momentum"])
        data_dir = config.get("data_dir", pathlib.Path(__file__).parent.resolve())
        train_transforms, val_transforms, test_transforms = get_transforms()
        datamodule = OxfordIIITPetDataModule(data_dir=data_dir, batch_size=config["batch_size"],
                                             train_transforms=train_transforms, val_transforms=val_transforms,
                                             test_transforms=test_transforms, download=True,
                                             num_workers=config["n_workers"])
        logger = MLFlowLogger(experiment_name=config["mlflow"]["experiment_name"], run_name=run_name)
        logger._run_id = run.info.run_id
        callbacks = [PyTorchLightningPruningCallback(trial, monitor="val_BinaryJaccardIndex"),
                     LearningRateMonitor(logging_interval="epoch")]
        if "patience" in config:
            callbacks.append(EarlyStopping(monitor="val_BinaryJaccardIndex", mode="max", patience=config["patience"]))
        trainer = pl.Trainer(
            logger=logger,
            enable_checkpointing=False,
            limit_train_batches=config.get("limit_train_batches", 1.0),
            max_epochs=config["epochs"],
            log_every_n_steps=10,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=callbacks
        )
        trainer.logger.log_hyperparams(params)
        trainer.fit(model, datamodule=datamodule)
        if config["log_model"]:
            mlflow.pytorch.log_model(model, "model")

        return trainer.callback_metrics["val_BinaryJaccardIndex"].item()


def seed_rngs(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    with open(os.path.join(pathlib.Path(__file__).parent.resolve(), "config.yml"), "r") as stream:
        config = yaml.safe_load(stream)
    seed_rngs(config["seed"])
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    pruner = optuna.pruners.MedianPruner(**config["optuna"]["pruner"])
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: objective(trial, config["tune"]), n_trials=config["optuna"]["n_trials"])
    objective(study.best_trial, config["train"])
