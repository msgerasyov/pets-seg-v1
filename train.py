"""
Semantic segmentation example using PyTorch Lightning, Optuna, and MLflow.
Heavily inspired by:
https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
"""
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

import mlflow
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger, LightningLoggerBase

from utils import LightningDeepLabV3, OxfordIIITPetDataModule, get_transforms


def sample_configuration(trial: optuna.trial.Trial) -> Dict[str, Any]:
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.9, 0.99)

    return {"lr": lr, "momentum": momentum}


def train(
    model: LightningModule,
    datamodule: LightningDataModule,
    n_epochs: int,
    limit_train_batches: float = 1.0,
    logger: LightningLoggerBase | Iterable[LightningLoggerBase] | bool = True,
    callbacks: list[Callback] | Callback | None = None
) -> Dict[str, Any]:
    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=False,
        limit_train_batches=limit_train_batches,
        max_epochs=n_epochs,
        log_every_n_steps=10,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics


def update_experiment(
    configuration: Dict[str, Any],
    n_epochs: int,
    batch_size: int,
    limit_train_batches: float = 1.0,
    num_workers: int = 0,
    data_dir: Optional[str] = None,
    log_model: bool = False,
    seed: int = 0,
    experiment_name: str = "Default",
    run_name: Optional[str] = None,
    callbacks: list[Callback] | Callback | None = None
) -> Dict[str, Any]:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        exp_id = experiment.experiment_id
    else:
        exp_id = mlflow.create_experiment(experiment_name)
    if data_dir is None:
        data_dir = Path(__file__).parent.resolve()

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name
        )
        logger._run_id = run.info.run_id
        logger.log_hyperparams(configuration)
        model = LightningDeepLabV3(
            lr=configuration["lr"],
            momentum=configuration["momentum"]
        )
        train_transforms, val_transforms, test_transforms = get_transforms()
        datamodule = OxfordIIITPetDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            download=True,
            num_workers=num_workers,
            seed=seed
        )
        metrics = train(
            model,
            datamodule,
            n_epochs,
            limit_train_batches=limit_train_batches,
            logger=logger,
            callbacks=callbacks
        )
        if log_model:
            mlflow.pytorch.log_model(model, "model")

        return metrics


def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    configuration = sample_configuration(trial)
    callbacks = [
        PyTorchLightningPruningCallback(
            trial,
            monitor="val_iou"
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    if args.patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_iou",
                mode="max",
                patience=args.patience
            )
        )
    metrics = update_experiment(
        configuration,
        args.n_epochs,
        args.batch_size,
        limit_train_batches=args.limit_train_batches,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        seed=args.seed,
        experiment_name=args.experiment_name,
        run_name=str(trial.number),
        callbacks=callbacks
    )

    return metrics["val_iou"].item()


def seed_rngs(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train DeepLabV3 on Oxford-IIIT Pet Dataset')
    parser.add_argument('--batch-size', default=8, type=int, help="Batch size")
    parser.add_argument('--n-epochs', default=10, type=int,
                        help='Number of training epochs')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of workers for dataloaders')
    parser.add_argument('--data-dir', default=Path(__file__).parent.resolve(),
                        type=str, help='Directory to save dataset')
    parser.add_argument('--lr', default=7e-3, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.95, type=float,
                        help='SGD momentum')
    parser.add_argument('--patience', default=None, type=int,
                        help="Patience for early stopping")
    parser.add_argument("--optimize", action="store_true", default=False,
                        help="Run hyperparameters search")
    parser.add_argument('--n-trials', default=30, type=int,
                        help='Number of trials')
    parser.add_argument('--n-startup-trials', default=5, type=int,
                        help='Number of startup trials with disabled pruning')
    parser.add_argument('--n-warmup-steps', default=3, type=int,
                        help='Number of epochs with disabled pruning')
    parser.add_argument('--limit-train-batches', default=0.1, type=float,
                        help='Reduce the training set size while tuning')
    parser.add_argument('--seed', default=2023, type=int,
                        help='Seed for random number generators')
    parser.add_argument('--experiment-name', default="OxfordIIITPet", type=str,
                        help='Name of mlflow experiment')
    parser.add_argument('--run-name', default=None, type=str,
                        help='Name of mlflow run')
    args = parser.parse_args()
    seed_rngs(args.seed)
    if args.optimize:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials
        )
    else:
        callbacks = [LearningRateMonitor(logging_interval="epoch")]
        if args.patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val_iou",
                    mode="max",
                    patience=args.patience
                )
            )
        metrics = update_experiment(
            configuration={
                "lr": args.lr,
                "momentum": args.momentum
            },
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            limit_train_batches=1.0,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            log_model=True,
            seed=args.seed,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            callbacks=callbacks
        )
