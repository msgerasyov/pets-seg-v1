from typing import Any, Dict, Callable
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional import dice, jaccard_index
from torchvision import datasets
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


class LightningDeepLabV3(pl.LightningModule):
    def __init__(self, lr: float, momentum: float = 0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = deeplabv3_resnet50(num_classes=1, weights_backbone=weights)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)["out"].view(target.shape)
        loss = self.criterion(output, target)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx: int) -> Dict[str, Any]:
        iou, dice_score = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_iou": iou, "val_dice": dice_score}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx: int) -> Dict[str, Any]:
        iou, dice_score = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_iou": iou, "test_dice": dice_score}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)["out"].view(target.shape)
        dice_score = dice(output, target.to(torch.long))
        iou = jaccard_index(output, target.to(torch.long), task="binary")
        return iou, dice_score

    def configure_optimizers(self):
        opt = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=2)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_iou"
            }
        }


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
                 download: bool = False, num_workers: int = 0, seed: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms
        self.download = download
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: Optional[str] = None) -> None:
        self.pets_test = OxfordIIITPetSegmentationDataset(self.data_dir, split="test", transforms=self._test_transforms,
                                                          download=self.download)
        pets_train = OxfordIIITPetSegmentationDataset(self.data_dir, split="trainval",
                                                      transforms=self._train_transforms, download=self.download)
        pets_val = OxfordIIITPetSegmentationDataset(self.data_dir, split="trainval",
                                                    transforms=self._val_transforms, download=self.download)
        train_idx, val_idx = train_test_split(np.arange(len(pets_train)), test_size=0.15, random_state=self.seed)
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
