from typing import Literal
from dataclasses import dataclass
import platform
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_NO_TELEMETRY"] = "1"

import numpy as np
import grain.python as grain
import tensorflow_datasets as tfds
import albumentations as A
import cv2


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    num_workers: int = 32

    global_crops_scale: tuple[float, float] = (0.4, 1.0)
    global_crops_size: tuple[int, int] = (224, 224)
    local_crops_scale: tuple[float, float] = (0.05, 0.4)
    local_crops_size: tuple[int, int] = (96, 96)
    local_crops_number: int = 8
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)

    normalization_mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    normalization_std: tuple[float, float, float] = IMAGENET_DEFAULT_STD


class DINOAugmentations(grain.MapTransform):
    def __init__(
        self,
        cfg: DataConfig,
    ):
        super().__init__()
        self.local_crops_number = cfg.local_crops_number

        self.geom_aug_global = A.Compose(
            [
                A.RandomResizedCrop(
                    cfg.global_crops_size,
                    cfg.global_crops_scale,
                    cfg.ratio,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.HorizontalFlip(p=0.5),
            ]
        )
        self.geom_aug_local = A.Compose(
            [
                A.RandomResizedCrop(
                    cfg.local_crops_size,
                    cfg.local_crops_scale,
                    cfg.ratio,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.HorizontalFlip(p=0.5),
            ]
        )
        color_jittering = A.Compose(
            [
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                A.ToGray(p=0.2),
            ]
        )
        global_transfo1_extra = A.GaussianBlur(
            blur_limit=23, sigma_limit=(0.1, 2.0), p=1.0
        )
        global_transfo2_extra = A.Compose(
            [
                A.GaussianBlur(blur_limit=23, sigma_limit=(0.1, 2.0), p=0.1),
                A.Solarize(threshold_range=(0.5, 0.5), p=0.2),
            ]
        )

        local_transfo_extra = A.GaussianBlur(
            blur_limit=23, sigma_limit=(0.1, 2.0), p=0.5
        )
        self.normalize = A.Normalize(
            mean=cfg.normalization_mean, std=cfg.normalization_std
        )
        self.global_transfo1 = A.Compose(
            [color_jittering, global_transfo1_extra, self.normalize]
        )
        self.global_transfo2 = A.Compose(
            [color_jittering, global_transfo2_extra, self.normalize]
        )
        self.local_transfo = A.Compose(
            [color_jittering, local_transfo_extra, self.normalize]
        )

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        image = element["image"]

        im1_base = self.geom_aug_global(image=image)["image"]
        global_crop_1 = self.global_transfo1(image=im1_base)["image"]

        im2_base = self.geom_aug_global(image=image)["image"]
        global_crop_2 = self.global_transfo2(image=im2_base)["image"]

        local_crops = [
            self.local_transfo(image=self.geom_aug_local(image=image)["image"])["image"]
            for _ in range(self.local_crops_number)
        ]

        return {
            "global_crops": [global_crop_1, global_crop_2],
            "local_crops": local_crops,
        }


class DINOValidationAugmentations(grain.MapTransform):
    def __init__(self, cfg: DataConfig):
        self.transforms = A.Compose(
            [
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(cfg.normalization_mean, cfg.normalization_std),
            ]
        )

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        element["image"] = self.transforms(image=element["image"])["image"]
        return element


def create_dataloaders(
    cfg: DataConfig,
    batch_size: int,
    epochs: int,
    train_augmentations: bool = True,
) -> tuple[grain.DataLoader, grain.DataLoader, int, int]:
    scratch_path = "/scratch/tensorflow_datasets"
    using_scratch = "gpu" in platform.node() and os.path.exists(scratch_path)
    imagenet = tfds.data_source(
        "imagenet2012", split="train", data_dir=scratch_path if using_scratch else None
    )
    imagenet_val = tfds.data_source(
        "imagenet2012",
        split="validation",
        data_dir=scratch_path if using_scratch else None,
    )

    train_loader = grain.DataLoader(
        data_source=imagenet,
        operations=[
            DINOAugmentations(cfg)
            if train_augmentations
            else DINOValidationAugmentations(cfg),
            grain.Batch(batch_size, drop_remainder=True),
        ],
        sampler=grain.IndexSampler(
            num_records=len(imagenet),
            num_epochs=epochs,
            shard_options=grain.NoSharding(),
            shuffle=True,
            seed=0,
        ),
        worker_count=cfg.num_workers,
        read_options=grain.ReadOptions(num_threads=8, prefetch_buffer_size=32),
    )
    val_loader = grain.DataLoader(
        data_source=imagenet_val,
        operations=[
            DINOValidationAugmentations(cfg),
            grain.Batch(batch_size, drop_remainder=False),
        ],
        sampler=grain.IndexSampler(
            num_records=len(imagenet_val),
            num_epochs=1,
            shard_options=grain.NoSharding(),
            shuffle=False,
            seed=0,
        ),
        worker_count=cfg.num_workers,
    )
    return (
        train_loader,
        val_loader,
        len(imagenet) // batch_size,
        (len(imagenet_val) + batch_size - 1) // batch_size,
    )
