from typing import Literal, Dict, Optional, List
from types import SimpleNamespace
import platform
from dataclasses import dataclass
import warnings

from datasets import load_dataset
import jax
import jax.numpy as jnp
import dm_pix
import numpy as np
import dm_pix as pix
import grain.python as grain
import tensorflow_datasets as tfds
import albumentations as A
import cv2


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    dataset_name: Literal["imagenet-1k", "timm/imagenet-1k-wds"] = (
        "timm/imagenet-1k-wds"
    )

    global_crops_scale: tuple[float, float] = (0.4, 1.0)
    global_crops_size: int = 224
    local_crops_scale: tuple[float, float] = (0.05, 0.4)
    local_crops_size: int = 96
    local_crops_number: int = 8

    normalization_mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    normalization_std: tuple[float, float, float] = IMAGENET_DEFAULT_STD


class RandomResizedCrop(grain.MapTransform):
    def __init__(
        self,
        size: tuple[int, int] = (224, 224),
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: str = "bicubic",
    ):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def get_crop_size(self, image: np.ndarray | jax.Array) -> tuple[int, int, int]:
        height, width, _ = image.shape
        area = height * width
        log_ratio = np.log(self.ratio)

        for _ in range(10):
            target_area = area * self.np_rng.uniform(*self.scale)
            aspect_ratio = np.exp(self.np_rng.uniform(*log_ratio))
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                return h, w, 3

        # Fallback to center crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:
            w = width
            h = height
        return h, w, 3

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        image = element["image"]

        image = A.RandomResizedCrop(
            self.size, self.scale, self.ratio, interpolation=cv2.INTER_CUBIC
        )
        # image = jnp.asarray(image, dtype=jnp.float32)
        # crops = self.get_crop_size(image)
        # image = pix.random_crop(crop_key, image, crops)
        # image = jax.image.resize(image, self.size, method=self.interpolation)

        element["image"] = image
        return element


def create_dataloaders(
    key, batch_size, epochs
) -> tuple[grain.DataLoader, grain.DataLoader, int, int]:
    imagenet = tfds.data_source("imagenet2012", split="train")
    imagenet_val = tfds.data_source("imagenet2012", split="validation")
    train_loader = grain.DataLoader(
        data_source=imagenet,
        operations=[
            RandomResizedCrop(),
            grain.Batch(batch_size, drop_remainder=True),
        ],
        sampler=grain.IndexSampler(
            num_records=len(imagenet),
            num_epochs=epochs,
            shard_options=grain.NoSharding(),
            shuffle=True,
            seed=0,
        ),
        worker_count=32,
        read_options=grain.ReadOptions(num_threads=8, prefetch_buffer_size=500),
    )
    val_loader = grain.DataLoader(
        data_source=imagenet_val,
        operations=[grain.Batch(batch_size, drop_remainder=False)],
        sampler=grain.IndexSampler(
            num_records=len(imagenet_val),
            num_epochs=1,
            shard_options=grain.NoSharding(),
            shuffle=False,
            seed=0,
        ),
        worker_count=0,
    )
    return (
        train_loader,
        val_loader,
        len(imagenet) // batch_size,
        (len(imagenet_val) + batch_size - 1) // batch_size,
    )


if __name__ == "__main__":
    cfg = SimpleNamespace()
    cfg.epochs = 10

    key = jax.random.PRNGKey(0)
    train_loader, val_loader = create_dataloaders(key, 32, cfg)
    image = next(iter(train_loader))
