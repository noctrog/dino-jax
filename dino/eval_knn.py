from dataclasses import dataclass, field
from pathlib import Path
import os

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import flax.nnx as nnx
import orbax.checkpoint as ocp
import tyro
from tqdm import tqdm
from grain.python import DataLoader

from train import TrainConfig
from data import DataConfig, create_dataloaders
from model import SSLConfig, SSLDinoConfig, SSLTeacherStudent, ViT, ViTConfig

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


@dataclass
class Config:
    ckpt: Path

    nb_knn: list[int] = field(default_factory=lambda: [10, 20, 100])
    """Number of nearest neightbours"""
    temperature: float = 0.07
    """Temperature to use in the voting coefficient"""
    gpu_batch_size: int = 256
    """Batch size per gpu"""
    n_class_per_list: int = -1
    """Numberto take per class"""
    n_tries: int = 1
    """Number of tries"""

    seed: int = 0
    data: DataConfig = field(default_factory=lambda: DataConfig())


def extract_features_and_labels(
    vit: ViT,
    data_loader: DataLoader,
    num_samples: int,
    embed_dim: int,
    mesh: Mesh,
    *,
    no_features_sharding: bool = False,
) -> tuple[jax.Array, jax.Array]:
    global_batch_size = cfg.gpu_batch_size * jax.device_count()
    vit.eval()

    @nnx.jit
    def process_batch(
        vit: ViT, images: jax.Array, labels: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        features = vit(images)["cls"]
        return features, labels

    all_feature_list = []
    all_labels_list = []

    image_sharding = NamedSharding(mesh, P("data", None, None, None))
    label_sharding = NamedSharding(mesh, P("data"))

    for sample in tqdm(
        data_loader,
        desc="Extracting features",
        total=num_samples // global_batch_size,
        leave=False,
    ):
        # Make sure that the batch is divisible by the number of GPUs
        # For a proper evaluation, the last batch should be patched
        current_bs = sample["image"].shape[0]
        if current_bs % jax.device_count() != 0:
            new_bs = (current_bs // jax.device_count()) * jax.device_count()
            sample["image"] = sample["image"][:new_bs]
            sample["label"] = sample["label"][:new_bs]

        images = jax.device_put(sample["image"], image_sharding)
        labels = jax.device_put(sample["label"], label_sharding)

        feats_batch, labels_batch = process_batch(vit, images, labels)

        all_feature_list.append(jnp.squeeze(feats_batch, axis=1))
        all_labels_list.append(labels_batch)

    all_features = jnp.concatenate(all_feature_list, axis=0)
    all_labels = jnp.concatenate(all_labels_list, axis=0)

    all_features = all_features / jnp.linalg.norm(all_features, axis=1, keepdims=True)

    if no_features_sharding:
        all_features = jax.device_put(all_features, NamedSharding(mesh, P(None, None)))
    all_labels = jax.device_put(all_labels, NamedSharding(mesh, P(None)))

    return all_features, all_labels


def knn_classifier(
    train_features: jax.Array,
    train_labels: jax.Array,
    val_features: jax.Array,
    val_labels: jax.Array,
    k: int,
    temperature: float,
    num_classes: int = 1000,
    val_chunk_size: int = 1024,
) -> tuple[float, float]:
    num_val_chunks = (val_labels.shape[0] + val_chunk_size - 1) // val_chunk_size

    # @jax.jit
    def inner_loop_step(features: jax.Array, targets: jax.Array):
        similarity = jnp.matmul(features, train_features.T)
        dist, ids = jax.lax.top_k(similarity, k=k)
        dist = jnp.exp(dist / temperature)

        probs = jax.vmap(
            lambda labels, weights: jnp.bincount(labels, weights=weights, length=num_classes),
            in_axes=(0, 0),
        )(train_labels[ids], dist)
        preds = jnp.argsort(probs, axis=1, descending=True)
        correct = preds == targets[:, None]

        top_1 = correct[:, 0].sum()
        top_5 = correct[:, : min(5, k)].sum()
        return top_1, top_5

    top_1, top_5, total = 0.0, 0.0, 0
    for i in range(num_val_chunks):
        features = val_features[i * val_chunk_size : (i + 1) * val_chunk_size]
        targets = val_labels[i * val_chunk_size : (i + 1) * val_chunk_size]

        top_1_chunk, top_5_chunk = inner_loop_step(features, targets)

        top_1 += top_1_chunk.item()
        top_5 += top_5_chunk.item()
        total += targets.size

    top_1 = top_1 * 100.0 / total
    top_5 = top_5 * 100.0 / total

    return top_1, top_5


def load_model(cfg: Config, mesh: jax.sharding.Mesh | None) -> ViT:
    mngr = ocp.CheckpointManager(
        os.path.abspath(cfg.ckpt),
        item_names=("state", "optim", "config"),
        options=ocp.CheckpointManagerOptions(read_only=True),
    )

    step = mngr.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint found in {cfg.ckpt}")
    print(f"Found checkpoint at step {step}")

    ssl_cfg = mngr.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))["config"][
        "ssl"
    ]
    ssl_cfg = SSLConfig(dino=SSLDinoConfig(**ssl_cfg["dino"]), vit=ViTConfig(**ssl_cfg["vit"]))

    ssl = nnx.eval_shape(lambda: SSLTeacherStudent(ssl_cfg, mesh=mesh, rngs=nnx.Rngs(0)))
    graphdef, state = nnx.split(ssl)
    state = mngr.restore(step, args=ocp.args.Composite(state=ocp.args.PyTreeRestore(state)))[
        "state"
    ]
    ssl = nnx.merge(graphdef, state)
    vit = ssl.teacher
    del ssl
    return vit


def main(cfg: Config):
    num_devices = jax.device_count()
    batch_size = num_devices * cfg.gpu_batch_size

    mesh = jax.make_mesh((jax.device_count(),), ("data",))

    vit = load_model(cfg, mesh)

    train_loader, val_loader, nb_train_iters, nb_val_iters = create_dataloaders(
        cfg.data, batch_size, epochs=1, for_validation=True
    )

    train_features, train_labels = extract_features_and_labels(
        vit,
        data_loader=train_loader,
        num_samples=nb_train_iters * batch_size,
        embed_dim=vit.embed_dim,
        mesh=mesh,
    )

    val_features, val_labels = extract_features_and_labels(
        vit,
        data_loader=val_loader,
        num_samples=nb_val_iters * batch_size,
        embed_dim=vit.embed_dim,
        mesh=mesh,
        no_features_sharding=True,
    )
    del vit

    for k in cfg.nb_knn:
        top_1, top_5 = knn_classifier(
            train_features,
            train_labels,
            val_features,
            val_labels,
            k=k,
            temperature=cfg.temperature,
        )

        print(f"K: {k}\ttop_1: {top_1}\ttop_5: {top_5}")


if __name__ == "__main__":
    cfg: Config = tyro.cli(Config)
    main(cfg)
