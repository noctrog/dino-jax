from typing import Literal, Callable, Iterable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from itertools import islice
import os

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
import flax.nnx as nnx
import optax
from optax.schedules import warmup_cosine_decay_schedule
import orbax.checkpoint as ocp
from orbax.checkpoint.args import PyTreeRestore, PyTreeSave, JsonSave, JsonRestore, Composite
from grain.python import PyGrainCheckpointSave, PyGrainCheckpointRestore, DataLoader
import numpy as np
import tyro
from tqdm import tqdm
import wandb

from data import DataConfig, create_dataloaders, Prefetcher
from model import SSLConfig, SSLTeacherStudent, SSLDinoConfig, ViTConfig


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 1024

    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

    scaling_rule: Literal["sqrt_wrt_256", "sqrt_wrt_1024"] = "sqrt_wrt_256"
    lr_base: float = 0.0005
    lr_final: float = 1e-5
    lr_warmup_epochs: int = 10
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1

    student_temp: float = 0.1
    teacher_momentum_start: float = 0.996
    teacher_momentum_end: float = 1.0
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    teacher_temp_warmup_epochs: int = 30


@dataclass
class Config:
    seed: int = 42
    gpu_batch_size: int = 128
    """The batch size for a singe gpu at a given time (micro_batch // n_gpus)"""

    wandb: bool = False
    wandb_frequency: int = 20
    """Only log the kth iteration. NOTE: it does not average the intermediate results."""
    experiment_name: str | None = None
    checkpoint: bool = True
    num_checkpoints: int = 10
    """Indicates how many checkpoints are to be created during the entirety of the training duration."""
    restore: Path | None = None

    train: TrainConfig = field(default_factory=lambda: TrainConfig())
    data: DataConfig = field(default_factory=lambda: DataConfig())
    ssl: SSLConfig = field(default_factory=lambda: SSLConfig())


def config_from_dict(d: dict) -> tuple[TrainConfig, DataConfig, SSLConfig]:
    data = DataConfig(**d["data"])
    dino = SSLDinoConfig(**d["ssl"]["dino"])
    train = TrainConfig(**d["train"])
    vit = ViTConfig(**d["ssl"]["vit"])
    ssl = SSLConfig(dino=dino, vit=vit)

    return train, data, ssl


def cosine_scheduler(start: float, end: float, num_iter: int) -> Callable:
    def interpolate(i: int) -> float:
        if i < 0:
            return start
        elif i < num_iter:
            progress = i / num_iter
            return end + (start - end) * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            return end

    return interpolate


def cosine_scheduler_jax(start: float, end: float, num_iter: int) -> Callable:
    def interpolate_jax(i: int) -> jax.Array:
        progress = i / num_iter

        cosine_value = end + (start - end) * 0.5 * (1 + jnp.cos(jnp.pi * progress))
        value = jnp.where(i < 0, start, jnp.where(i < num_iter, cosine_value, end))
        return value

    return interpolate_jax


def build_schedules(
    cfg: TrainConfig, total_steps: int
) -> tuple[Callable, Callable, Callable, Callable]:
    steps_per_epoch = total_steps // cfg.epochs
    lr_base_scaled = cfg.lr_base * (cfg.batch_size / 256.0)
    lr_schedule = warmup_cosine_decay_schedule(
        init_value=cfg.lr_final,
        peak_value=lr_base_scaled,
        warmup_steps=cfg.lr_warmup_epochs * steps_per_epoch,
        decay_steps=total_steps,
        end_value=cfg.lr_final,
    )
    wd_schedule = cosine_scheduler_jax(cfg.weight_decay_start, cfg.weight_decay_end, total_steps)
    mo_schedule = cosine_scheduler(
        cfg.teacher_momentum_start, cfg.teacher_momentum_end, total_steps
    )
    tt_schedule = cosine_scheduler(
        cfg.teacher_temp_start,
        cfg.teacher_temp_end,
        cfg.teacher_temp_warmup_epochs * steps_per_epoch,
    )

    return lr_schedule, wd_schedule, mo_schedule, tt_schedule


def create_optimizer(
    model: SSLTeacherStudent, cfg: TrainConfig, train_iters: int, grad_acc_steps: int
) -> nnx.Optimizer:
    lr_schedule, wd_schedule, _, _ = build_schedules(cfg, train_iters)

    def mask_fn(path, param):
        if param in ["pos_embed", "cls_token"]:
            return False
        elif param.value.ndim != 2:
            return False
        return True

    wd_mask = nnx.map_state(mask_fn, nnx.state((model.student, model.dino_student_head), nnx.Param))
    chain = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_schedule,
            b1=cfg.adamw_beta1,
            b2=cfg.adamw_beta2,
            weight_decay=wd_schedule,
            mask=wd_mask,
        ),
    )

    if grad_acc_steps > 1:
        chain = optax.MultiSteps(chain, every_k_schedule=grad_acc_steps)
    return nnx.Optimizer((model.student, model.dino_student_head), chain, wrt=nnx.Param)


def restore_checkpoint(
    ckpt_path: Path,
    cfg: Config,
    data_iter: Prefetcher,
    train_iters: int,
    grad_acc_steps: int,
    mesh: jax.sharding.Mesh | None,
) -> tuple[SSLTeacherStudent, nnx.Optimizer, Iterable, int]:
    """Loads and returns a model and optimizer checkpoint. It also
    modifies the `train`, `data` and `ssl` fields of the config inplace.
    """
    mngr = ocp.CheckpointManager(
        os.path.abspath(ckpt_path),
        item_names=("state", "optim", "loader", "config"),
        options=ocp.CheckpointManagerOptions(read_only=True),
    )

    step = mngr.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint found in {ckpt_path}")
    print(f"Found checkpoint at step {step}. Restoring...")

    # Restore model
    cfg_ckpt = mngr.restore(step, args=Composite(config=JsonRestore()))["config"]
    cfg_ckpt["ssl"]["vit"] = {
        k: v
        for k, v in cfg_ckpt["ssl"]["vit"].items()
        if k not in ["embed_dim", "num_layers", "mlp_hidden_dim", "num_heads"]
    }
    ssl_ckpt = SSLConfig(
        dino=SSLDinoConfig(**cfg_ckpt["ssl"]["dino"]),
        vit=ViTConfig(**cfg_ckpt["ssl"]["vit"]),
    )
    train_ckpt = TrainConfig(**cfg_ckpt["train"])
    data_ckpt = DataConfig(**cfg_ckpt["data"])
    cfg.train, cfg.data, cfg.ssl = train_ckpt, data_ckpt, ssl_ckpt

    ssl = nnx.eval_shape(lambda: SSLTeacherStudent(ssl_ckpt, mesh=mesh, rngs=nnx.Rngs(0)))
    graphdef, state = nnx.split(ssl)
    state = mngr.restore(step, args=Composite(state=PyTreeRestore(state)))["state"]
    ssl = nnx.merge(graphdef, state)

    # Restore optimizer
    optim = create_optimizer(ssl, cfg.train, train_iters, grad_acc_steps)
    opt_state = mngr.restore(step, args=Composite(optim=PyTreeRestore(nnx.state(optim))))
    nnx.update(optim, opt_state["optim"])

    # Restore dataloader state
    loader = data_iter.get_underlying_iterator()
    loader = mngr.restore(step, args=Composite(loader=PyGrainCheckpointRestore(loader)))["loader"]
    data_iter = Prefetcher(loader, mesh)

    # Set to none because the restore path cannot be checkpointed
    cfg.restore = None
    return ssl, optim, data_iter, step


def main(cfg: Config):
    if cfg.wandb:
        wandb.init(project="dino-jax", name=cfg.experiment_name)

    rngs = nnx.Rngs(params=cfg.seed, dropout=cfg.seed + 1)

    mesh = jax.make_mesh((jax.device_count(),), ("data",))

    assert cfg.train.batch_size % (cfg.gpu_batch_size * jax.device_count()) == 0
    grad_acc_steps = int(cfg.train.batch_size / (cfg.gpu_batch_size * jax.device_count()))
    micro_bs = cfg.gpu_batch_size * jax.device_count()
    print("grad_acc_steps: ", grad_acc_steps)
    print("micro_batch_size: ", micro_bs)

    train_loader, total_train_iters = create_dataloaders(cfg.data, micro_bs, cfg.train.epochs)
    data_iter = Prefetcher(iter(train_loader), mesh)

    if cfg.restore is not None:
        model, optim, data_iter, step = restore_checkpoint(
            cfg.restore, cfg, data_iter, total_train_iters, grad_acc_steps, mesh=mesh
        )
    else:
        model = SSLTeacherStudent(cfg.ssl, mesh=mesh, rngs=rngs)
        optim = create_optimizer(model, cfg.train, total_train_iters, grad_acc_steps)
        step = 0

    model.train()
    student_params = jax.tree.leaves(nnx.state(model.student, nnx.Param))
    param_count = sum(jax.tree.map(lambda x: jnp.size(x), student_params))
    head_params = jax.tree.leaves(nnx.state(model.dino_student_head, nnx.Param))
    head_param_count = sum(jax.tree.map(lambda x: jnp.size(x), head_params))
    print(f"ViT backbone: {param_count / 1_000_000:.2f}M params")
    print(f"ViT head: {head_param_count / 1_000_000:.2f}M params")

    lr_schedule, wd_schedule, mo_schedule, tt_schedule = build_schedules(
        cfg.train, total_train_iters
    )

    ckpt_interval = total_train_iters // (cfg.num_checkpoints - 1)
    ckpt_steps = [i * ckpt_interval for i in range(1, cfg.num_checkpoints)]
    ckpt_steps.append(total_train_iters)
    opts = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        create=True,
        read_only=not cfg.checkpoint,
    )
    ckpt_path = os.path.abspath(wandb.run.dir if cfg.wandb else Path("outputs"))
    mngr = ocp.CheckpointManager(
        ckpt_path, options=opts, item_names=("state", "optim", "loader", "config")
    )

    iters_per_epoch = total_train_iters // cfg.train.epochs  # This is an approximation
    update_last_layer = lambda step: (step // iters_per_epoch >= cfg.train.freeze_last_layer_epochs)
    for samples in tqdm(
        data_iter,
        desc="Step",
        initial=step,
        total=total_train_iters,
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
    ):
        loss = model(
            optim,
            samples["global_crops"],
            samples["local_crops"],
            student_temp=cfg.train.student_temp,
            teacher_temp=tt_schedule(step),
            update_last_layer=update_last_layer(step),
        )

        if (step + 1) % grad_acc_steps == 0:
            model.update_teacher(mo_schedule(step))

            if cfg.wandb and step % (grad_acc_steps * cfg.wandb_frequency) == 0:
                wandb.log(
                    {
                        "iter": step,
                        "loss": loss,
                        "lr": lr_schedule(step),
                        "wd": wd_schedule(step),
                        "teacher_momentum": mo_schedule(step),
                        "teacher_temp": tt_schedule(step),
                    }
                )

        step += 1
        if step in ckpt_steps:
            mngr.save(
                step,
                args=Composite(
                    state=PyTreeSave(nnx.state(model)),
                    optim=PyTreeSave(nnx.state(optim)),
                    loader=PyGrainCheckpointSave(data_iter.get_underlying_iterator()),
                    config=JsonSave(asdict(cfg)),
                ),
            )
            mngr.wait_until_finished()

        if step == total_train_iters:
            break

    mngr.close()
    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    cfg: Config = tyro.cli(Config)
    main(cfg)
