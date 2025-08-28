from typing import Literal, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from itertools import islice

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
import flax.nnx as nnx
import optax
from optax.schedules import warmup_cosine_decay_schedule
import orbax.checkpoint as ocp
import numpy as np
import tyro
from tqdm import tqdm
import wandb

from data import DataConfig, create_dataloaders
from model import SSLConfig, SSLTeacherStudent, SSLDinoConfig, ViTConfig


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 1024
    gpu_batch_size: int = 128
    """The batch size for a singe gpu at a given time (micro_batch // n_gpus)"""

    wandb: bool = False
    experiment_name: str | None = None
    checkpoint_every: int = 0
    restore: bool = False

    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

    scaling_rule: Literal["sqrt_wrt_256", "sqrt_wrt_1024"] = "sqrt_wrt_256"
    lr_base: float = 0.0005
    lr_final: float = 1e-6
    lr_warmup_epochs: int = 10
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1
    layerwise_decay: float = 0.9
    patch_embed_lr_mult: float = 0.2

    student_temp: float = 0.1
    teacher_momentum_start: float = 0.996
    teacher_momentum_end: float = 1.0
    # teacher_warmup_temp: float = 0.04  # NOTE: unused for now
    teacher_temp: float = 0.04
    # teacher_temp_warmup_epochs: int = 30

    data: DataConfig = field(default_factory=lambda: DataConfig())
    ssl: SSLConfig = field(default_factory=lambda: SSLConfig())


def config_from_dict(d: dict) -> TrainConfig:
    data = DataConfig(**d["data"])
    dino = SSLDinoConfig(**d["ssl"]["dino"])
    vit = ViTConfig(**d["ssl"]["vit"])
    ssl = SSLConfig(dino=dino, vit=vit)

    # Handle the top-level fields, excluding the nested ones
    top_level = {k: v for k, v in d.items() if k not in ["data", "ssl"]}
    return TrainConfig(**top_level, data=data, ssl=ssl)


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


def build_schedules(
    cfg: TrainConfig, samples_per_epoch: int
) -> tuple[Callable, Callable, Callable]:
    total_steps = cfg.epochs * samples_per_epoch
    lr_base_scaled = cfg.lr_base * (cfg.batch_size / 256.0)
    lr_schedule = warmup_cosine_decay_schedule(
        init_value=cfg.lr_final,
        peak_value=lr_base_scaled,
        warmup_steps=cfg.lr_warmup_epochs * samples_per_epoch,
        decay_steps=total_steps,
        end_value=cfg.lr_final,
    )
    wd_schedule = cosine_scheduler(
        cfg.weight_decay_start, cfg.weight_decay_end, total_steps
    )
    mo_schedule = cosine_scheduler(
        cfg.teacher_momentum_start, cfg.teacher_momentum_end, total_steps
    )

    return lr_schedule, wd_schedule, mo_schedule


def main(cfg: TrainConfig):
    if cfg.wandb:
        wandb.init(project="dino-jax", name=cfg.experiment_name)

    rngs = nnx.Rngs(cfg.seed)

    mesh = jax.make_mesh((jax.device_count(),), ("data",))

    assert cfg.batch_size % (cfg.gpu_batch_size * jax.device_count()) == 0
    grad_acc_steps = int(cfg.batch_size / (cfg.gpu_batch_size * jax.device_count()))
    micro_batch_size = cfg.gpu_batch_size * jax.device_count()
    print("grad_acc_steps: ", grad_acc_steps)
    print("micro_batch_size: ", micro_batch_size)

    train_loader, val_loader, train_iters, val_iters = create_dataloaders(
        cfg.data, micro_batch_size, cfg.epochs
    )

    model = SSLTeacherStudent(cfg.ssl, mesh=mesh, rngs=rngs)
    param_count = sum(
        jax.tree.map(
            lambda x: jnp.size(x), jax.tree.leaves(nnx.state(model.student, nnx.Param))
        )
    )
    head_param_count = sum(
        jax.tree.map(
            lambda x: jnp.size(x),
            jax.tree.leaves(nnx.state(model.dino_student_head, nnx.Param)),
        )
    )
    print(f"ViT backbone: {param_count / 1_000_000:.2f}M params")
    print(f"ViT head: {head_param_count / 1_000_000:.2f}M params")

    lr_schedule, wd_schedule, mo_schedule = build_schedules(cfg, train_iters)

    chain = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=cfg.adamw_beta1,
            b2=cfg.adamw_beta2,
            weight_decay=cfg.weight_decay_start,  # TODO: how to make schedule?
        ),
    )
    optim = nnx.Optimizer(
        (model.student, model.dino_student_head),
        optax.MultiSteps(chain, every_k_schedule=grad_acc_steps)
        if grad_acc_steps > 1
        else chain,
        wrt=nnx.Param,
    )

    if cfg.checkpoint_every > 0:
        opts = ocp.CheckpointManagerOptions(save_interval_steps=cfg.checkpoint_every)
    else:
        opts = ocp.CheckpointManagerOptions(read_only=True)
    mngr = ocp.CheckpointManager(
        wandb.run.dir if cfg.wandb else Path("outputs"),
        options=opts,
        item_names=("state", "optim", "config"),
    )
    # TODO: this is incorrect? the model is not initialized with the train config
    if cfg.restore:
        step = mngr.latest_step()
        restored = mngr.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(nnx.state(model)),
                optim=ocp.args.StandardRestore(nnx.state(optim)),
                config=ocp.args.JsonRestore(),
            ),
        )
        # restored_state, restored_optim_state = mngr.restore(step)
        nnx.update(model, restored["state"])
        nnx.update(optim, restored["optim"])
        cfg = config_from_dict(restored["config"])

    global_iter = 0
    for epoch in tqdm(range(cfg.epochs), desc="Epoch"):
        train_iter = islice(iter(train_loader), train_iters)
        for samples in tqdm(train_iter, total=train_iters, desc="Batch"):
            samples = jax.device_put(
                samples, NamedSharding(mesh, P("data", None, None, None))
            )
            loss = model(
                optim,
                samples["global_crops"],
                samples["local_crops"],
                student_temp=cfg.student_temp,
                teacher_temp=cfg.teacher_temp,
                teacher_ema_mom=mo_schedule(global_iter),
                update_head=epoch >= cfg.freeze_last_layer_epochs,
            )

            if global_iter % grad_acc_steps == 0:
                model.update_teacher(mo_schedule(global_iter))
                if cfg.wandb:
                    wandb.log({"loss": loss, "lr": lr_schedule(global_iter)})

            global_iter += 1

        mngr.save(
            epoch,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(nnx.state(model)),
                optim=ocp.args.StandardSave(nnx.state(optim)),
                config=ocp.args.JsonSave(asdict(cfg)),
            ),
        )
        mngr.wait_until_finished()

    mngr.close()
    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    cfg: TrainConfig = tyro.cli(TrainConfig)
    main(cfg)
