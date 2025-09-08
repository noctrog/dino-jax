from typing import Literal
from functools import partial
from dataclasses import dataclass, field
import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import nnx
from einops import rearrange


@dataclass
class ViTConfig:
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    configuration: Literal["vitt", "vits", "vitb", "vitl"] = "vits"

    drop_rate: float = 0.0

    selective: bool = False
    """If True, checkpoint the attention computations for the backward pass."""

    def __post_init__(self):
        config = VIT_CONFIGS[self.configuration]
        self.embed_dim = config["embed_dim"]
        self.num_layers = config["num_layers"]
        self.mlp_hidden_dim = config["hidden_dim"]
        self.num_heads = config["num_heads"]


VIT_CONFIGS = {
    "vitt": {"embed_dim": 192, "num_layers": 12, "hidden_dim": 768, "num_heads": 3},
    "vits": {"embed_dim": 384, "num_layers": 12, "hidden_dim": 1536, "num_heads": 6},
    "vitb": {"embed_dim": 768, "num_layers": 12, "hidden_dim": 3072, "num_heads": 12},
    "vitl": {"embed_dim": 1024, "num_layers": 24, "hidden_dim": 4096, "num_heads": 16},
}


@dataclass
class SSLDinoConfig:
    weight: float = 1.0
    head_n_prototypes: int = 65536
    head_bottleneck: int = 256
    head_nlayers: int = 3
    head_hidden_dim: int = 2048
    norm_last_layer: bool = True
    """If True, the g term (scale) of the weight norm of the last layer will not be trained.
    This increases stability but hinders accuracy. It can be disabled to train ViT-T, ViT-S.
    """


@dataclass
class SSLConfig:
    dino: SSLDinoConfig = field(default_factory=lambda: SSLDinoConfig())
    vit: ViTConfig = field(default_factory=lambda: ViTConfig())


class DropPath(nnx.Module):
    def __init__(
        self,
        drop_prob: float,
        deterministic: bool = False,
        rng_collection: str = "dropout",
        *,
        rngs: nnx.Rngs,
    ):
        self.drop_prob = drop_prob
        self.deterministic = deterministic
        self.rng_collection = "dropout"
        self.rngs = rngs[self.rng_collection].fork()

    def __call__(self, x: jax.Array, deterministic: bool | None = None):
        det = deterministic if deterministic is not None else self.deterministic
        if self.drop_prob == 0.0 or det:
            return x
        else:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = jax.random.bernoulli(self.rngs(), p=keep_prob, shape=shape)
            mask = jnp.broadcast_to(mask, x.shape)
            return jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class PatchEmbed(nnx.Module):
    def __init__(self, cfg: ViTConfig, rngs: nnx.Rngs):
        image_hw = [cfg.img_size] * 2 if isinstance(cfg.img_size, int) else cfg.img_size
        patch_hw = [cfg.patch_size] * 2 if isinstance(cfg.patch_size, int) else cfg.patch_size
        grid_size = [image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1]]

        self.patch_size = cfg.patch_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.proj = nnx.Linear(
            cfg.in_channels * patch_hw[0] * patch_hw[1],
            cfg.embed_dim,
            use_bias=False,
            kernel_init=nnx.initializers.truncated_normal(0.02),
            bias_init=nnx.initializers.zeros_init(),
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        _, H, W, _ = x.shape
        ph, pw = self.patch_size, self.patch_size

        x = rearrange(x, "b (h ph) (w pw) c -> b (h w) (c ph pw)", ph=ph, pw=pw)
        x = self.proj(x)
        return x


class MLP(nnx.Module):
    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        linear_kwargs = {
            "use_bias": False,
            "dtype": jnp.bfloat16,
            "param_dtype": jnp.float32,
            "kernel_init": nnx.initializers.truncated_normal(0.02),
            "bias_init": nnx.initializers.zeros_init(),
        }
        self.up_proj = nnx.Linear(cfg.embed_dim, cfg.mlp_hidden_dim, rngs=rngs, **linear_kwargs)
        self.down_proj = nnx.Linear(cfg.mlp_hidden_dim, cfg.embed_dim, rngs=rngs, **linear_kwargs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(jnp.bfloat16)
        x = self.down_proj(jax.nn.gelu(self.up_proj(x)))
        return x.astype(jnp.float32)


class Attention(nnx.Module):
    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs):
        self.num_heads = cfg.num_heads
        self.attn_fn = (
            jax.checkpoint(jax.nn.dot_product_attention)
            if cfg.selective
            else jax.nn.dot_product_attention
        )

        linear_kwargs = {
            "use_bias": False,
            "dtype": jnp.bfloat16,
            "param_dtype": jnp.float32,
            "kernel_init": nnx.initializers.truncated_normal(0.02),
            "bias_init": nnx.initializers.zeros_init(),
        }
        self.qkv_proj = nnx.Linear(cfg.embed_dim, 3 * cfg.embed_dim, rngs=rngs, **linear_kwargs)
        self.o_proj = nnx.Linear(cfg.embed_dim, cfg.embed_dim, rngs=rngs, **linear_kwargs)

    def __call__(self, x: jax.Array, attention_mask: jax.Array | None = None) -> jax.Array:
        x = x.astype(jnp.bfloat16)
        q, k, v = jnp.split(self.qkv_proj(x), 3, axis=-1)

        q = rearrange(q, "b t (n c) -> b t n c", n=self.num_heads)
        k = rearrange(k, "b t (n c) -> b t n c", n=self.num_heads)
        v = rearrange(v, "b t (n c) -> b t n c", n=self.num_heads)
        att = self.attn_fn(q, k, v, mask=attention_mask, implementation="cudnn")
        att = rearrange(att, "b t n c -> b t (n c)")
        att = self.o_proj(att)
        att = att.astype(jnp.float32)
        return att


class TransformerDecoderLayer(nnx.Module):
    def __init__(self, cfg: ViTConfig, drop_prob: float, *, rngs: nnx.Rngs):
        self.attention = Attention(cfg, rngs=rngs)
        self.drop_path = DropPath(drop_prob, rngs=rngs)
        self.mlp = MLP(cfg, rngs=rngs)
        self.att_norm = nnx.LayerNorm(
            cfg.embed_dim,
            scale_init=nnx.initializers.ones_init(),
            bias_init=nnx.initializers.zeros_init(),
            epsilon=1e-6,
            rngs=rngs,
        )
        self.mlp_norm = nnx.LayerNorm(
            cfg.embed_dim,
            scale_init=nnx.initializers.ones_init(),
            bias_init=nnx.initializers.zeros_init(),
            epsilon=1e-6,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        attn = self.attention(self.att_norm(x), attention_mask)
        x = x + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))
        return x


class ViT(nnx.Module):
    def __init__(self, cfg: ViTConfig, rngs: nnx.Rngs):
        self.embed_dim = cfg.embed_dim
        self.patch_embed = PatchEmbed(cfg, rngs)

        pos_embed_init = nnx.initializers.truncated_normal(0.02)
        cls_kernel_init = nnx.initializers.truncated_normal(0.02)
        init_key = rngs.params()

        cls_key, pos_key = jax.random.split(init_key)
        self.cls_token = nnx.Param(cls_kernel_init(cls_key, (1, 1, cfg.embed_dim)))
        self.pos_embed = nnx.Param(
            pos_embed_init(pos_key, (1, self.patch_embed.num_patches + 1, cfg.embed_dim))
        )

        # TODO: add the drop rate into the transformer decoder layers
        # drp = [i * cfg.drop_rate / (cfg.num_layers - 1) for i in range(cfg.num_layers)]

        self.layers = [TransformerDecoderLayer(cfg, 0.0, rngs=rngs) for _ in range(cfg.num_layers)]

        self.norm = nnx.LayerNorm(
            cfg.embed_dim,
            scale_init=nnx.initializers.ones_init(),
            bias_init=nnx.initializers.zeros_init(),
            epsilon=1e-6,
            rngs=rngs,
        )

    def interpolate_pos_encoding(self, x: jax.Array) -> jax.Array:
        assert x.ndim == 3
        hw = math.isqrt(x.shape[1])
        assert hw**2 == x.shape[1]
        hw_posemb = math.isqrt(self.pos_embed.shape[1] - 1)
        assert hw_posemb**2 == self.pos_embed.shape[1] - 1

        if x.shape[1] == self.pos_embed.shape[1] - 1:
            return self.pos_embed[:, 1:]

        pos_embed_2d = rearrange(self.pos_embed[:, 1:], "b (h w) d -> b h w d", h=hw_posemb)
        pos_embed_resized = jax.image.resize(
            pos_embed_2d,
            shape=(1, hw, hw, self.embed_dim),
            method="bilinear",
            antialias=False,
        )
        return rearrange(pos_embed_resized, "b h w d -> b (h w) d")

    def __call__(self, x: jax.Array | list[jax.Array]) -> dict[str, jax.Array]:
        """Forward pass of the ViT.

        Args:
          x (jax.Array | list[jax.Array]): the input tensor of shape BHWD.
        """
        bs = x.shape[0] if isinstance(x, jax.Array) else x[0].shape[0]
        tokens = jax.tree.map(self.patch_embed, x)
        pos_embeds = jax.tree.map(self.interpolate_pos_encoding, tokens)
        cls_token = jnp.broadcast_to(self.cls_token + self.pos_embed[:, 0], (bs, 1, self.embed_dim))
        tokens = jax.tree.map(
            lambda x, p: jnp.concatenate((cls_token, x + p), axis=1), tokens, pos_embeds
        )
        lens = jax.tree.leaves(jax.tree.map(lambda x: x.shape[1], tokens))
        ones = jax.tree.map(lambda x: jnp.ones((x.shape[1], x.shape[1]), dtype=jnp.bool_), tokens)
        tokens = jnp.concatenate(jax.tree.leaves(tokens), axis=1)
        attn_mask = jax.scipy.linalg.block_diag(*jax.tree.leaves(ones))

        for block in self.layers:
            tokens = block(tokens, attention_mask=attn_mask)

        tokens = self.norm(tokens)
        starts = jnp.concatenate(
            (jnp.array([0]), jnp.cumsum(jnp.array(lens[:-1], dtype=jnp.int32)))
        )
        return {"cls": tokens[:, starts, :]}


def _build_mlp(
    nlayers: int,
    in_dim: int,
    bottleneck_dim: int,
    hidden_dim: int,
    *,
    use_bn: bool = False,
    bias: bool = True,
    rngs: nnx.Rngs,
) -> nnx.Module:
    if nlayers == 1:
        return nnx.Linear(
            in_dim,
            bottleneck_dim,
            use_bias=bias,
            kernel_init=nnx.initializers.truncated_normal(0.02),
            bias_init=nnx.initializers.zeros_init(),
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
    else:
        layers = [
            nnx.Linear(
                in_dim,
                hidden_dim,
                use_bias=bias,
                kernel_init=nnx.initializers.truncated_normal(0.02),
                bias_init=nnx.initializers.zeros_init(),
                dtype=jnp.bfloat16,
                param_dtype=jnp.float32,
                rngs=rngs,
            )
        ]
        if use_bn:
            layers.append(nnx.BatchNorm(hidden_dim, rngs=rngs))
        layers.append(nnx.gelu)
        for _ in range(nlayers - 2):
            layers.append(
                nnx.Linear(
                    hidden_dim,
                    hidden_dim,
                    use_bias=bias,
                    kernel_init=nnx.initializers.truncated_normal(0.02),
                    bias_init=nnx.initializers.zeros_init(),
                    dtype=jnp.bfloat16,
                    param_dtype=jnp.float32,
                    rngs=rngs,
                )
            )
            if use_bn:
                layers.append(nnx.BatchNorm(hidden_dim, rngs=rngs))
            layers.append(nnx.gelu)
        layers.append(
            nnx.Linear(
                hidden_dim,
                bottleneck_dim,
                kernel_init=nnx.initializers.truncated_normal(0.02),
                bias_init=nnx.initializers.zeros_init(),
                use_bias=bias,
                dtype=jnp.bfloat16,
                param_dtype=jnp.float32,
                rngs=rngs,
            )
        )
        return nnx.Sequential(*layers)


class DINOHead(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        use_bn: bool = False,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        mlp_bias: bool = True,
        norm_last_layer: bool = False,
        rngs: nnx.Rngs,
    ):
        num_layers = max(num_layers, 1)
        self.mlp = _build_mlp(
            num_layers,
            in_dim,
            bottleneck_dim,
            hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
            rngs=rngs,
        )

        self.norm_g = nnx.Param(jnp.ones((1, out_dim))) if not norm_last_layer else None
        self.last_layer = nnx.Param(
            nnx.initializers.truncated_normal(0.02)(rngs.params(), (bottleneck_dim, out_dim))
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(jnp.bfloat16)
        x = self.mlp(x).astype(jnp.float32)
        eps = 1e-12 if x.dtype == jnp.float32 else 1e-6
        norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        x = x / jnp.maximum(norm, eps)

        # Weight Normalization of the last layer
        v_norm = jnp.linalg.norm(self.last_layer, ord=2, axis=0, keepdims=True)
        w = self.last_layer / jnp.maximum(v_norm, eps)
        if self.norm_g is not None:
            w = w * self.norm_g
        x = x @ w
        return x


class DINOLoss(nnx.Module):
    def __init__(
        self,
        out_dim: int,
        center_momentum: float = 0.9,
        *,
        mesh: jax.sharding.Mesh | None,
    ):
        self.center_momentum = center_momentum
        self.center = nnx.Variable(jnp.zeros((1, out_dim)))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None
        self.mesh = mesh

    def __call__(
        self,
        student_logits: jax.Array,
        teacher_logits: jax.Array,
        student_temp: float,
        teacher_temp: float,
    ) -> tuple[float | jax.Array, jax.Array]:
        S, T = student_logits.shape[1], teacher_logits.shape[1]
        student_logprob = jax.nn.log_softmax(student_logits / student_temp, axis=-1)
        teacher_probs = jax.nn.softmax((teacher_logits - self.center) / teacher_temp, axis=-1)

        student_logprob = student_logprob[:, :, None, :]  # BS1L
        teacher_probs = teacher_probs[:, None, :, :]  # B1TL

        mask = jnp.ones((S, T), dtype=jnp.bool_).at[jnp.arange(T), jnp.arange(T)].set(False)
        batch_loss = -jnp.sum(teacher_probs * student_logprob, axis=-1).mean(axis=0) * mask  # ST
        n_terms = (S - 1) * T

        new_center = self.update_center(teacher_logits)
        return jnp.sum(batch_loss) / n_terms, new_center

    def update_center(self, teacher_output: jax.Array):
        def compute_global_center(teacher_output: jax.Array) -> jax.Array:
            teacher_output = teacher_output.reshape(-1, teacher_output.shape[-1])
            batch_center = jnp.sum(teacher_output, axis=0, keepdims=True)
            global_center = jax.lax.psum(batch_center, axis_name="data")
            total_samples = jax.lax.psum(teacher_output.shape[0], axis_name="data")
            return global_center / total_samples

        sharded_compute = jax.shard_map(
            compute_global_center,
            mesh=self.mesh,
            in_specs=P("data", None, None),
            out_specs=P(None, None),
        )

        global_center = sharded_compute(teacher_output)

        return self.center * self.center_momentum + global_center * (1 - self.center_momentum)


@partial(nnx.value_and_grad, argnums=(2, 3), has_aux=True)
def loss_fn(
    teacher_vit: ViT,
    teacher_head: DINOHead,
    student_vit: ViT,
    student_head: DINOHead,
    dino_loss: DINOLoss,
    global_crops: jax.Array,
    local_crops: jax.Array,
    student_temp: float,
    teacher_temp: float,
):
    """Returns the loss and the center update"""
    teacher_output = teacher_vit(global_crops)
    teacher_logits = teacher_head(teacher_output["cls"])
    student_output = student_vit(global_crops + local_crops)
    student_logits = student_head(student_output["cls"])
    return dino_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_temp=student_temp,
        teacher_temp=teacher_temp,
    )


@partial(nnx.jit, static_argnames=("update_last_layer",))
def train_step(
    ssl: "SSLTeacherStudent",
    optim: nnx.Optimizer,
    global_crops: jax.Array,
    local_crops: jax.Array,
    student_temp: float,
    teacher_temp: float,
    update_last_layer: bool,
) -> tuple[float | jax.Array, jax.Array]:
    (loss, new_center), grads = loss_fn(
        ssl.teacher,
        ssl.dino_teacher_head,
        ssl.student,
        ssl.dino_student_head,
        ssl.dino_loss,
        global_crops=global_crops,
        local_crops=local_crops,
        student_temp=student_temp,
        teacher_temp=teacher_temp,
    )
    if not update_last_layer:
        new_state = jax.tree.map(
            jnp.zeros_like,
            nnx.state(
                grads,
                nnx.Any(nnx.PathContains("last_layer"), nnx.PathContains("norm_g")),
            ),
        )
        nnx.update(grads, new_state)

    optim.update((ssl.student, ssl.dino_student_head), grads)
    return loss, new_center


class SSLTeacherStudent(nnx.Module):
    def __init__(self, cfg: SSLConfig, mesh: jax.sharding.Mesh | None, rngs: nnx.Rngs):
        self.cfg = cfg
        self.student = ViT(cfg=cfg.vit, rngs=rngs)
        self.teacher = ViT(cfg=cfg.vit, rngs=rngs)
        self.dino_student_head = DINOHead(
            cfg.vit.embed_dim,
            cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck,
            num_layers=cfg.dino.head_nlayers,
            norm_last_layer=cfg.dino.norm_last_layer,
            rngs=rngs,
        )
        self.dino_teacher_head = DINOHead(
            cfg.vit.embed_dim,
            cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck,
            num_layers=cfg.dino.head_nlayers,
            norm_last_layer=cfg.dino.norm_last_layer,
            rngs=rngs,
        )
        self.dino_loss = DINOLoss(cfg.dino.head_n_prototypes, mesh=mesh)

        nnx.update(self.teacher, nnx.state(self.student))
        nnx.update(self.dino_teacher_head, nnx.state(self.dino_student_head))

    def __call__(
        self,
        optim: nnx.Optimizer,
        global_crops: list[jax.Array],
        local_crops: list[jax.Array],
        student_temp: float,
        teacher_temp: float,
        update_last_layer: bool = True,
    ) -> tuple[float | jax.Array, nnx.GraphState]:
        loss, new_center = train_step(
            self,
            optim,
            global_crops=global_crops,
            local_crops=local_crops,
            student_temp=student_temp,
            teacher_temp=teacher_temp,
            update_last_layer=update_last_layer,
        )
        self.dino_loss.center.value = new_center
        return loss

    @nnx.jit
    def update_teacher(self, momentum: float) -> None:
        new_state = jax.tree.map(
            lambda t, s: t * momentum + s * (1 - momentum),
            nnx.state((self.teacher, self.dino_teacher_head), nnx.Param),
            nnx.state((self.student, self.dino_student_head), nnx.Param),
        )
        nnx.update((self.teacher, self.dino_teacher_head), new_state)
