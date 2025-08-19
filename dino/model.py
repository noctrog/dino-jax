from typing import Tuple
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

    embed_dim: int = 384
    num_layers: int = 12
    mlp_hidden_dim: int = 1536
    num_heads: int = 6


@dataclass
class SSLDinoConfig:
    weight: float = 1.0
    head_n_prototypes: int = 65536
    head_bottleneck: int = 256
    head_nlayers: int = 3
    head_hidden_dim: int = 2048


@dataclass
class SSLConfig:
    dino: SSLDinoConfig = field(default_factory=lambda: SSLDinoConfig())
    vit: ViTConfig = field(default_factory=lambda: ViTConfig())


class PatchEmbed(nnx.Module):
    def __init__(self, cfg: ViTConfig, rngs: nnx.Rngs):
        image_hw = [cfg.img_size] * 2 if isinstance(cfg.img_size, int) else cfg.img_size
        patch_hw = (
            [cfg.patch_size] * 2 if isinstance(cfg.patch_size, int) else cfg.patch_size
        )
        grid_size = [image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1]]

        self.patch_size = cfg.patch_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.proj = nnx.Linear(
            cfg.in_channels * patch_hw[0] * patch_hw[1],
            cfg.embed_dim,
            use_bias=False,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        _, H, W, _ = x.shape
        ph, pw = self.patch_size

        x = rearrange(x, "b (h ph) (w pw) c -> b (h w) (c ph pw)", ph=ph, pw=pw)
        x = self.proj(x)
        return x


class MLP(nnx.Module):
    def __init__(self, cfg: ViTConfig, rngs: nnx.Rngs):
        linear_kwargs = {"use_bias": False, "param_dtype": jnp.float32}
        self.up_proj = nnx.Linear(
            cfg.embed_dim, cfg.mlp_hidden_dim, rngs=rngs, **linear_kwargs
        )
        self.gate_proj = nnx.Linear(
            cfg.embed_dim, cfg.mlp_hidden_dim, rngs=rngs, **linear_kwargs
        )
        self.down_proj = nnx.Linear(
            cfg.mlp_hidden_dim, cfg.embed_dim, rngs=rngs, **linear_kwargs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nnx.Module):
    def __init__(self, cfg: ViTConfig, rngs: nnx.Rngs):
        self.num_heads = cfg.num_heads

        linear_kwargs = {"use_bias": False, "param_dtype": jnp.float32}
        self.qkv_proj = nnx.Linear(
            cfg.embed_dim, 3 * cfg.embed_dim, rngs=rngs, **linear_kwargs
        )
        self.o_proj = nnx.Linear(
            cfg.embed_dim, cfg.embed_dim, rngs=rngs, **linear_kwargs
        )

    def __call__(
        self, x: jax.Array, attention_mask: jax.Array | None = None
    ) -> jax.Array:
        q, k, v = jnp.split(self.qkv_proj(x), 3, axis=-1)

        q = rearrange(q, "b t (n c) -> b t n c", n=self.num_heads)
        k = rearrange(k, "b t (n c) -> b t n c", n=self.num_heads)
        v = rearrange(v, "b t (n c) -> b t n c", n=self.num_heads)
        att = jax.nn.dot_product_attention(q, k, v, mask=attention_mask)
        att = rearrange(att, "b t n c -> b t (n c)")
        att = self.o_proj(att)
        return att


class TransformerDecoderLayer(nnx.Module):
    def __init__(self, cfg: ViTConfig, rngs: nnx.Rngs):
        self.attention = Attention(cfg, rngs)
        self.mlp = MLP(cfg, rngs)
        self.att_norm = nnx.RMSNorm(cfg.embed_dim, rngs=rngs)
        self.mlp_norm = nnx.RMSNorm(cfg.embed_dim, rngs=rngs)

    def __call__(
        self, x: jax.Array, attention_mask: jax.Array | None = None
    ) -> jax.Array:
        x = x + self.attention(self.att_norm(x), attention_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ViT(nnx.Module):
    def __init__(self, cfg: ViTConfig, rngs: nnx.Rngs):
        self.embed_dim = cfg.embed_dim
        self.patch_embed = PatchEmbed(cfg, rngs)

        default_kernel_init = nnx.initializers.lecun_normal()
        init_key = rngs.params()
        cls_key, init_key = jax.random.split(init_key)
        self.cls_token = nnx.Param(default_kernel_init(cls_key, (1, 1, cfg.embed_dim)))
        pos_key, init_key = jax.random.split(init_key)
        self.pos_embed = nnx.Param(
            default_kernel_init(
                pos_key, (1, self.patch_embed.num_patches, cfg.embed_dim)
            )
        )

        self.layers = [
            TransformerDecoderLayer(cfg, rngs) for _ in range(cfg.num_layers)
        ]
        self.norm = nnx.RMSNorm(cfg.embed_dim, epsilon=1e-6, rngs=rngs)

    def interpolate_pos_encoding(self, x: jax.Array) -> jax.Array:
        assert x.ndim == 3
        hw = math.isqrt(x.shape[1])
        assert hw**2 == x.shape[1]
        hw_posemb = math.isqrt(self.pos_embed.shape[1])
        assert hw_posemb**2 == self.pos_embed.shape[1]

        if x.shape[1] == self.pos_embed.shape[1]:
            return self.pos_embed

        pos_embed_2d = rearrange(self.pos_embed, "b (h w) d -> b h w d", h=hw_posemb)
        pos_embed_resized = jax.image.resize(
            pos_embed_2d, shape=(hw, hw), method="bilinear", antialias=False
        )
        return rearrange(pos_embed_resized, "b h w d -> b (h w) d")

    def __call__(self, x: jax.Array | list[jax.Array]) -> dict[str, jax.Array]:
        """Forward pass of the ViT.

        Args:
          x (jax.Array): the input tensor of shape BHWD.
        """
        bs = x.shape[0] if isinstance(x, jax.Array) else x[0].shape[0]
        tokens = jax.tree.map(self.patch_embed, x)
        cls_token = jnp.broadcast_to(self.cls_token, (bs, 1, self.embed_dim))
        tokens = jax.tree.map(lambda x: jnp.concatenate((cls_token, x), axis=1), tokens)
        lens = jax.tree.map(lambda x: x.shape[1], tokens)
        ones = jax.tree.map(
            lambda x: jnp.ones((x.shape[1], x.shape[1]), dtype=jnp.bool_), tokens
        )
        tokens = jnp.concatenate(jax.tree.leaves(tokens), axis=1)
        attn_mask = jax.scipy.linalg.block_diag(*jax.tree.leaves(ones))

        for block in self.layers:
            tokens = block(tokens, attention_mask=attn_mask)

        tokens = self.norm(tokens)
        starts = jnp.concatenate((jnp.array([0]), jnp.cumsum(jnp.array(lens[:-1]))))
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
            kernel_init=nnx.initializers.truncated_normal(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
    else:
        layers = [
            nnx.Linear(
                in_dim,
                hidden_dim,
                use_bias=bias,
                kernel_init=nnx.initializers.truncated_normal(),
                bias_init=nnx.initializers.zeros_init(),
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
                    kernel_init=nnx.initializers.truncated_normal(),
                    bias_init=nnx.initializers.zeros_init(),
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
                kernel_init=nnx.initializers.truncated_normal(),
                bias_init=nnx.initializers.zeros_init(),
                use_bias=bias,
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
        self.last_layer = nnx.Linear(bottleneck_dim, out_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.mlp(x)
        eps = 1e-12 if x.dtype == jnp.float32 else 1e-6
        norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        x = x / jnp.maximum(norm, eps)

        kernel = self.last_layer.kernel
        kernel_norm = jnp.linalg.norm(kernel, ord=2, axis=0, keepdims=True)
        normalized_kernel = kernel / jnp.maximum(kernel_norm, eps)

        x = x @ normalized_kernel
        return x


class DINOLoss(nnx.Module):
    def __init__(
        self, out_dim: int, center_momentum: float = 0.9, *, mesh: jax.sharding.Mesh
    ):
        self.center_momentum = center_momentum
        self.center = nnx.Variable(jnp.zeros((1, 1, out_dim)))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None
        self.mesh = mesh

    def __call__(
        self,
        student_logits: list[jax.Array],
        teacher_logits: list[jax.Array],
        student_temp: float,
        teacher_temp: float,
    ) -> tuple[float | jax.Array, jax.Array]:
        student_logprob = jax.tree.map(
            lambda x: jax.nn.log_softmax(x / student_temp, axis=-1), student_logits
        )
        teacher_probs = jax.tree.map(
            lambda x: jax.nn.softmax((x - self.center) / teacher_temp, axis=-1),
            teacher_logits,
        )
        n_terms = 0
        total_loss = 0
        for i_s in range(student_logprob.shape[1]):
            for i_t in range(teacher_probs.shape[1]):
                if i_s == i_t:
                    continue
                lsm, t = student_logprob[:, i_s], teacher_probs[:, i_t]
                total_loss -= jnp.sum(t * lsm, axis=-1).mean()
                n_terms += 1

        new_center = self.update_center(teacher_logits)
        return total_loss / n_terms, new_center

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

        return self.center * self.center_momentum + global_center * (
            1 - self.center_momentum
        )


class SSLTeacherStudent(nnx.Module):
    def __init__(self, cfg: SSLConfig, mesh: jax.sharding.Mesh, rngs: nnx.Rngs):
        self.cfg = cfg
        self.student = ViT(cfg=cfg.vit, rngs=rngs)
        self.teacher = ViT(cfg=cfg.vit, rngs=rngs)
        self.dino_student_head = DINOHead(
            cfg.vit.embed_dim,
            cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck,
            num_layers=cfg.dino.head_nlayers,
            rngs=rngs,
        )
        self.dino_teacher_head = DINOHead(
            cfg.vit.embed_dim,
            cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck,
            num_layers=cfg.dino.head_nlayers,
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
    ) -> tuple[float | jax.Array, nnx.GraphState]:
        teacher_output = self.teacher(global_crops)
        teacher_logits = self.dino_teacher_head(teacher_output["cls"])  # (B, N, C)

        def loss_fn(vit: ViT, head: DINOHead, teacher_logits: jax.Array):
            student_output = vit(global_crops + local_crops)
            student_logits = head(student_output["cls"])
            return self.dino_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_temp=student_temp,
                teacher_temp=teacher_temp,
            )

        grad_fn = nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
        (loss, new_center), grads = grad_fn(
            self.student, self.dino_student_head, teacher_logits
        )
        self.dino_loss.center = new_center
        optim.update([self.student, self.dino_student_head], grads)
        return loss, grads

    def update_teacher(self, momentum: float) -> None:
        new_teacher_state = jax.tree.map(
            lambda t, s: t * momentum + s * (1 - momentum),
            nnx.state(self.teacher),
            nnx.state(self.student),
        )
        new_teacher_head_state = jax.tree.map(
            lambda t, s: t * momentum + s * (1 - momentum),
            nnx.state(self.dino_teacher_head),
            nnx.state(self.dino_student_head),
        )

        nnx.update(self.teacher, new_teacher_state)
        nnx.update(self.dino_teacher_head, new_teacher_head_state)


if __name__ == "__main__":
    from jax.sharding import PartitionSpec as P, NamedSharding
    import optax

    mesh = jax.make_mesh((2,), ("data",))
    input = [jnp.ones((32, 224, 224, 3)), jnp.ones((32, 96, 96, 3))]
    input = jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, P("data", None, None, None))),
        input,
    )
    rngs = nnx.Rngs(0)

    ssl = SSLTeacherStudent(SSLConfig(), mesh=mesh, rngs=rngs)
    optim = nnx.Optimizer(
        (ssl.student, ssl.dino_student_head), optax.adamw(3e-4, 0.9), wrt=nnx.Param
    )

    @nnx.jit
    def step(ssl, optim, input1, input2):
        return ssl(optim, input1, input2, 1.0, 1.0)

    loss, grads = step(ssl, optim, input, input)
    # jax.debug.visualize_array_sharding(input[0][:, :, 0, 0])
    # jax.debug.visualize_array_sharding(jax.tree.leaves(grads)[2])
    print("loss: ", loss)
    ssl.update_teacher(0.9)
