from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.inference.star import config as _config


class TorchStarLayerController:
    """STAR controller for layer-wise PyTorch hidden-state correction."""

    def __init__(self, config: _config.StarConfig, total_layers: int):
        self._config = config
        self._total_layers = int(total_layers)
        self._correction_start_layer = config.resolve_correction_start(self._total_layers)
        self._beta_min, self._beta_max = config.resolve_beta_bounds()
        self._step_ctx = {
            "c_t": 0.0,
            "v_t": 0.0,
            "j_t": 0.0,
            "w_t": 0.0,
            "u_t": 0.0,
            "r_t": 0.0,
            "r_t_tilde": 0.0,
            "beta_t": 0.5 * (self._beta_min + self._beta_max),
            # Keep legacy key for compatibility with existing logging paths.
            "g_t": 0.5 * (self._beta_min + self._beta_max),
        }
        self.reset()

    @property
    def correction_start_layer(self) -> int:
        return self._correction_start_layer

    @property
    def probe_end(self) -> int:
        # Legacy property retained for compatibility with earlier tests/logging.
        return max(1, self._correction_start_layer - 1)

    def set_step_context(self, ctx: dict[str, float]) -> None:
        self._step_ctx = dict(ctx)

    def reset(self) -> None:
        self.layer_count = 0
        self.am: torch.Tensor | None = None

    def step(self, h_in: torch.Tensor, h_out: torch.Tensor) -> torch.Tensor:
        self.layer_count += 1
        delta = h_out - h_in

        if self.layer_count < self._correction_start_layer:
            return h_out

        beta_t = float(self._step_ctx.get("beta_t", self._step_ctx.get("g_t", self._beta_min)))
        beta_t = float(np.clip(beta_t, self._beta_min, self._beta_max))
        beta = torch.full((delta.shape[0], 1, 1), beta_t, dtype=delta.dtype, device=delta.device)
        # EMA warm start: initialize with first correction-layer delta so the first corrected
        # update matches the original transformer update (am == delta).
        am_prev = delta if self.am is None else self.am
        self.am = beta * am_prev + (1.0 - beta) * delta
        return h_in + self.am


@struct.dataclass
class JaxStarParams:
    beta_min: float
    beta_max: float
    correction_start_layer: int = struct.field(pytree_node=False)
    action_expert_index: int = struct.field(pytree_node=False)


@struct.dataclass
class JaxStarContext:
    c_t: jax.Array
    v_t: jax.Array
    j_t: jax.Array
    w_t: jax.Array
    u_t: jax.Array
    r_t: jax.Array
    r_t_tilde: jax.Array
    beta_t: jax.Array
    g_t: jax.Array


@struct.dataclass
class JaxStarState:
    layer_count: jax.Array
    am: jax.Array
    has_correction: jax.Array


def make_jax_params(config: _config.StarConfig, total_layers: int) -> JaxStarParams:
    beta_min, beta_max = config.resolve_beta_bounds()
    return JaxStarParams(
        beta_min=float(beta_min),
        beta_max=float(beta_max),
        correction_start_layer=int(config.resolve_correction_start(total_layers)),
        action_expert_index=int(config.action_expert_index),
    )


def make_jax_context(step_ctx: dict[str, float]) -> JaxStarContext:
    beta_val = float(step_ctx.get("beta_t", step_ctx.get("g_t", 0.5)))
    return JaxStarContext(
        c_t=jnp.asarray(step_ctx.get("c_t", 0.0), dtype=jnp.float32),
        v_t=jnp.asarray(step_ctx.get("v_t", 0.0), dtype=jnp.float32),
        j_t=jnp.asarray(step_ctx.get("j_t", 0.0), dtype=jnp.float32),
        w_t=jnp.asarray(step_ctx.get("w_t", 0.0), dtype=jnp.float32),
        u_t=jnp.asarray(step_ctx.get("u_t", 0.0), dtype=jnp.float32),
        r_t=jnp.asarray(step_ctx.get("r_t", 0.0), dtype=jnp.float32),
        r_t_tilde=jnp.asarray(step_ctx.get("r_t_tilde", 0.0), dtype=jnp.float32),
        beta_t=jnp.asarray(beta_val, dtype=jnp.float32),
        g_t=jnp.asarray(step_ctx.get("g_t", beta_val), dtype=jnp.float32),
    )


def init_jax_state(hidden_like: jax.Array, params: JaxStarParams) -> JaxStarState:
    return JaxStarState(
        layer_count=jnp.asarray(0, dtype=jnp.int32),
        am=jnp.zeros_like(hidden_like),
        has_correction=jnp.asarray(0, dtype=jnp.bool_),
    )


def jax_star_step(
    state: JaxStarState,
    h_in: jax.Array,
    h_out: jax.Array,
    ctx: JaxStarContext,
    params: JaxStarParams,
) -> tuple[JaxStarState, jax.Array]:
    delta = (h_out - h_in).astype(state.am.dtype)
    layer_count = state.layer_count + jnp.asarray(1, dtype=jnp.int32)
    correction_start = jnp.asarray(params.correction_start_layer, dtype=jnp.int32)

    def _skip_branch(_: None) -> tuple[JaxStarState, jax.Array]:
        return state.replace(layer_count=layer_count), h_out

    def _corr_branch(_: None) -> tuple[JaxStarState, jax.Array]:
        beta_scalar = jnp.clip(
            ctx.beta_t.astype(jnp.float32),
            jnp.asarray(params.beta_min, dtype=jnp.float32),
            jnp.asarray(params.beta_max, dtype=jnp.float32),
        )
        beta = jnp.broadcast_to(beta_scalar.astype(delta.dtype), (delta.shape[0], 1, 1))
        am_prev = jax.lax.cond(
            state.has_correction,
            lambda _: state.am,
            # EMA warm start with the first correction-layer delta.
            lambda _: delta,
            operand=None,
        )
        am = beta * am_prev + (1.0 - beta) * delta
        corrected = (h_in + am).astype(h_out.dtype)
        next_state = state.replace(
            layer_count=layer_count,
            am=am,
            has_correction=jnp.asarray(1, dtype=jnp.bool_),
        )
        return next_state, corrected

    return jax.lax.cond(layer_count < correction_start, _skip_branch, _corr_branch, operand=None)
