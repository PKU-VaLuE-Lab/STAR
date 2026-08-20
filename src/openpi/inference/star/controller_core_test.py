import jax.numpy as jnp
import numpy as np
import torch

from openpi.inference.star import config as _config
from openpi.inference.star import controller_core_v2 as _controller_core


def test_torch_controller_applies_momentum_only_on_last_layers():
    cfg = _config.StarConfig(correction_last_n_layers=2, beta_min=0.2, beta_max=0.2)
    controller = _controller_core.TorchStarLayerController(cfg, total_layers=4)
    controller.set_step_context({"beta_t": 0.2})

    h = torch.zeros((1, 2, 3), dtype=torch.float32)
    outs = []
    for i in range(4):
        h_next = h + torch.full_like(h, (i + 1) * 0.1)
        out = controller.step(h, h_next)
        outs.append((h_next, out))
        h = h_next

    assert torch.allclose(outs[0][1], outs[0][0])
    assert torch.allclose(outs[1][1], outs[1][0])
    # First correction layer is EMA warm-started to delta, so it matches the raw update.
    assert torch.allclose(outs[2][1], outs[2][0], atol=1e-7)
    assert not torch.allclose(outs[3][1], outs[3][0])


def test_torch_controller_clamps_beta_to_configured_bounds():
    cfg = _config.StarConfig(correction_last_n_layers=1, beta_min=0.2, beta_max=0.8)
    controller = _controller_core.TorchStarLayerController(cfg, total_layers=1)
    controller.set_step_context({"beta_t": 10.0})  # should clamp to 0.8

    h_in = torch.zeros((1, 2, 3), dtype=torch.float32)
    h_out = torch.ones((1, 2, 3), dtype=torch.float32)
    out = controller.step(h_in, h_out)

    # First correction layer uses warm-start am_prev=delta, so corrected output equals h_out.
    expected = torch.ones_like(h_in)
    assert torch.allclose(out, expected, atol=1e-6)


def test_jax_controller_applies_momentum_only_in_correction_band():
    cfg = _config.StarConfig(correction_last_n_layers=2, beta_min=0.2, beta_max=0.2)
    params = _controller_core.make_jax_params(cfg, total_layers=4)
    ctx = _controller_core.make_jax_context({"beta_t": 0.2, "r_t_tilde": 0.0})

    h = jnp.zeros((1, 2, 3), dtype=jnp.float32)
    state = _controller_core.init_jax_state(h, params)
    outs = []
    for i in range(4):
        h_next = h + np.float32((i + 1) * 0.1)
        state, out = _controller_core.jax_star_step(state, h, h_next, ctx, params)
        outs.append((np.asarray(h_next), np.asarray(out)))
        h = h_next

    assert np.allclose(outs[0][1], outs[0][0], atol=1e-7)
    assert np.allclose(outs[1][1], outs[1][0], atol=1e-7)
    assert np.allclose(outs[2][1], outs[2][0], atol=1e-7)
    assert not np.allclose(outs[3][1], outs[3][0], atol=1e-7)
