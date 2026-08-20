from flax import nnx
import jax
import numpy as np
import pytest

from openpi.inference.star import config as _star_config
from openpi.inference.star import controller_core_v2 as _star_controller
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


def test_pi0_star_jax_jit_parity():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=8,
    )
    model = config.create(key)
    obs = config.fake_obs(batch_size=1)

    star_config = _star_config.StarConfig()
    star_ctx = _star_controller.make_jax_context(
        {
            "c_t": 0.0,
            "v_t": 0.0,
            "j_t": 0.0,
            "w_t": 0.0,
            "u_t": 0.0,
            "g_t": 0.5,
        }
    )
    star_params = _star_controller.make_jax_params(star_config, model.action_expert_depth)
    noise = jax.random.normal(jax.random.key(2), (1, config.action_horizon, config.action_dim))
    sample_key = jax.random.key(3)

    non_jit_actions = model.sample_actions(
        sample_key,
        obs,
        num_steps=4,
        noise=noise,
        star_ctx=star_ctx,
        star_params=star_params,
    )
    jit_actions = nnx_utils.module_jit(model.sample_actions)(
        sample_key,
        obs,
        num_steps=4,
        noise=noise,
        star_ctx=star_ctx,
        star_params=star_params,
    )

    np.testing.assert_allclose(np.asarray(jit_actions), np.asarray(non_jit_actions), rtol=1e-5, atol=1e-5)


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
