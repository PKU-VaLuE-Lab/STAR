from unittest import mock

import jax.numpy as jnp
import numpy as np
import torch  # pyright: ignore[reportMissingImports]

from openpi.inference.star import config as _star_config
from openpi.inference.star import controller_core_v2 as _star_controller
from openpi.inference.star import scheduler as _scheduler
from openpi.policies import policy as _policy


class _DummyModel:
    def __init__(self):
        self.reset_calls = 0

    def sample_actions(self, *_args, **_kwargs):
        return np.zeros((1, 2, 3), dtype=np.float32)

    def reset_star_state(self):
        self.reset_calls += 1


def test_policy_reset_calls_scheduler_and_model_reset():
    model = _DummyModel()
    config = _star_config.StarConfig()
    scheduler = _scheduler.ProprioPhaseScheduler(config)
    policy = _policy.Policy(
        model,
        is_pytorch=False,
        inference_mode=_star_config.InferenceMode.STAR,
        star_config=config,
        star_scheduler=scheduler,
        star_jax_jit=False,
    )
    scheduler.prepare_context({"observation/state": np.zeros(8, dtype=np.float32)})
    scheduler.update_after_action()

    policy.reset()
    assert model.reset_calls == 1


class _TorchNativeDummyModel:
    def __init__(self):
        self.last_kwargs = None

    def to(self, _device):
        return self

    def eval(self):
        return self

    def sample_actions(self, _device, _observation, **kwargs):
        self.last_kwargs = kwargs
        return torch.zeros((1, 2, 3), dtype=torch.float32)


def test_native_mode_does_not_inject_star_kwargs():
    model = _TorchNativeDummyModel()
    policy = _policy.Policy(
        model,
        is_pytorch=True,
        inference_mode=_star_config.InferenceMode.NATIVE,
    )
    obs = {
        "state": np.zeros(3, dtype=np.float32),
        "image": {
            "base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.array(True),
            "left_wrist_0_rgb": np.array(True),
            "right_wrist_0_rgb": np.array(True),
        },
    }
    policy.infer(obs)
    assert model.last_kwargs is not None
    assert "star_ctx" not in model.last_kwargs


class _JaxStarDummyModel:
    def __init__(self):
        self.action_expert_depth = 4
        self.last_kwargs = None

    def sample_actions(self, *_args, **kwargs):
        self.last_kwargs = kwargs
        return jnp.zeros((1, 2, 3), dtype=jnp.float32)

    def reset_star_state(self):
        return None


def _make_dummy_obs():
    return {
        "state": np.zeros(8, dtype=np.float32),
        "image": {
            "base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.array(True),
            "left_wrist_0_rgb": np.array(True),
            "right_wrist_0_rgb": np.array(True),
        },
        "tokenized_prompt": np.zeros((200,), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((200,), dtype=bool),
    }


def test_star_jax_infer_injects_typed_context_and_params():
    model = _JaxStarDummyModel()
    config = _star_config.StarConfig()
    scheduler = _scheduler.ProprioPhaseScheduler(config)
    policy = _policy.Policy(
        model,
        is_pytorch=False,
        inference_mode=_star_config.InferenceMode.STAR,
        star_config=config,
        star_scheduler=scheduler,
        star_jax_jit=False,
    )

    policy.infer(_make_dummy_obs())

    assert model.last_kwargs is not None
    assert isinstance(model.last_kwargs.get("star_ctx"), _star_controller.JaxStarContext)
    assert isinstance(model.last_kwargs.get("star_params"), _star_controller.JaxStarParams)


def test_star_jax_infer_replays_state_history():
    model = _JaxStarDummyModel()
    config = _star_config.StarConfig(stats_window_steps=20, norm_min_count=2)
    scheduler = _scheduler.ProprioPhaseScheduler(config)
    policy = _policy.Policy(
        model,
        is_pytorch=False,
        inference_mode=_star_config.InferenceMode.STAR,
        star_config=config,
        star_scheduler=scheduler,
        star_jax_jit=False,
    )

    obs = _make_dummy_obs()
    obs["observation/state"] = np.zeros(8, dtype=np.float32)
    obs["observation/state_history"] = np.stack(
        [
            np.array([0.1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            np.array([0.2, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            np.array([0.3, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        ],
        axis=0,
    )

    with mock.patch.object(scheduler, "ingest_state_history", wraps=scheduler.ingest_state_history) as ingest_history:
        policy.infer(obs)

    ingest_history.assert_called_once()


def test_star_jax_jit_path_selection():
    config = _star_config.StarConfig()
    scheduler = _scheduler.ProprioPhaseScheduler(config)

    with mock.patch.object(_policy.nnx_utils, "module_jit", side_effect=lambda fn: fn) as module_jit:
        _policy.Policy(
            _JaxStarDummyModel(),
            is_pytorch=False,
            inference_mode=_star_config.InferenceMode.STAR,
            star_config=config,
            star_scheduler=scheduler,
            star_jax_jit=True,
        )
        assert module_jit.call_count == 1

    with mock.patch.object(_policy.nnx_utils, "module_jit", side_effect=lambda fn: fn) as module_jit:
        _policy.Policy(
            _JaxStarDummyModel(),
            is_pytorch=False,
            inference_mode=_star_config.InferenceMode.STAR,
            star_config=config,
            star_scheduler=scheduler,
            star_jax_jit=False,
        )
        assert module_jit.call_count == 0
