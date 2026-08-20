import numpy as np
import pytest

from openpi.inference.star import config as _config
from openpi.inference.star import scheduler as _scheduler


def test_config_rejects_reversed_beta_bounds():
    with pytest.raises(ValueError, match="beta bounds"):
        _config.StarConfig(beta_min=0.8, beta_max=0.2).resolve_beta_bounds()


def test_scheduler_returns_valid_momentum_context():
    cfg = _config.StarConfig(stats_window_steps=4, norm_min_count=2, beta_min=0.2, beta_max=0.8)
    scheduler = _scheduler.ProprioPhaseScheduler(cfg)

    ctx0 = scheduler.prepare_context({"observation/state": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])})
    scheduler.update_after_action()
    ctx1 = scheduler.prepare_context({"observation/state": np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])})
    scheduler.update_after_action()
    ctx2 = scheduler.prepare_context({"observation/state": np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0])})

    assert np.isclose(ctx0["beta_t"], 0.5, atol=1e-6)
    assert cfg.resolve_beta_bounds()[0] <= ctx1["beta_t"] <= cfg.resolve_beta_bounds()[1]
    assert cfg.resolve_beta_bounds()[0] <= ctx2["beta_t"] <= cfg.resolve_beta_bounds()[1]
    assert set(ctx2) == {"c_t", "v_t", "j_t", "w_t", "u_t", "r_t", "r_t_tilde", "beta_t", "g_t"}
    assert np.isclose(ctx2["g_t"], ctx2["beta_t"], atol=1e-6)


def test_scheduler_reset_clears_episode_state():
    cfg = _config.StarConfig(stats_window_steps=4, norm_min_count=2, beta_min=0.2, beta_max=0.8)
    scheduler = _scheduler.ProprioPhaseScheduler(cfg)
    scheduler.prepare_context({"observation/state": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])})
    scheduler.update_after_action()

    scheduler.reset_episode()
    ctx = scheduler.prepare_context({"observation/state": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])})
    assert np.isclose(ctx["beta_t"], 0.5, atol=1e-6)


def test_decompose_state_for_6d_7d_8d_layouts():
    pos6, rot6, grip6 = _scheduler.ProprioPhaseScheduler._decompose_state(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
    assert np.allclose(pos6, np.array([1, 2, 3], dtype=np.float32))
    assert np.allclose(rot6, np.array([4, 5, 6], dtype=np.float32))
    assert grip6 is None

    pos7, rot7, grip7 = _scheduler.ProprioPhaseScheduler._decompose_state(
        np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    )
    assert np.allclose(pos7, np.array([1, 2, 3], dtype=np.float32))
    assert np.allclose(rot7, np.array([4, 5, 6], dtype=np.float32))
    assert grip7 == 7.0

    pos8, rot8, grip8 = _scheduler.ProprioPhaseScheduler._decompose_state(
        np.array([1, 2, 3, 4, 5, 6, 8, 10], dtype=np.float32)
    )
    assert np.allclose(pos8, np.array([1, 2, 3], dtype=np.float32))
    assert np.allclose(rot8, np.array([4, 5, 6], dtype=np.float32))
    assert grip8 == 9.0


def test_scheduler_rotation_delta_ignores_gripper_change():
    cfg = _config.StarConfig(stats_window_steps=4, norm_min_count=2)
    scheduler = _scheduler.ProprioPhaseScheduler(cfg)

    scheduler.prepare_context({"observation/state": np.array([0, 0, 0, 0.1, 0.2, 0.3, 0.0, 0.0], dtype=np.float32)})
    scheduler.update_after_action()
    ctx = scheduler.prepare_context({"observation/state": np.array([0, 0, 0, 0.1, 0.2, 0.3, 1.0, 1.0], dtype=np.float32)})

    assert ctx["w_t"] == 0.0
    assert ctx["u_t"] == 1.0


def _state(pos_x: float, rot_z: float, grip: float) -> dict[str, np.ndarray]:
    return {
        "observation/state": np.array([pos_x, 0.0, 0.0, 0.0, 0.0, rot_z, grip, grip], dtype=np.float32),
    }


def _prime_scheduler(scheduler: _scheduler.ProprioPhaseScheduler) -> None:
    scheduler.prepare_context(_state(0.0, 0.0, 0.0))
    scheduler.update_after_action()
    scheduler.prepare_context(_state(0.1, 0.1, 0.0))
    scheduler.update_after_action()
    scheduler.prepare_context(_state(0.2, 0.2, 0.0))
    scheduler.update_after_action()


def test_rotation_distance_uses_quaternion_path_only_for_4d():
    axis_prev = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    axis_cur = np.array([0.0, 0.0, 0.05], dtype=np.float32)
    assert np.isclose(_scheduler.ProprioPhaseScheduler._rotation_distance(axis_cur, axis_prev), 0.05, atol=1e-6)

    # 90-degree rotation around z-axis: quaternion [x, y, z, w].
    quat_prev = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    quat_cur = np.array([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], dtype=np.float32)
    distance = _scheduler.ProprioPhaseScheduler._rotation_distance(quat_cur, quat_prev)
    assert np.isclose(distance, np.pi / 2.0, atol=1e-5)


def test_scheduler_clips_norm_min_count_when_window_is_short(caplog):
    cfg = _config.StarConfig(control_hz=2.0, stats_window_sec=2.0, norm_min_count=8)
    with caplog.at_level("WARNING"):
        scheduler = _scheduler.ProprioPhaseScheduler(cfg)

    assert scheduler._c_norm._min_count == 4
    assert scheduler._v_norm._min_count == 4
    assert any("clipping min_count" in rec.message for rec in caplog.records)


def test_scheduler_low_hz_large_norm_count_beta_not_stuck():
    cfg = _config.StarConfig(control_hz=2.0, stats_window_sec=2.0, norm_min_count=8, beta_min=0.2, beta_max=0.8)
    scheduler = _scheduler.ProprioPhaseScheduler(cfg)

    beta_values = []
    for pos_x in [0.0, 0.1, 0.2, 0.4, 0.8, 1.0, 1.1]:
        ctx = scheduler.prepare_context(
            {"observation/state": np.array([pos_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)}
        )
        beta_values.append(ctx["beta_t"])
        scheduler.update_after_action()

    assert any(abs(beta - 0.5) > 1e-3 for beta in beta_values[4:])


def test_ingest_state_history_drops_current_state_sample():
    cfg = _config.StarConfig(stats_window_steps=20, norm_min_count=2)
    scheduler = _scheduler.ProprioPhaseScheduler(cfg)

    s0 = _state(0.0, 0.0, 0.0)["observation/state"]
    s1 = _state(0.1, 0.0, 0.0)["observation/state"]
    s2 = _state(0.2, 0.0, 0.0)["observation/state"]
    s3 = _state(0.3, 0.0, 0.0)["observation/state"]
    s4 = _state(0.4, 0.0, 0.0)["observation/state"]
    s5 = _state(0.5, 0.0, 0.0)["observation/state"]

    scheduler.prepare_context({"observation/state": s0})
    scheduler.update_after_action()

    replayed = scheduler.ingest_state_history(
        np.stack([s1, s2, s3, s4, s5], axis=0),
        current_observation={"observation/state": s5},
    )
    assert replayed == 4

    # Current-step context should still be computed from s4 -> s5, not from s5 -> s5.
    ctx = scheduler.prepare_context({"observation/state": s5})
    assert ctx["v_t"] > 0.0


def test_ingest_state_history_drops_noisy_current_state_sample():
    cfg = _config.StarConfig(stats_window_steps=20, norm_min_count=2)
    scheduler = _scheduler.ProprioPhaseScheduler(cfg)

    s0 = _state(0.0, 0.0, 0.0)["observation/state"]
    s1 = _state(0.1, 0.0, 0.0)["observation/state"]
    s2 = _state(0.2, 0.0, 0.0)["observation/state"]
    s3 = _state(0.3, 0.0, 0.0)["observation/state"]
    s4 = _state(0.4, 0.0, 0.0)["observation/state"]
    s5 = _state(0.5, 0.0, 0.0)["observation/state"]
    s5_noisy = s5.astype(np.float64) + np.array([5e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    scheduler.prepare_context({"observation/state": s0})
    scheduler.update_after_action()

    replayed = scheduler.ingest_state_history(
        np.stack([s1, s2, s3, s4, s5_noisy], axis=0),
        current_observation={"observation/state": s5.astype(np.float64)},
    )
    assert replayed == 4

    ctx = scheduler.prepare_context({"observation/state": s5})
    assert ctx["v_t"] > 0.0
