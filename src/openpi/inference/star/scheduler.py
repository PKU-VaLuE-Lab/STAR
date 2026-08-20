from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import dataclasses
import logging
from typing import Any

import numpy as np

from openpi.inference.star import config as _config


class SlidingWindowNorm:
    """Causal z-score over a fixed-length history window."""

    def __init__(self, window_size: int, *, eps: float = 1e-6, min_count: int = 4):
        self._window_size = max(1, int(window_size))
        self._eps = float(eps)
        self._min_count = max(1, int(min_count))
        self._buffer: deque[float] = deque(maxlen=self._window_size)

    def reset(self) -> None:
        self._buffer.clear()

    def update(self, value: float) -> None:
        self._buffer.append(float(value))

    def normalize(self, value: float) -> float:
        if len(self._buffer) < self._min_count:
            return 0.0
        arr = np.asarray(self._buffer, dtype=np.float32)
        # Sample std with ddof=1 is undefined for a single element.
        if arr.size < 2:
            return 0.0
        std = float(arr.std(ddof=1))
        if not np.isfinite(std):
            return 0.0
        z = (float(value) - float(arr.mean())) / (std + self._eps)
        if not np.isfinite(z):
            return 0.0
        return float(z)


@dataclasses.dataclass
class _PendingState:
    c_t: float
    v_t: float
    j_t: float
    w_t: float
    u_t: float
    r_t: float
    pos_cur: np.ndarray | None
    rot_cur: np.ndarray | None
    gripper_cur: float | None


class ProprioPhaseScheduler:
    """Compute STAR step context from raw proprioception observations."""

    def __init__(self, config: _config.StarConfig):
        self._config = config
        self._beta_min, self._beta_max = config.resolve_beta_bounds()
        self._beta_kappa = float(config.beta_kappa)
        if config.stats_window_steps > 0:
            window_steps = int(config.stats_window_steps)
        else:
            window_steps = max(1, round(config.control_hz * config.stats_window_sec))

        effective_min_count = max(1, min(int(config.norm_min_count), window_steps))
        if effective_min_count < int(config.norm_min_count):
            logging.warning(
                "STAR norm_min_count=%d exceeds window_steps=%d; clipping min_count to %d.",
                int(config.norm_min_count),
                window_steps,
                effective_min_count,
            )

        self._c_norm = SlidingWindowNorm(window_steps, min_count=effective_min_count)
        self._v_norm = SlidingWindowNorm(window_steps, min_count=effective_min_count)
        self._j_norm = SlidingWindowNorm(window_steps, min_count=effective_min_count)
        self._w_norm = SlidingWindowNorm(window_steps, min_count=effective_min_count)
        self._r_window_size = window_steps
        self._r_min_count = effective_min_count
        self._r_buffer: deque[float] = deque(maxlen=self._r_window_size)
        self._r_eps = 1e-6

        self._prev_pos: np.ndarray | None = None
        self._prev2_pos: np.ndarray | None = None
        self._prev_rot: np.ndarray | None = None
        self._prev_gripper: float | None = None
        self._pending: _PendingState | None = None

    def reset_episode(self) -> None:
        self._prev_pos = None
        self._prev2_pos = None
        self._prev_rot = None
        self._prev_gripper = None
        self._pending = None
        self._c_norm.reset()
        self._v_norm.reset()
        self._j_norm.reset()
        self._w_norm.reset()
        self._r_buffer.clear()

    def prepare_context(self, observation: Mapping[str, Any]) -> dict[str, float]:
        state = self._extract_state(observation)
        pos_cur, rot_cur, gripper_cur = self._decompose_state(state)

        c_t = v_t = j_t = w_t = u_t = 0.0
        dx_t: np.ndarray | None = None
        if pos_cur is not None and self._prev_pos is not None:
            dx_t = pos_cur - self._prev_pos
            v_t = float(np.linalg.norm(dx_t))
        if dx_t is not None and self._prev_pos is not None and self._prev2_pos is not None:
            dx_prev = self._prev_pos - self._prev2_pos
            c_t = self._safe_cos(dx_t, dx_prev)
            j_t = float(np.linalg.norm(dx_t - dx_prev))

        if rot_cur is not None and self._prev_rot is not None:
            w_t = self._rotation_distance(rot_cur, self._prev_rot)
        if gripper_cur is not None and self._prev_gripper is not None:
            u_t = abs(gripper_cur - self._prev_gripper)

        zc = self._clip(self._c_norm.normalize(c_t))
        zv = self._clip(self._v_norm.normalize(v_t))
        zj = self._clip(self._j_norm.normalize(j_t))
        zw = self._clip(self._w_norm.normalize(w_t))
        # New STAR scheduler: r_t is a weighted sum of normalized proprio metrics.
        r_t = (
            float(self._config.phase_coeff_c) * zc
            + float(self._config.phase_coeff_v) * zv
            + float(self._config.phase_coeff_j) * zj
            + float(self._config.phase_coeff_w) * zw
        )
        r_t_tilde = self._normalize_r(r_t)
        beta_t = self._beta_from_r_tilde(r_t_tilde)

        self._pending = _PendingState(
            c_t=float(c_t),
            v_t=float(v_t),
            j_t=float(j_t),
            w_t=float(w_t),
            u_t=float(u_t),
            r_t=float(r_t),
            pos_cur=None if pos_cur is None else pos_cur.copy(),
            rot_cur=None if rot_cur is None else rot_cur.copy(),
            gripper_cur=None if gripper_cur is None else float(gripper_cur),
        )
        return {
            "c_t": float(c_t),
            "v_t": float(v_t),
            "j_t": float(j_t),
            "w_t": float(w_t),
            "u_t": float(u_t),
            "r_t": float(r_t),
            "r_t_tilde": float(r_t_tilde),
            "beta_t": float(beta_t),
            # Legacy key retained for compatibility with existing downstream logs/interfaces.
            "g_t": float(beta_t),
        }

    def update_after_action(self) -> None:
        if self._pending is None:
            return
        self._c_norm.update(self._pending.c_t)
        self._v_norm.update(self._pending.v_t)
        self._j_norm.update(self._pending.j_t)
        self._w_norm.update(self._pending.w_t)
        self._r_buffer.append(float(self._pending.r_t))

        self._prev2_pos = None if self._prev_pos is None else self._prev_pos.copy()
        self._prev_pos = self._pending.pos_cur
        self._prev_rot = self._pending.rot_cur
        self._prev_gripper = self._pending.gripper_cur
        self._pending = None

    def ingest_state_history(self, state_history: Any, *, current_observation: Mapping[str, Any] | None = None) -> int:
        """Replay executed proprio states so stats track real env-step sequences.

        This is useful for chunked control loops where policy inference runs every
        N steps but environment actions are executed at a higher rate.
        """
        seq = self._to_state_sequence(state_history)
        if seq is None or seq.shape[0] == 0:
            return 0

        # If history includes the latest observation state, drop it so we can
        # still compute current-step context from (prev_state -> current_state).
        if current_observation is not None:
            current_state = self._extract_state(current_observation)
            if current_state is not None and self._states_close(seq[-1], current_state, atol=1e-4):
                seq = seq[:-1]
        if seq.shape[0] == 0:
            return 0

        replayed = 0
        for state in seq:
            self.prepare_context({"observation/state": state})
            self.update_after_action()
            replayed += 1
        return replayed

    def _normalize_r(self, r_t: float) -> float:
        if len(self._r_buffer) < self._r_min_count:
            return 0.0
        arr = np.asarray(self._r_buffer, dtype=np.float32)
        if arr.size < 2:
            return 0.0
        q25 = float(np.quantile(arr, 0.25))
        q50 = float(np.quantile(arr, 0.50))
        q75 = float(np.quantile(arr, 0.75))
        iqr = q75 - q25
        if not np.isfinite(iqr) or abs(iqr) < self._r_eps:
            return 0.0
        z = (float(r_t) - q50) / (iqr + self._r_eps)
        if not np.isfinite(z):
            return 0.0
        return float(z)

    def _beta_from_r_tilde(self, r_t_tilde: float) -> float:
        sigmoid_input = float(np.clip(self._beta_kappa * float(r_t_tilde), -60.0, 60.0))
        sig = float(1.0 / (1.0 + np.exp(-sigmoid_input)))
        beta = self._beta_min + (self._beta_max - self._beta_min) * sig
        return float(np.clip(beta, self._beta_min, self._beta_max))

    @staticmethod
    def _clip(value: float, clip_val: float = 2.5) -> float:
        return float(np.clip(value, -clip_val, clip_val))

    @staticmethod
    def _safe_cos(a: np.ndarray, b: np.ndarray) -> float:
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an < 1e-8 or bn < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    @staticmethod
    def _rotation_distance(cur: np.ndarray, prev: np.ndarray) -> float:
        cur = cur.reshape(-1).astype(np.float32, copy=False)
        prev = prev.reshape(-1).astype(np.float32, copy=False)
        # Use geodesic distance only for explicit 4D quaternion inputs.
        # Axis-angle / Euler-like vectors should use Euclidean distance.
        if cur.size == 4 and prev.size == 4:
            q1 = cur[:4]
            q2 = prev[:4]
            n1 = float(np.linalg.norm(q1))
            n2 = float(np.linalg.norm(q2))
            if n1 < 1e-8 or n2 < 1e-8:
                return 0.0
            q1 = q1 / n1
            q2 = q2 / n2
            cos_sim = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
            return float(2.0 * np.arccos(cos_sim))
        dim = min(cur.size, prev.size)
        return float(np.linalg.norm(cur[:dim] - prev[:dim]))

    @staticmethod
    def _extract_state(observation: Mapping[str, Any]) -> np.ndarray | None:
        if "observation/state" in observation:
            return ProprioPhaseScheduler._to_state_vec(observation["observation/state"])
        if "state" in observation:
            return ProprioPhaseScheduler._to_state_vec(observation["state"])
        obs_obj = observation.get("observation")
        if isinstance(obs_obj, Mapping) and "state" in obs_obj:
            return ProprioPhaseScheduler._to_state_vec(obs_obj["state"])
        return None

    @staticmethod
    def _to_state_vec(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1)
        elif arr.ndim == 2:
            arr = arr[-1].reshape(-1)
        else:
            arr = arr.reshape(arr.shape[0], -1)[-1]
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _to_state_sequence(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim == 2:
            arr = arr.reshape(arr.shape[0], -1)
        else:
            arr = arr.reshape(arr.shape[0], -1)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _states_close(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-6) -> bool:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        dim = min(a.size, b.size)
        if dim == 0:
            return False
        return bool(np.allclose(a[:dim], b[:dim], atol=atol, rtol=0.0))

    @staticmethod
    def _decompose_state(state: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        if state is None:
            return None, None, None
        if state.size < 3:
            return None, None, None
        pos = state[:3]
        rot: np.ndarray | None = None
        # For LIBERO/openpi proprio states, rotation is represented as 3D axis-angle
        # and gripper channels come afterwards.
        if state.size >= 6:
            rot = state[3:6]
        gripper: float | None = None
        if state.size >= 8:
            gripper = float(np.mean(state[6:8]))
        elif state.size >= 7:
            gripper = float(state[6])
        return pos, rot, gripper
