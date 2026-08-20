from __future__ import annotations

import dataclasses
import enum


class InferenceMode(str, enum.Enum):
    """Inference mode for policy serving."""

    NATIVE = "native"
    STAR = "star"


@dataclasses.dataclass(frozen=True)
class StarConfig:
    """Configuration for STAR inference-time correction."""

    # Paper setting in beta_t = beta_min + (beta_max-beta_min) * sigmoid(kappa * r_tilde).
    beta_min: float = 0.25
    beta_max: float = 0.55
    beta_kappa: float = 1.0
    # pi0.5 has 18 action-expert blocks, so six blocks are the final third.
    correction_last_n_layers: int = 6

    control_hz: float = 10.0
    stats_window_sec: float = 2.0
    stats_window_steps: int = 0
    norm_min_count: int = 8
    # Proprio weights used to compute:
    # r_t = phase_coeff_c * z(c_t) + phase_coeff_v * z(v_t)
    #     + phase_coeff_j * z(j_t) + phase_coeff_w * z(w_t)
    # beta_t = beta_min + (beta_max - beta_min) * sigmoid(beta_kappa * r_t_tilde)
    #
    phase_coeff_c: float = 1.0
    phase_coeff_v: float = 1.0
    phase_coeff_j: float = -1.0
    phase_coeff_w: float = -1.0

    # Expert index in the hybrid Gemma stack.
    # For pi0/pi0.5 this is the action expert.
    action_expert_index: int = 1

    def resolve_beta_bounds(self) -> tuple[float, float]:
        """Resolve beta range used by the continuous momentum schedule."""
        lo = float(self.beta_min)
        hi = float(self.beta_max)
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"STAR beta bounds must lie in [0, 1], got ({lo}, {hi})")
        return lo, hi

    def resolve_correction_start(self, total_layers: int) -> int:
        """Compute 1-based layer index where correction starts (last-N layers)."""
        if total_layers <= 0:
            return 1
        n = max(1, int(self.correction_last_n_layers))
        return max(1, int(total_layers) - n + 1)
