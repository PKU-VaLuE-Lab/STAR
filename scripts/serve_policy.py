import dataclasses
import enum
import logging
import socket

import tyro

from openpi.inference.star import config as _star_config
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class StarArgs:
    """STAR runtime args used when inference_mode=star."""

    beta_min: float = 0.25
    beta_max: float = 0.55
    beta_kappa: float = 1.0
    correction_last_n_layers: int = 6
    control_hz: float = 10.0
    stats_window_sec: float = 2.0
    stats_window_steps: int = 0
    norm_min_count: int = 8
    phase_coeff_c: float = 1.0
    phase_coeff_v: float = 1.0
    phase_coeff_j: float = -1.0
    phase_coeff_w: float = -1.0
    action_expert_index: int = 1

    def to_config(self) -> _star_config.StarConfig:
        return _star_config.StarConfig(
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            beta_kappa=self.beta_kappa,
            correction_last_n_layers=self.correction_last_n_layers,
            control_hz=self.control_hz,
            stats_window_sec=self.stats_window_sec,
            stats_window_steps=self.stats_window_steps,
            norm_min_count=self.norm_min_count,
            phase_coeff_c=self.phase_coeff_c,
            phase_coeff_v=self.phase_coeff_v,
            phase_coeff_j=self.phase_coeff_j,
            phase_coeff_w=self.phase_coeff_w,
            action_expert_index=self.action_expert_index,
        )


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Runtime inference mode.
    inference_mode: _star_config.InferenceMode = _star_config.InferenceMode.NATIVE
    # STAR hyperparameters (used only when inference_mode=star).
    star: StarArgs = dataclasses.field(default_factory=StarArgs)
    # Optional control-loop frequency hints for chunked clients (e.g., LIBERO).
    # When set, serve_policy derives effective scheduler frequency as env_control_hz / replan_steps.
    star_env_control_hz: float = 10.0
    star_replan_steps: int | None = None
    # Whether to JIT-compile JAX STAR sample_actions.
    star_jax_jit: bool = True
    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _resolve_star_config(args: Args) -> _star_config.StarConfig | None:
    if args.inference_mode == _star_config.InferenceMode.STAR:
        star_config = args.star.to_config()
        if args.star_replan_steps is not None:
            if args.star_replan_steps <= 0:
                raise ValueError(f"star_replan_steps must be > 0, got {args.star_replan_steps}")
            if args.star_env_control_hz <= 0:
                raise ValueError(f"star_env_control_hz must be > 0, got {args.star_env_control_hz}")

            effective_scheduler_hz = float(args.star_env_control_hz) / float(args.star_replan_steps)
            configured_hz = float(star_config.control_hz)
            tolerance = max(1e-6, 0.05 * max(abs(configured_hz), abs(effective_scheduler_hz), 1.0))
            if abs(configured_hz - effective_scheduler_hz) > tolerance:
                logging.warning(
                    "STAR control_hz=%.3f mismatches effective infer rate %.3f (= %.3f / %d); "
                    "overriding control_hz to %.3f",
                    configured_hz,
                    effective_scheduler_hz,
                    float(args.star_env_control_hz),
                    int(args.star_replan_steps),
                    effective_scheduler_hz,
                )
            updated_fields = {"control_hz": effective_scheduler_hz}
            if int(star_config.stats_window_steps) <= 0:
                window_steps = max(1, round(float(args.star_env_control_hz) * float(star_config.stats_window_sec)))
                updated_fields["stats_window_steps"] = window_steps
                logging.info(
                    "STAR chunked control detected: using stats_window_steps=%d (%.3f Hz * %.3f s) "
                    "to track executed env-step states.",
                    window_steps,
                    float(args.star_env_control_hz),
                    float(star_config.stats_window_sec),
                )
            star_config = dataclasses.replace(star_config, **updated_fields)
        return star_config
    return None


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    inference_mode: _star_config.InferenceMode = _star_config.InferenceMode.NATIVE,
    star_config: _star_config.StarConfig | None = None,
    star_jax_jit: bool = True,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            inference_mode=inference_mode,
            star_config=star_config,
            star_jax_jit=star_jax_jit,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args, *, star_config: _star_config.StarConfig | None = None) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if star_config is None:
        star_config = _resolve_star_config(args)
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                inference_mode=args.inference_mode,
                star_config=star_config,
                star_jax_jit=args.star_jax_jit,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                inference_mode=args.inference_mode,
                star_config=star_config,
                star_jax_jit=args.star_jax_jit,
            )


def main(args: Args) -> None:
    star_config = _resolve_star_config(args)
    policy = create_policy(args, star_config=star_config)
    policy_metadata = {
        **policy.metadata,
        "inference_mode": args.inference_mode.value,
    }
    if star_config is not None:
        policy_metadata["star_jax_jit"] = bool(args.star_jax_jit)
        policy_metadata["star_control_hz"] = float(star_config.control_hz)
        policy_metadata["star_stats_window_sec"] = float(star_config.stats_window_sec)
        policy_metadata["star_stats_window_steps"] = int(star_config.stats_window_steps)
        if args.star_replan_steps is not None:
            policy_metadata["star_replan_steps"] = int(args.star_replan_steps)
            policy_metadata["star_env_control_hz"] = float(args.star_env_control_hz)
    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
