from __future__ import annotations

import logging
import types
from typing import Any

from openpi.inference.star import config as _config
from openpi.inference.star import controller_core_v2 as _controller_core

logger = logging.getLogger(__name__)


def enable_torch_star(model: Any, star_config: _config.StarConfig) -> None:
    """Enable STAR layer controller on the PyTorch action expert stack."""
    try:
        gemma_model = model.paligemma_with_expert.gemma_expert.model
    except AttributeError as exc:
        raise ValueError("Cannot locate PyTorch action expert model to enable STAR.") from exc

    if not hasattr(gemma_model, "layers"):
        raise ValueError("Action expert model does not expose decoder layers.")

    total_layers = len(gemma_model.layers)
    controller = _controller_core.TorchStarLayerController(star_config, total_layers)

    if getattr(gemma_model, "_star_patched", False):
        gemma_model._star_controller = controller  # noqa: SLF001
        model._star_controller = controller  # noqa: SLF001
        model.reset_star_state = lambda: controller.reset()  # type: ignore[attr-defined]
        return

    wrapped_layers = []
    for idx, layer in enumerate(gemma_model.layers):
        orig_forward = layer.forward
        wrapped_layers.append(orig_forward)

        def _wrapped_forward(self_layer, *args, __orig=orig_forward, __idx=idx, **kwargs):
            hidden_in = args[0] if args else kwargs.get("hidden_states")
            if hidden_in is None:
                return __orig(*args, **kwargs)
            active_controller = getattr(gemma_model, "_star_controller", None)
            if active_controller is None:
                return __orig(*args, **kwargs)
            if __idx == 0:
                active_controller.reset()
            out = __orig(*args, **kwargs)
            hidden_out = out[0] if isinstance(out, tuple) else out
            hidden_out = active_controller.step(hidden_in, hidden_out)
            if isinstance(out, tuple):
                return (hidden_out, *out[1:])
            return hidden_out

        layer.forward = types.MethodType(_wrapped_forward, layer)

    gemma_model._star_orig_forwards = wrapped_layers  # noqa: SLF001
    gemma_model._star_patched = True  # noqa: SLF001
    gemma_model._star_controller = controller  # noqa: SLF001
    model._star_controller = controller  # noqa: SLF001
    model.reset_star_state = lambda: controller.reset()  # type: ignore[attr-defined]
    logger.info(
        "Enabled PyTorch STAR (layers=%s, correction_start_layer=%s).",
        total_layers,
        controller.correction_start_layer,
    )


def reset_torch_star(model: Any) -> None:
    controller = getattr(model, "_star_controller", None)
    if controller is not None:
        controller.reset()
