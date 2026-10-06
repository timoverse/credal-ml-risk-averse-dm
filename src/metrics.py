"""Per-instance losses + scalar aggregations, with registries for config-driven dispatch.

A single risk value is a composition: aggregation(loss_per_instance(probs, targets)).
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

# =========================
# PER-INSTANCE LOSSES (probs, targets) -> vector
# =========================


def log_loss_per_instance(
    probs: npt.NDArray[np.floating], targets: npt.NDArray[np.integer]
) -> npt.NDArray[np.floating]:
    """Per-instance log loss -log probs[i, targets[i]], clamped at the dtype machine epsilon to avoid -inf."""
    p_true = probs[np.arange(len(targets)), targets]
    eps = np.finfo(probs.dtype).eps
    return -np.log(np.maximum(p_true, eps))


def brier_per_instance(probs: npt.NDArray[np.floating], targets: npt.NDArray[np.integer]) -> npt.NDArray[np.floating]:
    """Per-instance Brier score, the squared distance to the one-hot label.

    Equals sum_j probs[i, j] ** 2 - 2 probs[i, targets[i]] + 1.
    """
    p_true = probs[np.arange(len(targets)), targets]
    return (probs**2).sum(axis=-1) - 2.0 * p_true + 1.0


def zero_one_per_instance(
    probs: npt.NDArray[np.floating], targets: npt.NDArray[np.integer]
) -> npt.NDArray[np.floating]:
    """Per-instance 0-1 loss: 0 if argmax matches the target, 1 otherwise."""
    preds = probs.argmax(axis=-1)
    return (preds != targets).astype(np.float64)


PER_INSTANCE_LOSS_REGISTRY: dict[str, Callable[[npt.NDArray, npt.NDArray], npt.NDArray[np.floating]]] = {
    "log_loss": log_loss_per_instance,
    "brier": brier_per_instance,
    "zero_one": zero_one_per_instance,
}


# =========================
# AGGREGATIONS: vector -> scalar
# =========================


def cvar(losses: npt.NDArray[np.floating], q: float) -> float:
    """Mean of the worst q fraction of per-instance losses."""
    sorted_losses = np.sort(losses)
    return float(sorted_losses[-int(q * len(losses)) :].mean())


def cvar_fixed(losses: npt.NDArray[np.floating], n: int) -> float:
    """Mean of the worst n per-instance losses."""
    sorted_losses = np.sort(losses)
    return float(sorted_losses[-n:].mean())


def _make_cvar(q: float) -> Callable[[npt.NDArray], float]:
    return lambda losses: cvar(losses, q=q)


def _make_cvar_fixed(n: int) -> Callable[[npt.NDArray], float]:
    return lambda losses: cvar_fixed(losses, n=n)


AGGREGATION_REGISTRY: dict[str, Callable[[npt.NDArray], float]] = {
    "mean": lambda losses: float(np.mean(losses)),
    "cvar_0.01": _make_cvar(0.01),
    "cvar_0.025": _make_cvar(0.025),
    "cvar_0.05": _make_cvar(0.05),
    "cvar_0.1": _make_cvar(0.1),
    "cvar_0.2": _make_cvar(0.2),
    "cvar_0.4": _make_cvar(0.4),
    "cvar_0.5": _make_cvar(0.5),
    "cvar_0.6": _make_cvar(0.6),
    "cvar_0.8": _make_cvar(0.8),
    "cvar_0.95": _make_cvar(0.95),
    "cvar_1.0": _make_cvar(1.0),  # worst 100% = mean (expected risk)
    "cvar_10": _make_cvar_fixed(10),
    "cvar_50": _make_cvar_fixed(50),
    "cvar_100": _make_cvar_fixed(100),
    "cvar_200": _make_cvar_fixed(200),
}


def risk_metric(loss_name: str, aggregation_name: str) -> Callable[[npt.NDArray, npt.NDArray], float]:
    """Compose a per-instance loss with a scalar aggregation into a (probs, targets) -> scalar callable.

    Args:
        loss_name: Key in PER_INSTANCE_LOSS_REGISTRY.
        aggregation_name: Key in AGGREGATION_REGISTRY.

    Returns:
        Callable that maps (probs, targets) to aggregation(loss_per_instance(probs, targets)).

    Raises:
        ValueError: loss_name or aggregation_name is not in the corresponding registry.
    """
    if loss_name not in PER_INSTANCE_LOSS_REGISTRY:
        raise ValueError(f"Unknown loss={loss_name!r}. Choose from {sorted(PER_INSTANCE_LOSS_REGISTRY)}.")
    if aggregation_name not in AGGREGATION_REGISTRY:
        raise ValueError(f"Unknown aggregation={aggregation_name!r}. Choose from {sorted(AGGREGATION_REGISTRY)}.")
    loss_fn = PER_INSTANCE_LOSS_REGISTRY[loss_name]
    agg_fn = AGGREGATION_REGISTRY[aggregation_name]
    return lambda probs, targets: agg_fn(loss_fn(probs, targets))
