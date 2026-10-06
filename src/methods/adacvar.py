"""Adaptive sampling machinery for the AdaCVaR baseline.

Implements the algorithm-named adacvar variant from Curi, Levy, Jegelka, Krause,
Adaptive Sampling for Stochastic Risk-Averse Learning (NeurIPS 2020). The training
loop draws minibatches from a non-uniform sampler that emphasises high-loss training
examples; eval is identical to a plain base classifier. Reference implementation at
github.com/sebascuri/adacvar.

The k-DPP marginal approximation follows Barthelme, Amblard, Tremblay, Asymptotic
equivalence of fixed-size and varying-size determinantal point processes (arXiv:1803.01576).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from torch.utils.data import BatchSampler, Dataset

if TYPE_CHECKING:
    from collections.abc import Iterator


def kdpp_marginals(log_values: np.ndarray, k: int) -> np.ndarray:
    """Approximate k-DPP marginals.

    Given unnormalised per-element log-weights, returns marginals lambda summing
    to k with lambda_i = sigmoid(nu + log_values_i). The scalar nu is the unique
    root of sum_i sigmoid(nu + log_values_i) = k, found via scipy.optimize.brentq
    with an adaptively expanded bracket centred at log(k/(n-k)). log_values are
    shifted by their max for numerical stability; the marginals are shift-invariant
    in nu plus log_values, so this does not change the result.

    Edge cases:
        k == n returns all-ones. The equation has no finite root because each
        sigmoid asymptotes to 1; the limiting behaviour is lambda_i = 1, which
        corresponds to deterministically including every index in the size-n subset.
        k == 1 is well-defined; the resulting lambda places most mass on the highest
        log_value with a soft drop-off.

    Args:
        log_values: Per-element unnormalised log-weights. Shape (n,).
        k: Target marginal sum. Must satisfy 1 <= k <= n.

    Returns:
        Marginals of shape (n,) with sum equal to k within scipy.optimize.brentq's
        xtol-derived precision, well under 1e-6 in practice, and elementwise values
        in [0, 1].

    Raises:
        ValueError: if k is not in the inclusive range [1, n], if log_values is not
            1-D, or if log_values contains any non-finite entries (NaN or Inf).
        RuntimeError: if the adaptive bracket expansion exceeds 1e8 without locating
            a positive value of f, indicating numerically pathological inputs.
    """
    n = int(log_values.shape[0])
    if k < 1 or k > n:
        raise ValueError(f"Need 1 <= k <= n={n}, got k={k}.")
    if log_values.ndim != 1:
        raise ValueError(f"log_values must be 1-D, got shape {log_values.shape}.")
    if not np.all(np.isfinite(log_values)):
        raise ValueError("log_values must be finite (no NaN, no Inf).")
    if k == n:
        return np.ones(n, dtype=np.float64)
    log_v = log_values - log_values.max()  # shift for numerical stability

    def f(nu: float) -> float:
        return float(expit(nu + log_v).sum() - k)

    center = math.log(k / (n - k))
    fc = f(center)
    # In theory f(center) <= 0 whenever log_v <= 0 (post-shift), with equality iff
    # log_v is uniform. In practice floating-point error can leave fc tiny positive
    # when log_v is uniform or near-uniform. In that case center already is (or is
    # numerically indistinguishable from) the root, so return its marginals directly
    # without invoking brentq (which would refuse a same-sign bracket).
    if fc >= 0.0:
        return expit(center + log_v)
    # f(center) < 0: expand b upward until f(b) > 0 to bracket the root.
    b = center + 50.0
    fb = f(b)
    while fb <= 0.0:
        b += b - center  # double the distance from center
        if b > center + 1e8:
            raise RuntimeError(
                f"kdpp_marginals: bracket expansion exceeded 1e8. "
                f"log_values range: [{float(log_v.min())}, {float(log_v.max())}], k={k}, n={n}."
            )
        fb = f(b)
    nu = brentq(f, center, b)
    return expit(nu + log_v)


class IndexedDataset(Dataset[tuple[Any, Any, int]]):
    """Wraps a Dataset to yield (x, y, idx) tuples.

    The idx is the position in this wrapper, 0 to len(base) - 1. It is used by
    Exp3Sampler to address its per-example weight vector. Underlying Subset or
    random_split datasets pass through unchanged because we only thread the position
    number, not the original-dataset index.
    """

    def __init__(self, base: Dataset[tuple[Any, Any]]) -> None:
        """Wrap base. Length and item access delegate to it.

        Args:
            base: Dataset yielding (x, y) on getitem.
        """
        self.base = base

    def __len__(self) -> int:
        """Forward to the wrapped dataset."""
        return len(self.base)  # ty: ignore[invalid-argument-type]

    def __getitem__(self, index: int) -> tuple[Any, Any, int]:
        """Return (x, y, index) where x, y come from the wrapped dataset.

        Args:
            index: Position in this wrapper.

        Returns:
            Tuple of the unpacked wrapped item plus the position.
        """
        x, y = self.base[index]
        return x, y, index


class Exp3Sampler(BatchSampler):
    """Two-stage adaptive batch sampler for AdaCVaR.

    Yields lists of indices to be used as the batch_sampler in a DataLoader. Each
    iteration first samples k = ceil(alpha * N) indices without replacement weighted
    by approximate k-DPP marginals, then samples batch_size indices uniformly with
    replacement from that subset. The per-example log-weights are mutated externally
    via update(losses, indices) after each training step.

    The marginals follow Barthelme et al. (2018); see kdpp_marginals. The EG update
    follows Exp3 with importance weighting by the marginals; see Auer et al. (2002).
    Loss values are clipped to [0, 1] before the update, matching the reference at
    github.com/sebascuri/adacvar/blob/master/adacvar/util/adaptive_algorithm.py.
    """

    def __init__(
        self,
        num_actions: int,
        batch_size: int,
        alpha: float,
        eta: float,
        gamma: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise with all log-weights at 0 (uniform marginals at start).

        Args:
            num_actions: Total training set size N.
            batch_size: Minibatch size B yielded each iteration.
            alpha: CVaR tail fraction. Sets k = max(1, ceil(alpha * N)).
            eta: Constant EG learning rate applied per update step.
            gamma: Uniform mixing coefficient. Marginals become
                (1 - gamma) * lambda + gamma * (k / N). Default 0.
            rng: Optional numpy Generator for reproducible sampling. None uses
                np.random.default_rng() seeded by global state.

        Raises:
            ValueError: if alpha is outside (0, 1] or batch_size > num_actions.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
        if batch_size > num_actions:
            raise ValueError(f"batch_size={batch_size} exceeds num_actions={num_actions}.")
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.k = max(1, int(np.ceil(alpha * num_actions)))
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.log_values = np.zeros(num_actions, dtype=np.float64)
        self._rng = rng if rng is not None else np.random.default_rng()

    @property
    def probabilities(self) -> np.ndarray:
        """Current k-DPP marginals, optionally mixed with uniform.

        Returns:
            Marginals of shape (N,), each in [0, 1], summing to k within tolerance.
        """
        lam = kdpp_marginals(self.log_values, self.k)
        if self.gamma > 0.0:
            uniform = self.k / self.num_actions
            lam = (1.0 - self.gamma) * lam + self.gamma * uniform
        return lam

    def __len__(self) -> int:
        """Number of batches per epoch (N // batch_size, matching reference)."""
        return self.num_actions // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        """Yield successive minibatches via the two-stage sampling.

        Yields:
            A list of batch_size integer indices in 0 to num_actions - 1.
        """
        for _ in range(len(self)):
            lam = self.probabilities
            total = lam.sum()
            p = lam / total
            subset = self._rng.choice(self.num_actions, size=self.k, replace=False, p=p)
            batch = self._rng.choice(subset, size=self.batch_size, replace=True)
            yield batch.tolist()

    def update(self, losses: np.ndarray, indices: np.ndarray) -> None:
        """EG update on the sampled minibatch.

        Applies log_values[i] -= eta * (1 - clip(L_i, 0, 1)) / lambda_i for each i in
        indices. Uses np.add.at to accumulate when an index appears multiple times in
        the minibatch (which happens because the batch is sampled with replacement
        from the subset). The importance correction divides by the k-DPP marginal,
        matching the reference Exp3.update.

        Args:
            losses: Per-instance losses observed on the minibatch. Shape (B,).
            indices: Minibatch indices, aligned with losses. Shape (B,).
        """
        lam = self.probabilities
        clipped = np.clip(losses, 0.0, 1.0)
        delta = -self.eta * (1.0 - clipped) / (lam[indices] + 1e-12)
        np.add.at(self.log_values, indices, delta)

    def normalize(self) -> None:
        """Subtract max(log_values) to keep numerical range bounded.

        kdpp_marginals is shift-invariant in nu plus log_values, so subtracting a
        scalar from log_values does not change the resulting marginals. Called once
        per epoch by the training loop.
        """
        self.log_values -= self.log_values.max()
