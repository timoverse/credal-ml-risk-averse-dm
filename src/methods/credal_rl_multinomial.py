"""Local multinomial relative-likelihood credal sets; see compute_multinomial_bounds for the algorithm."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast, override

import torch
import torch.nn as nn
from probly.predictor import LogitDistributionPredictor
from probly.representation.credal_set import ProbabilityIntervalsCredalSet
from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet
from probly.representer._representer import Representer, representer

if TYPE_CHECKING:
    from collections.abc import Mapping


def fit_evidence_reference(
    features: torch.Tensor,
    targets: torch.Tensor,
    ridge: float,
    num_classes: int,
    *,
    whitening: float = 1.0,
    bandwidth_anchor: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit the whitened reference the local evidence is collected against.

    Features are L2-normalised and mapped by a tempered whitening of the ridged tied covariance of
    the normalised features: eigenvalues are raised to the power -whitening/2, so whitening = 1
    gives full Mahalanobis whitening, whitening = 0 a pure rotation (plain normalised distances),
    and values between interpolate. Full whitening sharpens far-OOD detection but manufactures
    hub structure on datasets with a dense majority class (the dermamnist and covid_xray vote
    collapse); lower exponents remove the hubs at the source. The whitening = 1 computation is the
    verbatim legacy path, so earlier artifacts and results reproduce bitwise. The returned
    bandwidth base is the median distance of each reference point to its ``bandwidth_anchor``-th
    nearest OTHER reference point in the chosen metric; the working kernel bandwidth is a configured
    multiple of it. Everything is computed in float64 and returned in float32 for storage.

    WHY THE ANCHOR IS A PARAMETER. compute_evidence sums the kernel over the ``num_neighbors``
    nearest reference points, but with the default anchor of 1 the SCALE of that kernel is set by
    how close the single closest point is. The two only track each other while the reference is
    dense. Shrink the reference and the radius that actually contains ``num_neighbors`` points grows,
    while the 1-nearest-neighbour distance barely moves -- in 2048-d Inception features it is nearly
    invariant, and it is further flattened by near-duplicate images of the same patient. Since a
    neighbour contributes exp(-d^2 / 2h^2), that decoupling is exponential: measured on covid_xray,
    dropping the training set to 10% (14815 -> 1482 points) left the base at 48.8 versus 50.8 but
    collapsed the evidence mass past the vacuity cutoff, so EVERY credal set became the unit box
    while the kernel vote itself stayed accurate (0.916 vs 0.921).

    Setting ``bandwidth_anchor = num_neighbors`` measures instead the radius that contains exactly
    the points the kernel sums over, which makes the typical d/h ratio -- the only thing the Gaussian
    sees -- invariant to the reference size.

    NOT to be confused with a per-query adaptive bandwidth. The bandwidth must adapt to the SIZE OF
    THE REFERENCE, a global property of the fit, and never to the local density at a query: the
    collected mass IS this method's epistemic signal (it sets the credal radius -log(alpha)/mass),
    so normalising it away per query would give every instance the same width and delete the method.

    Args:
        features: training features, shape (N, D).
        targets: integer training labels, shape (N,).
        ridge: Tikhonov ridge on the covariance, as a fraction of its mean diagonal.
        num_classes: number of classes K.
        whitening: metric exponent in [0, 1]; see above.
        bandwidth_anchor: which self-excluded neighbour rank the bandwidth base is measured at.
            1 (the default) is the legacy behaviour and keeps every existing artifact bitwise
            reproducible; pass num_neighbors to make the scale track the kernel's own window.
            Clamped to the reference size, so a tiny reference cannot ask for a rank it lacks.

    Returns:
        whitener of shape (D, D) mapping normalised features to the whitened space, the projected
        reference of shape (N, D), and the scalar bandwidth base (median self-excluded distance at
        the anchor rank).

    Raises:
        ValueError: fewer than two reference points, whitening outside [0, 1], or a non-positive
            bandwidth_anchor.
    """
    if features.shape[0] < 2:
        raise ValueError(f"Need at least two reference points, got {features.shape[0]}.")
    if not 0.0 <= whitening <= 1.0:
        raise ValueError(f"whitening must be in [0, 1], got {whitening!r}.")
    if bandwidth_anchor < 1:
        raise ValueError(f"bandwidth_anchor must be at least 1, got {bandwidth_anchor!r}.")
    feats = features.double()
    normalised = feats / feats.norm(dim=1, keepdim=True)
    means = torch.stack([normalised[targets == c].mean(0) for c in range(num_classes)])  # (K, D)
    dim = feats.shape[1]
    pooled = torch.zeros(dim, dim, dtype=torch.float64, device=feats.device)
    for c in range(num_classes):
        centred = normalised[targets == c] - means[c]
        pooled = pooled + centred.T @ centred
    pooled = pooled / feats.shape[0]
    ridged = pooled + ridge * pooled.diag().mean() * torch.eye(dim, dtype=torch.float64, device=feats.device)
    if whitening == 1.0:
        # Verbatim legacy path (invert, then eigh): keeps existing full-whitening artifacts and
        # results bitwise reproducible.
        cov_inv = torch.linalg.inv(ridged)
        evals, evecs = torch.linalg.eigh(cov_inv)
        whitener = evecs * evals.clamp_min(0.0).sqrt()  # (D, D); x @ whitener has Mahalanobis geometry
    else:
        lam, vecs = torch.linalg.eigh(ridged)
        whitener = vecs * lam.clamp_min(1e-12) ** (-whitening / 2.0)  # exponent 0: rotation only
    reference = (normalised @ whitener).float()

    # rank + 1 columns because the self-distance of 0 always occupies the first; the last column is
    # then the anchor-th OTHER point. Clamped so a reference smaller than the anchor still works.
    anchor_k = min(bandwidth_anchor + 1, reference.shape[0])
    nearest_other = []
    for start in range(0, reference.shape[0], 4096):
        distances = torch.cdist(reference[start : start + 4096], reference)
        nearest_other.append(distances.topk(anchor_k, largest=False).values[:, -1])
    bandwidth_base = torch.cat(nearest_other).median()
    return whitener.float(), reference, bandwidth_base


# Auto-mode whitening selection: descend the grid and keep the largest exponent whose leave-one-out
# vote is healthy. The margins are guardrail constants of the rule, not tuning knobs: the share
# margin tolerates mild skew above the true majority share, and the accuracy margin ties health to
# the exponent-zero vote, which is free of whitening-induced hubs by construction.
WHITENING_GRID: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)
_WHITENING_SHARE_MARGIN = 0.10
_WHITENING_ACC_MARGIN = 0.02


def loo_vote_health(
    reference: torch.Tensor,
    reference_targets: torch.Tensor,
    bandwidth: float,
    num_neighbors: int,
    num_classes: int,
) -> tuple[float, float]:
    """Leave-one-out kernel-vote accuracy and maximum predicted class share on the reference.

    The hub-collapse check: a healthy reference votes its own labels back at roughly the encoder's
    accuracy with class shares near the data's, while a hub-collapsed one votes one class nearly
    everywhere (dermamnist and covid_xray failed exactly this way at full whitening, with the
    collapsed accuracy pinned at the majority share).

    Args:
        reference: projected reference features, shape (N, D).
        reference_targets: integer reference labels, shape (N,).
        bandwidth: kernel bandwidth in the reference's metric.
        num_neighbors: number of nearest reference points the kernel is evaluated on.
        num_classes: number of classes K.

    Returns:
        Leave-one-out vote accuracy and the maximum class share among the predicted labels.
    """
    ref = reference.float()
    n = ref.shape[0]
    k = min(num_neighbors + 1, n)
    counts = torch.zeros(n, num_classes, dtype=torch.float64, device=ref.device)
    for start in range(0, n, 2048):
        distances, indices = torch.cdist(ref[start : start + 2048], ref).topk(k, largest=False)
        distances, indices = distances[:, 1:], indices[:, 1:]  # drop self
        weights = torch.exp(-(distances.double() ** 2) / (2.0 * bandwidth * bandwidth))
        counts[start : start + 2048].scatter_add_(1, reference_targets[indices].long(), weights)
    predictions = counts.argmax(dim=1)
    accuracy = float((predictions == reference_targets).float().mean())
    share = float(torch.bincount(predictions, minlength=num_classes).float().max() / n)
    return accuracy, share


def select_whitening(
    features: torch.Tensor,
    targets: torch.Tensor,
    ridge: float,
    num_classes: int,
    bandwidth_mult: float,
    num_neighbors: int,
    bandwidth_anchor: int = 1,
) -> tuple[float, dict[float, tuple[float, float]]]:
    """Largest exponent on WHITENING_GRID whose leave-one-out vote on the reference is healthy.

    Health at an exponent: maximum vote share at most the true majority share plus the share
    margin, and accuracy at least the exponent-zero accuracy minus the accuracy margin. Exponent
    zero passes its own bar, so the selection is total. Deterministic given the features; the cost
    is one reference fit and one leave-one-out pass per grid point. On the measured grids this
    keeps full whitening on cifar10 and backs dermamnist off to 0.25 to 0.5.

    Args:
        features: training features, shape (N, D).
        targets: integer training labels, shape (N,).
        ridge: Tikhonov ridge on the covariance, as a fraction of its mean diagonal.
        num_classes: number of classes K.
        bandwidth_mult: kernel bandwidth as a multiple of each candidate metric's bandwidth base.
        num_neighbors: number of nearest reference points the kernel is evaluated on.
        bandwidth_anchor: neighbour rank the bandwidth base is measured at; must match what the
            final fit uses, or the health check would score a different kernel than the one built.

    Returns:
        The chosen exponent and the per-exponent (accuracy, max share) statistics for logging.
    """
    stats: dict[float, tuple[float, float]] = {}
    for gamma in WHITENING_GRID:
        _, reference, base = fit_evidence_reference(
            features, targets, ridge, num_classes, whitening=gamma, bandwidth_anchor=bandwidth_anchor
        )
        stats[gamma] = loo_vote_health(reference, targets, float(bandwidth_mult * base), num_neighbors, num_classes)
    true_share = float(torch.bincount(targets, minlength=num_classes).float().max() / targets.numel())
    accuracy_floor = stats[0.0][0] - _WHITENING_ACC_MARGIN
    for gamma in WHITENING_GRID:
        accuracy, share = stats[gamma]
        if share <= true_share + _WHITENING_SHARE_MARGIN and accuracy >= accuracy_floor:
            return gamma, stats
    return 0.0, stats


@torch.compiler.disable
def compute_evidence(
    features: torch.Tensor,
    whitener: torch.Tensor,
    reference: torch.Tensor,
    reference_targets: torch.Tensor | None,
    bandwidth: float,
    num_neighbors: int,
    num_classes: int,
    reference_class_masses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel-weighted class counts and evidence mass of test features against the reference.

    Each of the num_neighbors nearest reference points contributes a Gaussian kernel weight
    exp(-d^2 / (2 bandwidth^2)); the truncation is numerically immaterial because the kernel tail
    beyond the nearest few hundred points is zero to machine precision. With reference_targets the
    weight goes to the point's own class (the full-reference form); with reference_class_masses
    each reference point is a landmark carrying a soft class histogram, and the weight is spread
    over it (the compressed form; a one-hot mass matrix reproduces the full form exactly). The
    evidence mass, the total weight collected, is the alpha-free epistemic statistic of the
    method: the effective local sample size, whose reciprocal scales the credal set's radius.
    Computed and returned in float64.

    Args:
        features: test features, shape (T, D).
        whitener: whitening map from fit_evidence_reference, shape (D, D).
        reference: projected reference features, shape (N, D).
        reference_targets: integer reference labels, shape (N,), or None when
            reference_class_masses is given.
        bandwidth: kernel bandwidth in the whitened metric.
        num_neighbors: number of nearest reference points the kernel is evaluated on.
        num_classes: number of classes K.
        reference_class_masses: per-reference-point class masses, shape (N, K); overrides
            reference_targets when given.

    Returns:
        counts of shape (T, K) and mass of shape (T,), with mass equal to counts summed over
        classes.

    Raises:
        ValueError: neither reference_targets nor reference_class_masses is given.
    """
    if reference_targets is None and reference_class_masses is None:
        raise ValueError("compute_evidence needs reference_targets or reference_class_masses.")
    feats = features.double()
    normalised = (feats / feats.norm(dim=1, keepdim=True)) @ whitener.double()
    queries = normalised.float()
    reference = reference.float()  # the stored buffer may be float16
    k = min(num_neighbors, reference.shape[0])
    counts = torch.zeros(features.shape[0], num_classes, dtype=torch.float64, device=features.device)
    for start in range(0, queries.shape[0], 2048):
        distances, indices = torch.cdist(queries[start : start + 2048], reference).topk(k, largest=False)
        weights = torch.exp(-(distances.double() ** 2) / (2.0 * bandwidth * bandwidth))  # (B, k)
        if reference_class_masses is not None:
            counts[start : start + 2048] = torch.einsum("bm,bmk->bk", weights, reference_class_masses[indices].double())
        else:
            counts[start : start + 2048].scatter_add_(1, reference_targets[indices].long(), weights)  # ty: ignore[not-subscriptable]
    return counts, counts.sum(dim=1)


@torch.compiler.disable
def compute_multinomial_bounds(
    counts: torch.Tensor,
    mass: torch.Tensor,
    alpha: float,
    num_bisection_steps: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class probability bounds of the alpha-cut of the local multinomial relative likelihood.

    The weighted labels around a test point form a multinomial sample about the conditional class
    distribution; the credal set is the alpha-cut of its relative likelihood, which equals the
    reverse Kullback-Leibler ball around the kernel vote with radius -log(alpha) divided by the
    evidence mass. The stored per-class bounds are its profile-likelihood projections: profiling
    the remaining coordinates collapses the relative likelihood of class c to a binomial one in
    (counts_c, mass), whose alpha-cut endpoints are found by bisection on each monotone side of
    the vote. Points with vanishing mass have a flat likelihood and get the vacuous unit
    interval. alpha at most zero returns the vacuous box, alpha at least one the point mass at
    the vote.

    Args:
        counts: kernel-weighted class counts, shape (T, K).
        mass: evidence mass, shape (T,).
        alpha: relative-likelihood level in [0, 1].
        num_bisection_steps: bisection iterations per bound; 50 gives float64-level endpoints.

    Returns:
        Lower and upper probability bounds, each shape (T, K), in float64.
    """
    vote = (counts / mass.clamp_min(1e-12).unsqueeze(1)).clamp(0.0, 1.0)  # (T, K)
    if alpha <= 0.0:
        return torch.zeros_like(vote), torch.ones_like(vote)
    if alpha >= 1.0:
        return vote.clone(), vote.clone()

    log_alpha = math.log(alpha)
    eps = 1e-12
    rest = mass.unsqueeze(1) - counts  # (T, K) weight on the other classes

    def log_profile_rl(p: torch.Tensor) -> torch.Tensor:
        own = torch.where(
            counts > eps, counts * (p.clamp_min(eps).log() - vote.clamp_min(eps).log()), torch.zeros_like(p)
        )
        other = torch.where(
            rest > eps,
            rest * ((1.0 - p).clamp_min(eps).log() - (1.0 - vote).clamp_min(eps).log()),
            torch.zeros_like(p),
        )
        return own + other

    lower_lo, lower_hi = torch.zeros_like(vote), vote.clone()
    for _ in range(num_bisection_steps):  # log_profile_rl is increasing on [0, vote]
        mid = (lower_lo + lower_hi) / 2.0
        inside = log_profile_rl(mid) >= log_alpha
        lower_hi = torch.where(inside, mid, lower_hi)
        lower_lo = torch.where(inside, lower_lo, mid)
    upper_lo, upper_hi = vote.clone(), torch.ones_like(vote)
    for _ in range(num_bisection_steps):  # and decreasing on [vote, 1]
        mid = (upper_lo + upper_hi) / 2.0
        inside = log_profile_rl(mid) >= log_alpha
        upper_lo = torch.where(inside, mid, upper_lo)
        upper_hi = torch.where(inside, upper_hi, mid)

    vacuous = (mass <= 1e-6).unsqueeze(1).expand_as(vote)
    lower = torch.where(vacuous, torch.zeros_like(vote), lower_hi)
    upper = torch.where(vacuous, torch.ones_like(vote), upper_lo)
    return lower, upper


class CredalRLMultinomialPredictor(nn.Module):
    """Encoder plus buffers defining the local multinomial RL credal set; see compute_multinomial_bounds."""

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        *,
        ridge: float = 1e-2,
        bandwidth_mult: float = 0.55,
        num_neighbors: int = 200,
        whitening: float | str = "auto",
        bandwidth_anchor: int | str = 1,
    ) -> None:
        """Wrap an encoder; the reference and scales are filled by the training routine.

        Args:
            encoder: feature encoder (head-less; the method never uses a classification head).
            num_classes: number of classes K; kept explicitly because a small reference may miss
                classes entirely, in which case they simply collect zero evidence.
            ridge: Tikhonov ridge on the whitening covariance, as a fraction of its mean diagonal.
            bandwidth_mult: kernel bandwidth as a multiple of the median self-excluded nearest
                neighbour distance of the reference.
            num_neighbors: number of nearest reference points the kernel is evaluated on.
            bandwidth_anchor: neighbour rank the bandwidth base is measured at, or the string
                "neighborhood" for num_neighbors. 1 (the default) is the legacy scale and keeps
                existing artifacts bitwise reproducible; "neighborhood" makes the bandwidth track
                the window the kernel actually sums over, which is what keeps the evidence mass
                meaningful when the reference size changes (see fit_evidence_reference).
            whitening: metric exponent in [0, 1], or the string auto to let the trainer select the
                largest exponent whose leave-one-out vote is healthy (see select_whitening). The
                exponent actually used is stored in the whitening_gamma buffer at fit time.

        Raises:
            ValueError: num_classes, ridge, bandwidth_mult or num_neighbors is not positive, or
                whitening is neither auto nor a number in [0, 1].
        """
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes!r}.")
        if ridge <= 0:
            raise ValueError(f"ridge must be positive, got {ridge!r}.")
        if bandwidth_mult <= 0:
            raise ValueError(f"bandwidth_mult must be positive, got {bandwidth_mult!r}.")
        if num_neighbors <= 0:
            raise ValueError(f"num_neighbors must be positive, got {num_neighbors!r}.")
        if bandwidth_anchor != "neighborhood" and not (isinstance(bandwidth_anchor, int) and bandwidth_anchor >= 1):
            raise ValueError(f"bandwidth_anchor must be 'neighborhood' or an integer >= 1, got {bandwidth_anchor!r}.")
        if whitening != "auto" and not (isinstance(whitening, int | float) and 0.0 <= float(whitening) <= 1.0):
            raise ValueError(f"whitening must be 'auto' or a number in [0, 1], got {whitening!r}.")
        self.encoder = encoder
        self.num_classes = num_classes
        self.ridge = ridge
        self.bandwidth_mult = bandwidth_mult
        self.num_neighbors = num_neighbors
        # Resolved here rather than at fit time so the stored model records the rank it was built
        # with, not the string that asked for it.
        self.bandwidth_anchor = num_neighbors if bandwidth_anchor == "neighborhood" else int(bandwidth_anchor)
        self.whitening = whitening
        self.register_buffer("whitener", None)
        self.register_buffer("reference", None)
        self.register_buffer("reference_targets", None)
        self.register_buffer("reference_class_masses", None)
        self.register_buffer("bandwidth", None)
        self.register_buffer("alpha", None)
        self.register_buffer("whitening_gamma", None)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> Any:  # noqa: ANN401, FBT001, FBT002
        """Promote None-initialised buffers from the saved tensors before super-load.

        Artifacts trained before the whitening parameter carry no whitening_gamma buffer; they
        load unchanged (the buffer stays None) and behave as the exponent-one fits they are.
        """
        for name in (
            "whitener",
            "reference",
            "reference_targets",
            "reference_class_masses",
            "bandwidth",
            "alpha",
            "whitening_gamma",
        ):
            if name in state_dict and isinstance(state_dict[name], torch.Tensor):
                self.register_buffer(name, state_dict[name].clone())
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def evidence(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Kernel-weighted counts and evidence mass of encoded features against the reference.

        The mass is the method's alpha-free epistemic statistic (the effective local sample size);
        ood_detection scores component=leverage with its negative.

        Args:
            features: encoder outputs, shape (T, D).

        Returns:
            counts of shape (T, K) and mass of shape (T,), in float64.
        """
        return compute_evidence(
            features,
            cast("torch.Tensor", self.whitener),
            cast("torch.Tensor", self.reference),
            cast("torch.Tensor | None", self.reference_targets),
            float(cast("torch.Tensor", self.bandwidth).item()),
            self.num_neighbors,
            self.num_classes,
            reference_class_masses=cast("torch.Tensor | None", self.reference_class_masses),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log probabilities of the kernel vote, the local maximum likelihood estimate.

        Returned on the log scale so downstream metric code that treats forward outputs as logits
        (cross entropy, argmax accuracy) computes the correct quantities: softmax of the log vote
        is the vote itself.
        """
        features = cast("torch.Tensor", self.encoder(x)).detach()
        counts, mass = self.evidence(features)
        vote = counts / mass.clamp_min(1e-12).unsqueeze(1)
        return vote.clamp_min(1e-12).log().to(features.dtype)

    @property
    def mle_member(self) -> nn.Module:
        """Log-vote point predictor, so pipeline point metrics grade the local MLE.

        Without it, evaluate() falls through to the representer and reports the accuracy of the
        credal set's barycenter, which widens with the set and is therefore alpha-dependent; the
        vote is the method's alpha-free point prediction, comparable to the anchor softmax the
        sibling methods expose.
        """
        return _CredalRLMultinomialVoteMember(self)


class _CredalRLMultinomialVoteMember(nn.Module):
    """Log-vote view of the predictor; logit classifier shape for representer dispatch."""

    def __init__(self, predictor: CredalRLMultinomialPredictor) -> None:
        """Hold the predictor without registering it as a submodule (evaluation-only view)."""
        super().__init__()
        object.__setattr__(self, "_predictor", predictor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log probabilities of the kernel vote (softmax-compatible logits)."""
        return cast("CredalRLMultinomialPredictor", self._predictor)(x)


# Register so representer dispatch treats the vote member as a logit classifier and returns
# CategoricalDistribution(softmax(log vote)) = the vote itself.
LogitDistributionPredictor.register(_CredalRLMultinomialVoteMember)


@representer.register(CredalRLMultinomialPredictor)
class CredalRLMultinomialRepresenter[**In, Out, C: ProbabilityIntervalsCredalSet](Representer[Any, In, Out, C]):
    """Per-batch box credal set from the profile bounds of the local multinomial RL alpha-cut."""

    predictor: CredalRLMultinomialPredictor

    def __init__(self, predictor: CredalRLMultinomialPredictor) -> None:
        """Initialize with a local multinomial RL credal predictor."""
        super().__init__(predictor)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> C:
        """Run the encoder, collect local evidence, return the profile-bound probability box."""
        p = self.predictor
        if (
            p.whitener is None
            or p.reference is None
            or (p.reference_targets is None and p.reference_class_masses is None)
            or p.bandwidth is None
            or p.alpha is None
        ):
            msg = (
                "CredalRLMultinomialPredictor has uninitialised buffers; train via "
                "train.py method=credal_rl_multinomial recipe=... before requesting a representation."
            )
            raise RuntimeError(msg)
        if len(args) != 1 or kwargs:
            msg = "CredalRLMultinomialRepresenter.represent expects a single positional tensor argument."
            raise TypeError(msg)
        x: torch.Tensor = args[0]  # ty: ignore[invalid-assignment]

        features = cast("torch.Tensor", p.encoder(x)).detach()
        counts, mass = p.evidence(features)
        lower, upper = compute_multinomial_bounds(counts, mass, float(cast("torch.Tensor", p.alpha).item()))
        return TorchProbabilityIntervalsCredalSet(lower.float().clamp(0, 1), upper.float().clamp(0, 1))  # ty: ignore[invalid-return-type]
