"""Decision rules: reduce a representer output to a per-instance distribution.

A decision rule turns a representation rep_out (categorical, probability intervals, or
convex hull credal set) into a (B, K) tensor of class probabilities for downstream loss
evaluation. Rule / representation compatibility is enforced at apply time so nonsensical
combinations fail loudly instead of silently producing a misleading number.

Terminology follows Berger (1985, Statistical Decision Theory and Bayesian Analysis): a
decision rule is any map from observations to acts. The imprecise-probability literature
uses 'decision criterion' more narrowly for Gamma-maximin / Gamma-maximax / maximality /
E-admissibility (Troffaes 2007, IJAR 45(1); Augustin et al. 2014, Introduction to Imprecise
Probabilities, ch. 8).
"""

from __future__ import annotations

from typing import Any

import torch
from probly.decider import categorical_from_mean
from probly.quantification.measure.credal_set import upper_entropy
from probly.representation.credal_set.torch import TorchConvexCredalSet, TorchProbabilityIntervalsCredalSet
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution
from tqdm import tqdm

SUPPORTED_DECISION_RULES = ("minimax", "barycenter", "mle", "cvar_minimax")


def _maximin_one_hot(probs_or_lower: torch.Tensor) -> torch.Tensor:
    """One-hot argmax along the last axis, cast to the input's dtype.

    Gamma-minimax decision under 0-1 loss: when the input is a credal set lower bound,
    argmax_y min_p p(y) = argmax of the lower bounds. When the input is a plain predictive
    distribution, this reduces to the standard argmax classifier.
    """
    argmax = probs_or_lower.argmax(dim=-1)
    num_classes = probs_or_lower.shape[-1]
    return torch.nn.functional.one_hot(argmax, num_classes=num_classes).to(probs_or_lower.dtype)


def _minimax_probs(rep_out: Any, loss: str) -> torch.Tensor:  # noqa: ANN401
    """Gamma-minimax distribution for the given per-instance loss on a credal-set rep_out.

    Caller must ensure rep_out is a credal set; apply_decision_rule enforces this.

    - log_loss: the upper-entropy-maximizing distribution in the credal set, obtained from
      probly's upper_entropy(..., return_distribution=True) (water-filling on the box for
      TorchProbabilityIntervalsCredalSet, LBFGS on softmax weights for TorchConvexCredalSet).
    - zero_one: argmax of the per-class lower probability (lower_bounds for intervals,
      per-class min-over-vertices for the hull, since p(y) is linear in p).
    """
    if loss == "log_loss":
        if isinstance(rep_out, TorchProbabilityIntervalsCredalSet | TorchConvexCredalSet):
            _, p = upper_entropy(rep_out, return_distribution=True)
            return p  # ty: ignore[invalid-return-type]
    if loss == "zero_one":
        if isinstance(rep_out, TorchProbabilityIntervalsCredalSet):
            return _maximin_one_hot(rep_out.lower_bounds)
        if isinstance(rep_out, TorchConvexCredalSet):
            return _maximin_one_hot(rep_out.tensor.probabilities.min(dim=-2).values)
    raise ValueError(f"minimax not implemented for loss={loss!r} on rep_out of type {type(rep_out).__name__}.")


def _per_class_loss(p: torch.Tensor, loss: str) -> torch.Tensor:
    """Per-class loss of a prediction: loss(p, y) for every class y, shape (B, K).

    The single torch definition of the per-instance losses. The worst-case truncated loss scores
    every class against the worst credal distribution, and the realised loss indexes the true class,
    so both go through this. It mirrors the numpy losses in metrics.py (a test checks they agree).

    Args:
        p: Predicted distributions, shape (B, K).
        loss: Per-instance loss, one of log_loss or brier.

    Returns:
        The per-class loss, shape (B, K): entry (b, y) is the loss of p[b] if the truth were y.

    Raises:
        ValueError: loss is neither log_loss nor brier.
    """
    if loss == "log_loss":
        # Floor p at the dtype machine epsilon before the log (as metrics.py and train_funcs.py do)
        # so a zero class probability does not send the loss to negative infinity.
        return -torch.log(p.clamp_min(torch.finfo(p.dtype).eps))  # -log p_y for every class y
    if loss == "brier":
        return (p * p).sum(dim=-1, keepdim=True) - 2.0 * p + 1.0  # ||p - e_y||^2 for every class y
    raise ValueError(f"Unsupported loss={loss!r} for cvar_minimax; choose log_loss or brier.")


def _worst_case_truncated_loss(
    rep_out: Any,  # noqa: ANN401
    p: torch.Tensor,
    loss: str,
    v: float | torch.Tensor,
) -> torch.Tensor:
    """Worst-case truncated loss of a prediction over a credal set, per instance.

    For a fixed prediction p this returns the supremum over distributions q in the credal set
    of the mean truncated loss, that is the sup over q of sum_y q(y) times (loss(p, y) - v)
    clamped at zero. This is the inner maximisation of the cvar_minimax objective; the rule then
    minimises this quantity over p. At v equal to zero the truncation is inactive for log loss,
    so this reduces to the worst-case expected loss that the Gamma-minimax rule minimises.

    The supremum is closed form because the objective is linear in q: the worst q puts as much
    mass as the credal set allows on the classes with the largest truncated loss. For a convex
    hull this is attained at a vertex; for probability intervals it is a greedy water-filling on
    the box. It is written with differentiable torch operations, so autograd through it yields the
    Danskin gradient with respect to p.

    Args:
        rep_out: A credal set, either TorchConvexCredalSet or TorchProbabilityIntervalsCredalSet,
            over B instances.
        p: Predicted distributions, shape (B, K), or (V, B, K) with a leading axis of independent
            predictions per instance (the calibration scores its whole threshold grid this way).
        loss: Per-instance loss, one of log_loss or brier.
        v: The truncation threshold, the VaR level: a float, or a tensor broadcastable against
            the per-class loss of p (the calibration passes shape (V, 1, 1)).

    Returns:
        The per-instance worst-case truncated loss, shape (B,) or (V, B).

    Raises:
        ValueError: loss is neither log_loss nor brier.
        TypeError: rep_out is neither supported credal-set type.
    """
    # NOTE: src/cost_sensitive/rules.py:worst_case_action_cost holds a second copy of the
    # water-filling below, generalised to an action axis. The two were deliberately not merged:
    # this function is core numerics shared with other in-flight experiments and the repo has no
    # test suite to catch a regression here. Keep them in sync by hand if the algorithm changes.

    # The per-class loss for every y, then keep only the part above the threshold v.
    excess = (_per_class_loss(p, loss) - v).clamp_min(0.0)  # (..., B, K) loss in excess of the threshold v

    # Worst distribution q: maximise sum_y q(y) excess_y over the credal set. The einsum ellipsis
    # and the expand before the gather let a leading axis on excess broadcast over the credal
    # set's batch axis; without one they are the plain (B, K) computation.
    if isinstance(rep_out, TorchConvexCredalSet):
        vertices = rep_out.tensor.probabilities  # (B, M, K)
        return torch.einsum("bmk,...bk->...bm", vertices, excess).amax(dim=-1)  # value at the best vertex
    if isinstance(rep_out, TorchProbabilityIntervalsCredalSet):
        lower: torch.Tensor = rep_out.lower_bounds  # (B, K)
        upper: torch.Tensor = rep_out.upper_bounds  # (B, K)
        # Water-filling: start every class at its lower bound, then pour the remaining mass onto
        # the largest-excess classes first, each up to its upper bound.
        leftover = (1.0 - lower.sum(dim=-1, keepdim=True)).clamp_min(0.0)  # (B, 1)
        slack = (upper - lower).clamp_min(0.0)  # (B, K) headroom above the lower bound
        excess_sorted, idx = excess.sort(dim=-1, descending=True)
        slack_sorted = torch.gather(slack.expand(excess.shape), dim=-1, index=idx)
        cumulative_before = slack_sorted.cumsum(dim=-1) - slack_sorted  # mass used by larger-excess classes
        take = (leftover - cumulative_before).clamp_min(0.0)
        used = torch.minimum(slack_sorted, take)  # mass actually placed on each class
        return (lower * excess).sum(dim=-1) + (used * excess_sorted).sum(dim=-1)  # (..., B)
    raise TypeError(f"Unsupported rep_out type for cvar_minimax: {type(rep_out).__name__}.")


def _device_type(rep_out: Any) -> str:  # noqa: ANN401
    """Device type of a credal set's tensors (cuda, cpu, mps), as torch.autocast expects it.

    Args:
        rep_out: A credal set, either TorchConvexCredalSet or TorchProbabilityIntervalsCredalSet.

    Returns:
        The device type string.
    """
    if isinstance(rep_out, TorchProbabilityIntervalsCredalSet):
        return rep_out.lower_bounds.device.type
    return rep_out.tensor.probabilities.device.type


def cvar_minimax_probs(
    rep_out: Any,  # noqa: ANN401
    loss: str,
    v: float | torch.Tensor,
    num_iters: int = 300,
    lr: float = 0.1,
    p_start: torch.Tensor | None = None,
) -> torch.Tensor:
    """The cvar_minimax prediction: the distribution minimising the worst-case truncated loss.

    Solves the argmin over p in the simplex of _worst_case_truncated_loss(rep_out, p, loss, v).
    The prediction is parameterised as p = softmax(z) so it always stays a valid distribution, and
    z is optimised with Adam, using autograd through the closed-form inner supremum for the
    gradient. It warm-starts at the maxent element (the v=0 solution). The objective is convex in
    p, so this converges to the global optimum; correctness is checked separately against an exact
    oracle in the tests. The solve runs with autocast disabled, so an enclosing amp region
    (shift_risk_metric applies the rule inside one) cannot drop it to half precision; rep_out must
    itself be full precision, which _as_credal_set guarantees on every path into the solver.

    Args:
        rep_out: A credal set, either TorchConvexCredalSet or TorchProbabilityIntervalsCredalSet.
        loss: Per-instance loss, one of log_loss or brier.
        v: The truncation threshold, the VaR level: a float, or a tensor broadcastable against
            the per-class loss of the iterates (see _worst_case_truncated_loss).
        num_iters: Number of Adam iterations.
        lr: Adam learning rate on the logits.
        p_start: Optional warm start; None computes the maxent element. The solver state takes
            its shape, so a leading axis on p_start (with a matching axis on v) runs independent
            solves per entry against the same credal set. Adam updates and every solver operation
            act per instance, so batching solves this way leaves each solution unchanged.

    Returns:
        The per-instance prediction p, shaped like p_start ((B, K) by default).
    """
    # Autocast off for the whole solve, the maxent warm start included: under an enclosing amp region
    # the inner supremum's einsum and the L-BFGS maxent would otherwise run in half precision.
    with torch.autocast(_device_type(rep_out), enabled=False):
        if p_start is None:
            p_start = _minimax_probs(rep_out, "log_loss")  # warm start at the maxent / v=0 solution

        # enable_grad overrides any outer torch.no_grad() context (the eval loop sets one); the solver
        # optimises z by backprop, so it needs autograd even though prediction runs under no_grad.
        with torch.enable_grad():
            z = p_start.clamp_min(torch.finfo(p_start.dtype).eps).log().detach().requires_grad_(True)

            # A fixed Adam step size only settles into a neighbourhood of the optimum, and where several
            # credal distributions tie as the worst case the gradient does not vanish there, so the last
            # iterate can drift away. Keep the best prediction seen (the lowest worst-case loss) per instance
            # and return that instead of the last iterate.
            best_p = z.softmax(dim=-1).detach()
            best_worst_case = _worst_case_truncated_loss(rep_out, best_p, loss, v).detach()  # (B,)

            optimizer = torch.optim.Adam([z], lr=lr)
            for _ in range(num_iters):
                optimizer.zero_grad()
                p = z.softmax(dim=-1)
                worst_case = _worst_case_truncated_loss(rep_out, p, loss, v)  # (B,)
                worst_case.sum().backward()  # each instance contributes its own gradient
                optimizer.step()
                with torch.no_grad():
                    improved = worst_case.detach() < best_worst_case  # (B,)
                    best_worst_case = torch.where(improved, worst_case.detach(), best_worst_case)
                    best_p = torch.where(improved.unsqueeze(-1), p.detach(), best_p)
    return best_p


def _realized_loss(p: torch.Tensor, loss: str, targets: torch.Tensor) -> torch.Tensor:
    """Loss of a prediction on the true label, per instance.

    Unlike _worst_case_truncated_loss, which scores every class against the worst credal
    distribution, this scores only the realised outcome y_i. Used by the v calibration, which
    evaluates the rule's predictions against the true validation labels.

    Args:
        p: Predicted distributions, shape (B, K), or (V, B, K) with a leading axis of independent
            predictions per instance.
        loss: Per-instance loss, one of log_loss or brier.
        targets: True labels, shape (B,); a leading axis on p scores every entry against them.

    Returns:
        The per-instance realised loss, shape (B,) or (V, B).

    Raises:
        ValueError: loss is neither log_loss nor brier.
    """
    per_class = _per_class_loss(p, loss)
    return per_class.gather(-1, targets.expand(per_class.shape[:-1]).unsqueeze(-1)).squeeze(
        -1
    )  # loss at the true class


def singleton_credal_set(probs: torch.Tensor) -> TorchConvexCredalSet:
    """Wrap a point prediction as the degenerate credal set containing only that distribution.

    Lets a point predictor be scored under cvar_minimax, the ablation that separates the
    contribution of the credal set from that of the decision rule: the credal methods get both,
    so a win over the risk-neutral baselines is only attributable to the set once the same rule
    on a point prediction has been measured. This is not new math, since the supremum over a
    one-point set is the value at that point, and the rule collapses to minimising the truncated
    expected loss under the model's own belief. Building it as a one-vertex hull rather than as a
    separate truncated-expectation function keeps the ablation and the credal arm on a single
    code path so they cannot drift apart. The same construction is used by the cost-sensitive
    experiment on its branch, in cost_sensitive.rules.singleton_credal_set.

    Args:
        probs: Predicted distributions, shape (B, K), each row a distribution over the K labels.

    Returns:
        A TorchConvexCredalSet with one vertex per instance, shape (B, 1, K).

    Raises:
        ValueError: probs is not a 2-D (B, K) tensor.
    """
    if probs.ndim != 2:
        raise ValueError(
            f"singleton_credal_set expects a (B, K) tensor of distributions, got shape {tuple(probs.shape)}."
        )
    return TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(probs.unsqueeze(1)))


# Dtypes the cvar_minimax solver must never run in; _as_credal_set lifts them to float32.
_HALF_PRECISION = (torch.float16, torch.bfloat16)


def _full_precision(distribution: Any) -> Any:  # noqa: ANN401
    """Rebuild a half-precision categorical distribution in float32; return any other one unchanged.

    The rebuild keeps the parameterisation, so a logit distribution gets its probabilities from the
    float32 logits and matches the same logits given in float32 bit for bit.

    Args:
        distribution: A torch categorical distribution, logit or probability parameterised.

    Returns:
        The distribution itself, or its float32 copy when its parameter tensor is float16 or bfloat16.
    """
    if distribution.tensor.dtype in _HALF_PRECISION:
        return type(distribution)(distribution.tensor.float())
    return distribution


def _as_credal_set(rep_out: Any) -> Any:  # noqa: ANN401
    """Return rep_out as a full-precision credal set, wrapping a categorical as a singleton.

    The single place the point-predictor arm enters the cvar_minimax path, so calibration and
    prediction treat it identically. It is also where half precision is removed. Under amp the base
    model's logits arrive as float16, and a float16 solve floors probabilities at float16's eps (9.8e-4):
    that caps every log loss at 6.93 and leaves the classes below the floor without a gradient, and
    probly's L-BFGS maxent can overflow outright. Float32 and float64 inputs pass through untouched.

    Args:
        rep_out: Output of rep.predict(...), a credal set or a plain categorical.

    Returns:
        A credal set in at least float32: rep_out itself, its float32 copy, or its distribution
        wrapped by singleton_credal_set.
    """
    if isinstance(rep_out, TorchProbabilityIntervalsCredalSet):
        if rep_out.lower_bounds.dtype in _HALF_PRECISION:
            return TorchProbabilityIntervalsCredalSet(
                lower_bounds=rep_out.lower_bounds.float(), upper_bounds=rep_out.upper_bounds.float()
            )
        return rep_out
    if isinstance(rep_out, TorchConvexCredalSet):
        vertices = _full_precision(rep_out.tensor)
        return rep_out if vertices is rep_out.tensor else TorchConvexCredalSet(tensor=vertices)
    return singleton_credal_set(_full_precision(categorical_from_mean(rep_out)).probabilities)


def _concat_credal_sets(rep_outs: list[Any]) -> Any:  # noqa: ANN401
    """Concatenate per-batch credal sets along the instance axis into one credal set.

    Args:
        rep_outs: Non-empty list of same-type credal sets, TorchConvexCredalSet (with equal
            vertex counts) or TorchProbabilityIntervalsCredalSet.

    Returns:
        One credal set holding all instances in order.

    Raises:
        TypeError: rep_outs holds neither supported credal-set type.
    """
    first = rep_outs[0]
    if len(rep_outs) == 1:
        return first
    if isinstance(first, TorchConvexCredalSet):
        vertices = torch.cat([r.tensor.probabilities for r in rep_outs])  # (N, M, K)
        return TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))
    if isinstance(first, TorchProbabilityIntervalsCredalSet):
        return TorchProbabilityIntervalsCredalSet(
            lower_bounds=torch.cat([r.lower_bounds for r in rep_outs]),
            upper_bounds=torch.cat([r.upper_bounds for r in rep_outs]),
        )
    raise TypeError(f"Unsupported rep_out type for cvar_minimax: {type(first).__name__}.")


# Cap on thresholds * num_instances * num_classes per batched solver call: bounds the solver's
# peak memory (intermediates at this cap total on the order of a GB) while typically still
# fitting a whole calibration grid or beta sweep into a single solve.
_CALIBRATION_MAX_ELEMENTS = 2**24


def maxent_predictions(rep_outs: list[Any]) -> torch.Tensor:  # noqa: ANN401
    """Maxent (v=0) cvar_minimax predictions of per-batch credal sets, concatenated.

    Computed per batch rather than on the concatenation: for hull sets the upper-entropy solve
    couples instances within a batch through its LBFGS line search, so per-batch evaluation is
    what reproduces the per-batch solver path exactly. Serves as the warm start (and, through
    its realised losses, the threshold range) for cvar_minimax_at.

    Args:
        rep_outs: Per-batch credal sets. Plain categoricals are accepted and wrapped as
            one-vertex credal sets (the point-predictor ablation).

    Returns:
        The maxent predictions, shape (N, K) over all batches in order.
    """
    return torch.cat([_minimax_probs(_as_credal_set(r), "log_loss") for r in rep_outs])


def cvar_minimax_at(
    rep_outs: list[Any],  # noqa: ANN401
    vs: list[float],
    loss: str,
    p_maxent: torch.Tensor,
    desc: str = "cvar_minimax solves",
) -> torch.Tensor:
    """cvar_minimax predictions of the same instances at several thresholds, in one batched solve.

    The thresholds are stacked on a leading axis and solved together, chunked to bound memory.
    Adam updates and every solver operation act per instance, so each (threshold, instance)
    solution is identical to a separate per-batch call at that scalar threshold (see
    cvar_minimax_probs on batching); this is the single implementation behind both threshold
    calibration and multi-beta shift evaluation.

    Args:
        rep_outs: Per-batch credal sets (categoricals wrapped as one-vertex sets).
        vs: Thresholds to solve at.
        loss: Per-instance loss, one of log_loss or brier.
        p_maxent: Warm start from maxent_predictions(rep_outs), shape (N, K).
        desc: Progress-bar label for the chunk loop.

    Returns:
        Predictions of shape (len(vs), N, K).
    """
    rep_all = _concat_credal_sets([_as_credal_set(r) for r in rep_outs])
    chunk = max(1, min(len(vs), _CALIBRATION_MAX_ELEMENTS // max(1, p_maxent.numel())))
    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(vs), chunk), desc=desc):
        group = vs[start : start + chunk]
        v_t = p_maxent.new_tensor(group).view(-1, 1, 1)  # (C, 1, 1), broadcast against (C, N, K)
        p_start = p_maxent.unsqueeze(0).expand(len(group), *p_maxent.shape)  # (C, N, K)
        chunks.append(cvar_minimax_probs(rep_all, loss, v_t, p_start=p_start))  # (C, N, K)
    return torch.cat(chunks)


def calibrate_var_thresholds(
    rep_outs: list[Any],  # noqa: ANN401
    targets: list[torch.Tensor],
    betas: list[float],
    loss: str,
    num_grid: int = 100,
) -> dict[float, float]:
    """Pick, for each beta, the VaR threshold v minimising the rule's realised CVaR on a validation set.

    Calibration is label-based: it minimises F_beta(v) = v + (1/beta) mean_i (loss(p_i(v), y_i) - v)+,
    the realised CVaR at level beta of the rule's predictions p_i(v) = cvar_minimax_probs(rep, loss,
    v), over a uniform grid of v on [0, v_max]. The objective is not guaranteed convex in v (the
    predictions depend on v through the solver), so a grid is used rather than a local optimiser.

    The grid predictions depend only on v, not on beta, so every beta shares the one batched grid
    solve (cvar_minimax_at) and differs only in the closing argmin over F_beta -- calibrating many
    betas costs the same as calibrating one, and each threshold is identical to a single-beta
    calibration.

    Args:
        rep_outs: Per-batch credal sets collected over the validation set. Plain categoricals are
            accepted too and wrapped as one-vertex credal sets, so the point-predictor ablation
            calibrates v through this identical path.
        targets: Per-batch true labels, aligned with rep_outs.
        betas: CVaR tail levels, each in (0, 1].
        loss: Per-instance loss, one of log_loss or brier.
        num_grid: Number of grid points for v.

    Returns:
        {beta: calibrated threshold v} for every requested beta.
    """
    p_maxent = maxent_predictions(rep_outs)  # (N, K)
    targets_all = torch.cat(targets)  # (N,)
    maxent_losses = _realized_loss(p_maxent, loss, targets_all)  # (N,)
    n = maxent_losses.numel()
    # v_max: the largest realised loss of the maxent (v=0) prediction. Above it the truncation is
    # always zero and F(v) = v only grows, so the optimum lies in [0, v_max].
    v_max = maxent_losses.max().item()

    grid = torch.linspace(0.0, v_max, num_grid).tolist()
    p = cvar_minimax_at(rep_outs, grid, loss, p_maxent, desc="Calibrating v")  # (G, N, K)
    realized = _realized_loss(p, loss, targets_all)  # (G, N)
    v_col = p_maxent.new_tensor(grid).view(-1, 1)  # (G, 1)
    # Excess sums per grid point, in float64 like the sequential python accumulation was.
    excess_sums = (realized - v_col).clamp_min(0.0).double().sum(dim=1).tolist()

    thresholds: dict[float, float] = {}
    for beta in betas:
        objectives = [v + excess / (beta * n) for v, excess in zip(grid, excess_sums, strict=True)]  # F_beta(v)
        # index of the first minimum: the same tie-breaking as the sequential strict-less-than update.
        thresholds[beta] = grid[objectives.index(min(objectives))]
    return thresholds


def calibrate_var_threshold(
    rep_outs: list[Any],  # noqa: ANN401
    targets: list[torch.Tensor],
    beta: float,
    loss: str,
    num_grid: int = 100,
) -> float:
    """Single-beta calibration; see calibrate_var_thresholds.

    Args:
        rep_outs: Per-batch credal sets collected over the validation set.
        targets: Per-batch true labels, aligned with rep_outs.
        beta: CVaR tail level in (0, 1].
        loss: Per-instance loss, one of log_loss or brier.
        num_grid: Number of grid points for v.

    Returns:
        The calibrated threshold v.
    """
    return calibrate_var_thresholds(rep_outs, targets, [beta], loss, num_grid)[beta]


def apply_decision_rule(
    rep_out: Any,  # noqa: ANN401
    loss: str,
    decision_rule: str,
    *,
    var_threshold: float | None = None,
) -> torch.Tensor:
    """Reduce a representation to a per-instance distribution under the chosen decision rule.

    Decision-rule / rep-type compatibility is enforced here so that nonsensical combinations
    fail loudly instead of silently producing a misleading number:

    - minimax / barycenter require a credal-set rep_out (TorchProbabilityIntervalsCredalSet or
      TorchConvexCredalSet). Raises on a plain categorical (e.g. base predictor).
    - mle requires a non-credal rep_out (a plain categorical). Only base produces this;
      every credal method produces a credal-set rep_out and raises. To get the MLE point for
      a credal method's underlying base predictor, evaluate the corresponding base artifact
      directly with decision_rule=mle.
    - cvar_minimax requires a calibrated var_threshold (from calibrate_var_threshold) and accepts
      either a credal-set rep_out or a plain categorical. A categorical is wrapped as the
      one-vertex credal set containing only that distribution (see singleton_credal_set), which
      is the ablation isolating the rule from the credal set. Under zero_one it reduces to
      standard minimax and ignores var_threshold.

    Args:
        rep_out: Output of rep.predict(...).
        loss: Per-instance loss the minimax and cvar_minimax rules target (ignored by others).
        decision_rule: One of SUPPORTED_DECISION_RULES.
        var_threshold: The calibrated VaR threshold v for cvar_minimax; ignored by other rules.

    Returns:
        Shape (B, K) tensor of per-instance distributions.
    """
    is_credal = isinstance(rep_out, TorchProbabilityIntervalsCredalSet | TorchConvexCredalSet)
    if decision_rule == "minimax":
        if not is_credal:
            raise ValueError(
                f"decision_rule=minimax requires a credal-set output but rep produced "
                f"{type(rep_out).__name__}; pick decision_rule=mle for non-credal methods (e.g. base)."
            )
        return _minimax_probs(rep_out, loss)
    if decision_rule == "barycenter":
        if not is_credal:
            raise ValueError(
                f"decision_rule=barycenter requires a credal-set output but rep produced "
                f"{type(rep_out).__name__}; pick decision_rule=mle for non-credal methods (e.g. base)."
            )
        return categorical_from_mean(rep_out).probabilities  # ty: ignore[invalid-return-type]
    if decision_rule == "mle":
        if is_credal:
            raise ValueError(
                f"decision_rule=mle is not defined for credal methods producing {type(rep_out).__name__}; "
                f"use minimax or barycenter, or evaluate the corresponding base artifact directly."
            )
        return categorical_from_mean(rep_out).probabilities  # ty: ignore[invalid-return-type]
    if decision_rule == "cvar_minimax":
        # A point predictor is admitted here, unlike for minimax and barycenter: it becomes the
        # one-vertex credal set containing only its own prediction, so the rule collapses to
        # minimising the truncated expected loss under that belief. This is the ablation arm
        # isolating the rule from the set; see singleton_credal_set.
        rep_out = _as_credal_set(rep_out)
        if loss == "zero_one":
            # The truncated 0-1 minimax reduces to standard Gamma-minimax for any v in [0, 1).
            return _minimax_probs(rep_out, "zero_one")
        if var_threshold is None:
            raise ValueError(
                "decision_rule=cvar_minimax requires var_threshold; calibrate it with "
                "calibrate_var_threshold(...) and pass the result here."
            )
        return cvar_minimax_probs(rep_out, loss, var_threshold)
    raise ValueError(f"Unknown decision_rule={decision_rule!r}. Choose from {list(SUPPORTED_DECISION_RULES)}.")
