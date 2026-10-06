"""Action decision rules for the cost-sensitive experiment.

The repo's src/decision_rules.py reduces a credal set to a *distribution* scored by a
proper loss. Here the decision maker instead picks one of a small number of *actions*,
scored by a cost matrix. Because the action set is finite, the minimisation is an
enumeration over actions rather than a solver on the simplex, and every quantity below
is exact and closed form.

The public surface is the shared primitive worst_case_action_cost, which scores every action
against the worst distribution in the credal set, the three decision rules built on it, and the
calibration that supplies the threshold two of them need.

- worst_case_action_cost: the closed-form inner supremum. Returns a value per instance and
  action rather than a chosen action, because the calibration in a later task needs the
  scores themselves, not just their argmin.
- cvar_minimax_actions (ours): pick the action minimising the worst-case truncated expected
  cost over the credal set at a calibrated threshold v. At v = 0 this is Gamma-minimax on
  the expected cost; on a singleton credal set it is best response on the truncated cost.
- cvar_minimax_point_actions (the ablation): the SAME rule on a point prediction, via
  singleton_credal_set. Separates the contribution of the credal set from that of the rule,
  which the credal arm would otherwise confound.
- best_response_actions: pick the action minimising the expected cost under a point
  prediction. The risk-neutral reference used by the non-credal baselines (base, sqwash,
  adacvar).
- calibrate_action_var_threshold: choose the threshold v that both cvar_minimax rules use, by
  minimising the rule's realized-cost CVaR on a labelled validation set. The ablation arm
  calibrates through the identical call, on singleton credal sets.
- calibrate_action_var_thresholds: the same calibration at SEVERAL tail levels at once, for the
  beta sweep, at the cost of a single-level calibration. See its docstring for why that is exact
  rather than an approximation, and note that the singular function is now a thin wrapper on it,
  so the sweep and a one-off calibration cannot return different answers for the same beta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from probly.representation.credal_set.torch import TorchConvexCredalSet, TorchProbabilityIntervalsCredalSet
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence


def _validate_interval_box(
    rep_out: TorchProbabilityIntervalsCredalSet,
    tol: float = 1e-6,
) -> None:
    """Check that a probability-interval box actually contains a distribution.

    The water-filling assumes the box is reachable: it starts at the lower bounds and pours the
    remaining mass upward. If sum(lower) > 1 the start point is already outside the simplex and the
    result is a supremum over nothing, over-estimating the worst case. If sum(upper) < 1 the fill
    runs out of headroom before reaching total mass 1, and the clamped leftover silently yields an
    UNDER-estimate: the risk-averse rule would then look safer than it is, which is the more
    dangerous of the two directions and the reason this is an exception rather than a clamp.

    Following the repo's fail-loudly posture (apply_decision_rule raises on incompatible rule and
    representation combinations rather than guessing), an unreachable box is an error, not something
    to silently repair.

    Args:
        rep_out: The probability-interval credal set to validate.
        tol: Slack absorbing floating-point error in the mass sums.

    Raises:
        ValueError: sum(lower) exceeds 1 or sum(upper) falls below 1 for at least one instance.
    """
    lower_mass = rep_out.lower_bounds.sum(dim=-1)  # (B,)
    upper_mass = rep_out.upper_bounds.sum(dim=-1)  # (B,)

    oversubscribed = lower_mass > 1.0 + tol
    if bool(oversubscribed.any()):
        count = int(oversubscribed.sum())
        worst = float(lower_mass.max())
        raise ValueError(
            f"Unreachable probability-interval box: sum of lower bounds exceeds 1 for {count} of "
            f"{lower_mass.numel()} instances (largest lower mass {worst:.6g}). The box contains no "
            f"distribution, so the worst-case cost is not defined."
        )

    undersubscribed = upper_mass < 1.0 - tol
    if bool(undersubscribed.any()):
        count = int(undersubscribed.sum())
        worst = float(upper_mass.min())
        raise ValueError(
            f"Unreachable probability-interval box: sum of upper bounds falls below 1 for {count} of "
            f"{upper_mass.numel()} instances (smallest upper mass {worst:.6g}). The box contains no "
            f"distribution, and water-filling would silently under-estimate the worst-case cost."
        )


def worst_case_action_cost(
    rep_out: TorchConvexCredalSet | TorchProbabilityIntervalsCredalSet,
    cost: torch.Tensor,
    v: float,
) -> torch.Tensor:
    """Worst-case truncated expected cost of every action over a credal set, per instance.

    For action a this returns sup over q in the credal set of sum_y q(y) * (cost[y, a] - v)+.
    The objective is linear in q, so the supremum is attained at a hull vertex, or by greedy
    water-filling on a probability-interval box: start every class at its lower bound and pour
    the remaining mass onto the largest-excess classes first, each up to its upper bound.
    This mirrors decision_rules._worst_case_truncated_loss with the per-class loss vector
    replaced by the cost column of each action.

    Args:
        rep_out: A credal set, either TorchConvexCredalSet or TorchProbabilityIntervalsCredalSet.
        cost: Cost matrix, shape (K, A); cost[y, a] is the cost of action a under label y.
        v: The truncation threshold, the VaR level.

    Returns:
        Worst-case truncated cost per instance and action, shape (B, A).

    Raises:
        TypeError: rep_out is neither supported credal-set type.
        ValueError: The probability-interval box is unreachable, so it contains no distribution.
    """
    excess = (cost - v).clamp_min(0.0)  # (K, A) cost in excess of the threshold v

    if isinstance(rep_out, TorchConvexCredalSet):
        vertices = rep_out.tensor.probabilities.to(excess.dtype)  # (B, M, K)
        return torch.einsum("bmk,ka->bma", vertices, excess).amax(dim=1)  # value at the best vertex, (B, A)

    if isinstance(rep_out, TorchProbabilityIntervalsCredalSet):
        _validate_interval_box(rep_out)

        # The water-filling itself is inherited from decision_rules; what is new here is the action
        # axis. There the excess was one vector per instance, so the sort had one ordering per
        # instance. Here every action has its own cost column and therefore its own ordering of the
        # classes, so the worst-case distribution differs per action: the box must be re-filled A
        # times. lower and upper gain a singleton axis at position 1 to broadcast against that action
        # axis, and excess is transposed to (A, K) so its class axis lines up with theirs. Every
        # reduction below stays on the class axis (-1); the action axis is only ever broadcast over,
        # never summed, which is what keeps the A fills independent.
        lower = rep_out.lower_bounds.to(excess.dtype).unsqueeze(1)  # (B, 1, K)
        upper = rep_out.upper_bounds.to(excess.dtype).unsqueeze(1)  # (B, 1, K)
        excess_t = excess.t()  # (A, K) the cost column of each action, as a row

        leftover = (1.0 - lower.sum(dim=-1, keepdim=True)).clamp_min(0.0)  # (B, 1, 1)
        slack = (upper - lower).clamp_min(0.0)  # (B, 1, K) headroom above the lower bound

        # The sort key is the cost matrix alone, which does not depend on the instance, so the
        # ordering is computed once on (A, K) and only the index is expanded to gather each
        # instance's slack. Sorting a (B, A, K) copy would recompute B identical orderings.
        excess_sorted, idx = excess_t.sort(dim=-1, descending=True)  # (A, K), (A, K)
        batch = lower.shape[0]
        idx_expanded = idx.unsqueeze(0).expand(batch, -1, -1)  # (B, A, K)
        slack_sorted = torch.gather(slack.expand_as(idx_expanded), dim=-1, index=idx_expanded)
        cumulative_before = slack_sorted.cumsum(dim=-1) - slack_sorted  # mass used by larger-excess classes
        take = (leftover - cumulative_before).clamp_min(0.0)
        used = torch.minimum(slack_sorted, take)  # mass actually placed on each class

        at_lower = (lower * excess_t).sum(dim=-1)  # (B, A) value of the all-at-lower-bound start point
        return at_lower + (used * excess_sorted).sum(dim=-1)  # (B, A)

    raise TypeError(f"Unsupported rep_out type for the cost-sensitive rules: {type(rep_out).__name__}.")


def cvar_minimax_actions(
    rep_out: TorchConvexCredalSet | TorchProbabilityIntervalsCredalSet,
    cost: torch.Tensor,
    v: float,
) -> torch.Tensor:
    """Our rule: the action minimising the worst-case truncated expected cost.

    Ties break towards the lowest action index (torch.argmin's first-minimum behaviour), so
    the rule is deterministic and runs reproduce exactly.

    Args:
        rep_out: A credal set, either TorchConvexCredalSet or TorchProbabilityIntervalsCredalSet.
        cost: Cost matrix, shape (K, A).
        v: The calibrated truncation threshold.

    Returns:
        Chosen action index per instance, shape (B,), dtype int64.
    """
    return worst_case_action_cost(rep_out, cost, v).argmin(dim=-1)


def singleton_credal_set(probs: torch.Tensor) -> TorchConvexCredalSet:
    """Wrap a point prediction as the degenerate credal set containing only that distribution.

    The ablation arm of this experiment is a point predictor under OUR rule, i.e.
    argmin_a E_p[(c(y, a) - v)+] with v calibrated exactly as for the credal methods. That is
    not new math: it is worst_case_action_cost on a credal set with a single vertex, because
    the supremum over a one-point set is the value at that point. Constructing it this way,
    rather than writing a separate truncated-expectation function, means the ablation and the
    credal arm share one code path and cannot drift apart -- which is the whole point of the
    comparison. (The user's parallel RL experiment calls this same arm cvar_mle.)

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


def cvar_minimax_point_actions(probs: torch.Tensor, cost: torch.Tensor, v: float) -> torch.Tensor:
    """Our rule applied to a POINT predictor: argmin_a E_p[(c(y, a) - v)+], the ablation arm.

    Isolates the rule from the credal set. The credal methods get both a credal set and this
    truncated-cost rule, so a win over the risk-neutral baselines cannot be attributed to the
    credal set unless the same rule on a point prediction is also measured. Implemented as
    cvar_minimax_actions on a singleton credal set (see singleton_credal_set), so the two arms
    differ in exactly one thing: whether the set has more than one member.

    Args:
        probs: Predicted distributions, shape (B, K).
        cost: Cost matrix, shape (K, A).
        v: The calibrated truncation threshold, calibrated the same way as for the credal arm.

    Returns:
        Chosen action index per instance, shape (B,), dtype int64.
    """
    return cvar_minimax_actions(singleton_credal_set(probs), cost, v)


def best_response_actions(probs: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    """Risk-neutral rule: the action minimising the expected cost under a point prediction.

    Args:
        probs: Predicted distributions, shape (B, K).
        cost: Cost matrix, shape (K, A).

    Returns:
        Chosen action index per instance, shape (B,), dtype int64.
    """
    expected_cost = probs.to(cost.dtype) @ cost  # (B, A)
    return expected_cost.argmin(dim=-1)  # (B,)


def calibrate_action_var_threshold(
    rep_outs: Sequence[TorchConvexCredalSet | TorchProbabilityIntervalsCredalSet],
    targets: Sequence[torch.Tensor],
    beta: float,
    cost: torch.Tensor,
    num_grid: int = 100,
) -> float:
    """Pick the VaR threshold v minimising the rule's realized-cost CVaR on a validation set.

    Calibration is label-based: it minimises F(v) = v + (1 / (beta * n)) * sum_i
    (cost(a_i(v), y_i) - v)+, the realized CVaR at level beta of the actions the rule takes at
    threshold v. Above the largest cost the truncation is always zero and F(v) = v only grows, so
    the optimum lies in [0, max cost]. This mirrors decision_rules.calibrate_var_threshold.

    THE CANDIDATE SET, and what it does and does not guarantee. Two distinct things move with v:

    1. The kinks of F GIVEN FIXED ACTIONS. With the actions held fixed, F is the Rockafellar-Uryasev
       objective on a fixed cost sample: piecewise linear and convex, with slope
       1 - #{i : c_i > v} / (beta * n), so its kinks sit EXACTLY at the realized cost values and its
       minimiser is always at one of them. Those values are entries of the cost table, a set of about
       ten integers. Evaluating F on that exact set (together with 0, the boundary) is therefore
       exact for this part, and a uniform grid is not: linspace(0, 10, 200) has step 10/199 and 199
       is prime, so it contains NO interior integer at all and the old default calibrated to
       v = 4.0201 where the true minimiser was 4.0.
    2. The ACTIONS themselves, through the rule's argmin. worst_case_action_cost is a supremum of
       functions that are piecewise linear in v, hence convex piecewise linear, and the action
       switches where two of them cross. Those crossings depend on each instance's credal set and
       are NOT confined to cost-table values, so F has jump discontinuities at points this function
       cannot enumerate cheaply (there are up to B * A^2 of them, instance by instance).

    So the search is a HYBRID and its guarantee is one-sided. Between consecutive action-switch
    points the action assignment is constant and the exact candidates contain that interval's
    unconstrained minimiser whenever it falls inside; what the exact set can miss is an optimum
    attained AT an action-switch point, where F drops discontinuously. The residual uniform grid of
    num_grid points is what samples those regions, at resolution v_max / (num_grid - 1) and with no
    optimality guarantee. In short: exact for the R-U kinks, approximate for the action switches.
    Claiming plain exactness here would be wrong.

    F is the Rockafellar-Uryasev objective. For a FIXED cost sample its minimum over v is the
    empirical CVaR at level beta, agreeing with metrics.cvar (the mean of the worst int(beta * N)
    costs) up to the rounding of that tail size, and exactly when beta * n is an integer. That
    equivalence does NOT survive this loop: the actions, and with them the cost sample, move with v,
    so F(v) is only an UPPER BOUND on the CVaR the driver reports and its slack is not a rounding
    term. The minimiser of F therefore need not minimise the reported metric. In spot checks during
    development the regret was zero in most runs but occasionally a few cost units; that measurement
    is not reproduced by anything in this repo, so treat it as an anecdote rather than a number to
    quote. The property is inherited from decision_rules.calibrate_var_threshold rather than
    introduced here, and R-U remains the right objective: minimising metrics.cvar directly would be
    a plateau-ridden step function of v.

    Args:
        rep_outs: Per-batch credal sets collected over the validation set.
        targets: Per-batch true labels, aligned with rep_outs.
        beta: CVaR tail level in (0, 1].
        cost: Cost matrix, shape (K, A), normally the cost_t of the resolved spec. Required:
            there is no default cost table, because a defaulted one would silently mis-score
            every other decision problem. Its unique entries are the exact part of the
            candidate set.
        num_grid: Size of the RESIDUAL uniform grid on [0, v_max], added to the exact candidates to
            sample the action-switch regions described above. It no longer controls the accuracy of
            the R-U minimisation, which is exact regardless, so the old advice to raise it for
            precision no longer applies; 0 or 1 disables the grid and searches the exact candidates
            alone. Every grid point costs a full pass over the validation set, so this is the entire
            cost of calibration -- the exact candidates are only about ten evaluations.

    Returns:
        The calibrated threshold v.

    Raises:
        ValueError: rep_outs is empty or beta lies outside (0, 1].
    """
    return calibrate_action_var_thresholds(rep_outs, targets, [beta], cost, num_grid)[0]


def calibrate_action_var_thresholds(
    rep_outs: Sequence[TorchConvexCredalSet | TorchProbabilityIntervalsCredalSet],
    targets: Sequence[torch.Tensor],
    betas: Sequence[float],
    cost: torch.Tensor,
    num_grid: int = 100,
) -> list[float]:
    """Calibrate v at SEVERAL tail levels at once, for the price of one.

    This is the singular calibration (see calibrate_action_var_threshold, which is now a one-line
    wrapper on this function and carries the full account of the objective and the candidate set)
    evaluated at every level in `betas`. It exists for the beta sweep, where calibrating each level
    independently would repeat identical work.

    WHY THIS IS EXACT AND NOT AN AMORTISED APPROXIMATION. The objective is

        F_beta(v) = v + excess(v) / (beta * n),   excess(v) = sum_i (cost(a_i(v), y_i) - v)+,

    and `excess` does not mention beta at all: the rule's actions a_i(v) depend on the credal sets,
    the cost table and v, and the truncation is at v. beta enters only as the scalar 1 / (beta * n)
    multiplying an already-computed number. So the expensive part -- one pass over the whole
    validation split per candidate v, running the rule and re-deciding every instance -- is shared
    by every level, and each additional beta costs one argmin over a stored vector of about 200
    floats. The returned values are bit-for-bit what the singular function returns for the same beta.

    What this does NOT change: F_beta remains an upper bound on the CVaR the driver reports, for the
    reason given in calibrate_action_var_threshold (the cost sample moves with v). Sweeping beta
    does not tighten that, and the minimiser at one beta says nothing about the reported metric at
    another.

    Args:
        rep_outs: Per-batch credal sets collected over the validation set.
        targets: Per-batch true labels, aligned with rep_outs.
        betas: CVaR tail levels, each in (0, 1]. Order is preserved in the result; duplicates are
            permitted and simply produce equal entries.
        cost: Cost matrix, shape (K, A). See calibrate_action_var_threshold.
        num_grid: Size of the residual uniform grid. See calibrate_action_var_threshold.

    Returns:
        One calibrated threshold per entry of `betas`, in the same order.

    Raises:
        ValueError: rep_outs or betas is empty, or some beta lies outside (0, 1].
    """
    if not rep_outs:
        raise ValueError("calibrate_action_var_thresholds needs at least one batch of credal sets.")
    if not betas:
        raise ValueError("calibrate_action_var_thresholds needs at least one beta.")
    bad = [b for b in betas if not 0.0 < b <= 1.0]
    if bad:
        raise ValueError(f"every beta must be in (0, 1], got {bad}.")

    total = sum(int(t.numel()) for t in targets)
    v_max = float(cost.max().item())

    # Exact candidates: every realized cost the table can produce, plus the boundary 0 (which every
    # current table contains anyway, but must not depend on). Sorted and de-duplicated so ties break
    # towards the smallest v exactly as the grid loop always did.
    candidates = {0.0} | {float(value) for value in cost.flatten().tolist()}
    if num_grid > 1:
        candidates |= {float(value) for value in torch.linspace(0.0, v_max, num_grid).tolist()}

    grid = sorted(candidates)
    # The beta-free half of the objective, one entry per candidate. Collected in full rather than
    # reduced on the fly precisely because the reduction is what differs per beta.
    excesses: list[float] = []
    for v in tqdm(grid, desc="Calibrating v (actions)"):
        excess = 0.0
        for rep_out, batch_targets in zip(rep_outs, targets, strict=True):
            actions = cvar_minimax_actions(rep_out, cost, v)
            realized = cost.to(actions.device)[batch_targets, actions]
            excess += float((realized - v).clamp_min(0.0).sum().item())
        excesses.append(excess)

    thresholds: list[float] = []
    for beta in betas:
        best_v, best_objective = 0.0, float("inf")
        for v, excess in zip(grid, excesses, strict=True):
            objective = v + excess / (beta * total)  # F(v): the realized CVaR at level beta
            # Strict <, so a tie keeps the earlier (smallest) v. Worth stating because F is genuinely
            # flat over stretches of v on a discrete cost sample; this makes the choice deterministic
            # and matches decision_rules.calibrate_var_threshold.
            if objective < best_objective:
                best_objective, best_v = objective, v
        thresholds.append(best_v)
    return thresholds
