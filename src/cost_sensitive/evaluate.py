"""Evaluation of decided actions: realized cost, its CVaR and mean, and critical mistakes.

The CVaR aggregation is imported from src/metrics.py rather than reimplemented, so the
tail definition (mean of the worst q fraction of values) is identical to every other
risk number reported in this repo.

The critical-mistake rate is CONDITIONAL on the true label being one of the spec's critical
labels (Sepsis for synthetic_triage), matching Kiyani et al. (2025) Figure 3b and the figure's
axis label. See critical_mistake_rate for why the marginal version is a trap, and
critical_mistake_rates_by_label for why the POOLED rate is not what the paper plots.

A cost table with a ZERO-COST action in each row makes the headline CVaR degenerate at a badly
chosen beta: see cvar_is_degenerate. Nothing in metrics.cvar can detect that, because the
degeneracy is a property of THAT cost table meeting THIS tail level, so the check lives here
and the caller runs it on every (method, seed).

The label-dependent helpers take a REQUIRED `spec`. There is deliberately no default: a default
matrix silently mis-scores any other decision problem whose label count happens to fit, and the
caller always knows which spec it is scoring under.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from metrics import cvar

if TYPE_CHECKING:
    import numpy.typing as npt

    from cost_sensitive.costs import CostSpec


def realized_costs(
    actions: npt.NDArray[np.integer],
    targets: npt.NDArray[np.integer],
    spec: CostSpec,
) -> npt.NDArray[np.float64]:
    """Cost actually incurred per instance: spec.cost[true label, chosen action].

    Args:
        actions: Chosen action index per instance, shape (N,).
        targets: True label index per instance, shape (N,).
        spec: The dataset's cost specification, resolved with get_cost_spec_by_name.

    Returns:
        Realized cost per instance, shape (N,).
    """
    return spec.cost[targets, actions]


def critical_mistake_rate(
    actions: npt.NDArray[np.integer],
    targets: npt.NDArray[np.integer],
    spec: CostSpec,
) -> float:
    """Fraction of the critical-label instances that were sent home with the no-action option.

    CONDITIONAL, not marginal: the denominator is the number of instances whose true label is one
    of the spec's critical labels, not the test-set size. This is the aggregation of what Kiyani
    et al. (2025) report in Figure 3b -- among samples with a critical ground-truth label, the
    fraction where the method chooses the worst action. The conditional and marginal versions
    differ by the critical base rate, so the marginal version reads as a better method than it is.
    Nothing downstream distinguishes them, hence this being its own named function with the
    semantics pinned by a selftest check.

    POOLED over the critical labels, which the paper is not: Figure 3b draws one bar group per
    critical label. Pooling weights each label by its base rate, so a method that fails badly on a
    rare critical label is averaged away by its performance on a common one. This function is kept
    because a single summary number is what the existing bar figure plots and it is a reasonable
    one, but critical_mistake_rates_by_label is the paper-faithful breakdown and both are stored
    on every record.

    Args:
        actions: Chosen action index per instance, shape (N,).
        targets: True label index per instance, shape (N,).
        spec: The dataset's cost specification, which fixes the critical labels.

    Returns:
        The conditional rate in [0, 1]. Returns 0.0 when the sample contains no critical-label
        instance at all: the rate is genuinely undefined there (0/0), but a NaN would
        propagate silently into the seed mean and blank one bar of the figure with no error, so
        the vacuous case -- no critical instance was mishandled, because there was none -- is
        reported instead. On the real test split the denominator is in the thousands, so this
        branch only fires on degenerate inputs such as a self-test fixture.
    """
    critical = np.isin(targets, spec.critical_label_indices)
    denominator = int(critical.sum())
    if denominator == 0:
        return 0.0
    return float(spec.is_critical_mistake(actions, targets).sum() / denominator)


def critical_mistake_rates_by_label(
    actions: npt.NDArray[np.integer],
    targets: npt.NDArray[np.integer],
    spec: CostSpec,
) -> dict[str, float]:
    """Per-critical-label conditional rate of choosing the no-action option, the paper-faithful form.

    Kiyani et al. (2025) Figure 3b has one bar GROUP per critical ground-truth label -- Pneumonia,
    COVID-19 and Lung Opacity -- not one pooled bar. The distinction is not cosmetic. The pooled
    rate in critical_mistake_rate is a base-rate weighted average of these, so a method that sends
    home half of the COVID-19 cases still looks respectable if COVID-19 is the rarest of the three
    and it handles the other two well. Reporting the breakdown is what makes that visible, and it
    is the only form in which our numbers can be put next to the paper's.

    The keys are label NAMES rather than indices so the mapping survives into the results pickle
    without the reader needing the spec to interpret it, and so a later figure can label its bar
    groups directly. Insertion order follows spec.critical_label_indices, i.e. label-index order,
    which fixes the group order in that figure.

    Args:
        actions: Chosen action index per instance, shape (N,).
        targets: True label index per instance, shape (N,).
        spec: The dataset's cost specification, which fixes the critical labels.

    Returns:
        Mapping from critical label name to its conditional rate in [0, 1]. Every critical label of
        the spec is present, whether or not the sample contains an instance of it. A label with NO
        instance in the sample gets 0.0, never NaN: the rate is genuinely undefined there (0/0), but
        a NaN would propagate into the seed mean and silently blank a bar of the figure with no
        error, so the vacuous reading -- no critical instance of this label was mishandled, because
        there was none -- is reported instead. This matches critical_mistake_rate's handling of the
        same 0/0 case. On a real test split every denominator is in the hundreds or thousands, so
        this branch fires only on degenerate inputs such as a self-test fixture.
    """
    mistakes = spec.is_critical_mistake(actions, targets)
    rates: dict[str, float] = {}
    for index in spec.critical_label_indices:
        of_this_label = targets == index
        denominator = int(of_this_label.sum())
        # Conditioning on the label first: is_critical_mistake already ANDs in "is a critical
        # label", so intersecting with this label's mask leaves exactly its no-action cases.
        rates[spec.labels[index]] = 0.0 if denominator == 0 else float((mistakes & of_this_label).sum() / denominator)
    return rates


def nonzero_cost_fraction(costs: npt.NDArray[np.floating]) -> float:
    """Fraction of instances that incurred a non-zero cost, i.e. where the decision was not free.

    Because every row of the cost matrix has a zero-cost action, this is exactly the fraction of
    instances on which the decision maker did NOT take the action that is optimal for the true
    label. It is
    the quantity the CVaR degeneracy turns on, so it is named rather than inlined, and it is
    stored on every record.

    Args:
        costs: Realized cost per instance, shape (N,).

    Returns:
        The fraction in [0, 1]; 0.0 for an empty input.
    """
    if costs.size == 0:
        return 0.0
    return float((costs > 0).mean())


def cvar_is_degenerate(costs: npt.NDArray[np.floating], beta: float) -> bool:
    """Whether CVaR_beta of this cost vector is a rescaled MEAN rather than a tail measurement.

    Every cost matrix in costs.py has a ZERO-COST action per row: a correct decision costs exactly 0.
    metrics.cvar averages the worst int(beta * N) costs. So once the tail is at least as large as
    the number of non-zero costs, the tail is padded out with zeros and

        CVaR_beta = sum(all costs) / int(beta * N) = mean(costs) * N / int(beta * N),

    an exact affine rescaling of the mean carrying no tail information whatsoever. The headline
    panel would then plot the same picture as the mean panel, up to that constant.

    The factor is mean / beta only when beta * N is an INTEGER, which is a property of the split
    size and not something to assume: at N = 400 and beta = 0.1 it holds (beta * N = 40 exactly),
    while at N = 4233 and beta = 0.02 it fails (beta * N = 84.66 while the divisor is
    int(beta * N) = 84, putting the reported CVaR about 0.8% above mean / beta). The displayed
    identity holds in general; "mean / beta" is an approximation of it that is exact only on a
    split where beta * N lands on an integer.

    Why this is not merely cosmetic: an ACCURATE risk-neutral baseline takes the zero-cost action
    most of the time, so it has few non-zero costs and falls into the degenerate regime, while our
    rule deliberately hedges onto off-diagonal actions and so has a genuine tail. At too large a
    beta the figure would compare an artifact against a real measurement.

    Two ways to be degenerate, both caught here:

    1. int(beta * N) >= (costs > 0).sum() -- the zero-padding above. Degenerate AT EQUALITY too:
       when the tail is exactly the non-zero entries their mean is sum / nnz = sum / int(beta * N),
       which is still mean / beta.
    2. int(beta * N) == 0 -- beta is below one instance's worth. This is a separate and easily
       missed hole: numpy's `sorted[-0:]` is the WHOLE array, not an empty one, so metrics.cvar
       silently returns the plain mean. Condition 1 alone would call this NON-degenerate whenever
       some cost is non-zero, which is exactly backwards.

    Args:
        costs: Realized cost per instance, shape (N,).
        beta: The CVaR tail fraction the metric is reported at.

    Returns:
        True if CVaR_beta over these costs carries no tail information beyond the mean.
    """
    if costs.size == 0:
        return True
    tail_size = int(beta * costs.size)
    if tail_size == 0:
        return True
    return tail_size >= int((costs > 0).sum())


def max_nondegenerate_beta(costs: npt.NDArray[np.floating]) -> float:
    """The exclusive upper bound on beta at which CVaR still measures a tail for these costs.

    From cvar_is_degenerate, CVaR_beta is non-degenerate exactly when int(beta * N) < nnz, which
    for integer nnz holds precisely when beta * N < nnz. The bound is therefore nnz / N, the
    non-zero cost fraction itself, and it is EXCLUSIVE: beta must sit strictly below it.

    This is the actionable half of the diagnostic. Knowing that a method is degenerate is not
    enough to fix a config; knowing that its costs stop being degenerate below beta = 0.031 is.

    Args:
        costs: Realized cost per instance, shape (N,).

    Returns:
        The exclusive upper bound on a usable beta. 0.0 when no cost is non-zero, i.e. a method
        that is perfect on the test split: no beta measures a tail, because there is no tail.
    """
    if costs.size == 0:
        return 0.0
    return float(int((costs > 0).sum()) / costs.size)


def metric_bundle(
    actions: npt.NDArray[np.integer],
    targets: npt.NDArray[np.integer],
    beta: float,
    spec: CostSpec,
) -> dict[str, float]:
    """The three reported metrics for one (method, seed) run.

    Args:
        actions: Chosen action index per instance, shape (N,).
        targets: True label index per instance, shape (N,).
        beta: CVaR tail fraction in (0, 1].
        spec: The dataset's cost specification, resolved with get_cost_spec_by_name.

    Returns:
        Dict with cvar_cost (the headline), mean_cost, and critical_mistake_rate. The last is
        CONDITIONAL on the true label being critical, and POOLED over the critical labels; see
        critical_mistake_rate.

    Note:
        Deliberately flat -- every value is a float, which is what the driver's per-method summary
        and the bar figure's seed aggregation assume. The paper-faithful per-label breakdown is a
        mapping, so it is computed by critical_mistake_rates_by_label and attached to the record by
        the driver rather than folded in here, where it would break that contract.
    """
    costs = realized_costs(actions, targets, spec)
    return {
        "cvar_cost": float(cvar(costs, beta)),
        "mean_cost": float(costs.mean()),
        "critical_mistake_rate": critical_mistake_rate(actions, targets, spec),
    }
