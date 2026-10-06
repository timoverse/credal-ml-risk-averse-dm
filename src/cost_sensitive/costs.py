"""Cost matrices and outcome bookkeeping for the cost-sensitive experiments.

A decision maker observes an instance and chooses one of a small number of treatment ACTIONS,
scored by a cost matrix over the true condition. Everything a decision problem needs to be run
through that pipeline -- label names, action names, the cost table, which action means
"do nothing" and which labels make doing nothing a critical mistake -- is bundled into one
frozen :class:`CostSpec`, registered by name and resolved with :func:`get_cost_spec_by_name`.

We work with costs rather than utilities so the numbers line up with the rest of the repo
(every risk metric in src/metrics.py is a loss to be minimised). A spec may nonetheless be
DEFINED from a published utility table (cost = max_utility - utility, a decreasing affine map
that turns utility maximisation into the equivalent cost minimisation); the `utility` field is
then primary, and otherwise it is the derived mirror max_cost - cost, kept only so every spec
exposes the same fields.

There is NO module-level default cost table, and no COST / COST_T / LABELS / ACTIONS aliases:
every consumer resolves a spec by name and passes it explicitly, because a defaulted matrix is
silently wrong rather than loudly wrong the moment a second decision problem exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

# The name of the "do nothing" action. Shared by every spec: the critical-mistake notion is
# defined as a severe label receiving this action, so the specs must agree on how to spell it.
NO_ACTION = "No Action"


@dataclass(frozen=True, eq=False, repr=False)
class CostSpec:
    """Everything the cost-sensitive pipeline needs to know about one dataset's decision problem.

    Frozen, and its numpy arrays are additionally frozen with setflags(write=False). These tables
    are shared state: the driver runs many methods and seeds in a single process and the rules
    truncate costs, so an accidental in-place write (say a truncation like
    np.clip(cost, None, v, out=cost)) would silently corrupt every method evaluated afterwards,
    yielding plausible but wrong numbers with no error. Freezing turns that mistake into an
    immediate ValueError. Fancy indexing such as cost[targets, actions] returns copies, so
    ordinary downstream use is unaffected.

    eq=False because the dataclass-generated __eq__ would compare numpy arrays elementwise and
    raise on the ambiguous truth value; specs are identified by `name`, and identity comparison
    is what callers actually want.

    Attributes:
        name: The spec name, the key it is registered under (e.g. "synthetic_triage").
        labels: Diagnosis classes, in the order used as label indices. This ordering is a hard
            contract with the data source, which must map its classes onto exactly these indices.
        actions: Treatment actions, in the order used as action indices.
        cost: cost[y, a] is the cost of action a when the true condition is y, shape (K, A).
            NOT necessarily square: there can be more conditions than distinct managements.
        utility: max_utility - cost, shape (K, A). Primary for a spec defined from a published
            utility table, derived otherwise; see the module docstring.
        max_utility: The offset relating the two. It is the largest cost entry ONLY when the
            smallest utility is 0, which is not a law: a spec whose published utility table
            bottoms out at, say, 2 has max_utility = cost.max() + 2. Utility-primary specs
            therefore declare it and _build_spec checks it; cost-primary specs leave it implicit
            and get cost.max().
        cost_t: The torch mirror of `cost`, so the decision rules do not rebuild a tensor from the
            numpy table at every call. Materialising it once matters because the rules run inside a
            per-batch evaluation loop, where torch.tensor(cost) would otherwise copy the table and
            allocate on every batch of every method and seed.

            torch has no equivalent of numpy's setflags(write=False), so unlike `cost` this cannot
            be hard-frozen: immutability is a convention here, not an enforced invariant. Treat it
            as read-only and never write into it in place (no out= targets, no index assignment).
            The numpy `cost` stays the frozen source of truth; this is a derived copy, so a stray
            write would desynchronise the two rather than raise.
        no_action_index: Index of the "do nothing" action within `actions`.
        critical_label_indices: Labels for which doing nothing is a severe omission of care (the
            "critical mistakes"). Defined by the rule in :func:`uniquely_worst_no_action_labels`,
            which _build_spec enforces.
    """

    name: str
    labels: tuple[str, ...]
    actions: tuple[str, ...]
    cost: npt.NDArray[np.float64]
    utility: npt.NDArray[np.float64]
    max_utility: float
    cost_t: torch.Tensor
    no_action_index: int
    critical_label_indices: tuple[int, ...]

    def is_critical_mistake(
        self, actions: npt.NDArray[np.integer], targets: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.bool_]:
        """Mask of critical mistakes: a critical-label case assigned the no-action option.

        Args:
            actions: Chosen action index per instance, shape (N,).
            targets: True label index per instance, shape (N,).

        Returns:
            Boolean mask of shape (N,), True where the decision is a critical mistake.
        """
        critical_condition = np.isin(targets, self.critical_label_indices)
        return critical_condition & (actions == self.no_action_index)

    def __repr__(self) -> str:
        """Short identification, without dumping the cost table.

        Returns:
            A one-line summary naming the dataset and the matrix shape.
        """
        return f"CostSpec(name={self.name!r}, labels={len(self.labels)}, actions={len(self.actions)})"


def uniquely_worst_no_action_labels(
    labels: tuple[str, ...],
    actions: tuple[str, ...],
    cost: npt.NDArray[np.float64],
) -> tuple[str, ...]:
    """The labels for which the no-action option is the UNIQUELY worst action, in label order.

    This is the definition of a "critical mistake" used by Kiyani et al. (2025). Their Figure 3
    caption states it directly -- "for samples with a critical ground-truth label, we report the
    fraction of cases in which each method chooses the WORST action" -- with the accompanying
    Figure 3(b) labelling each qualifying class "-> No action".

    Uniqueness is required, not merely being tied for worst. If some other action is equally bad,
    "the method chose the worst action" no longer singles out doing nothing, and a rate conditioned
    on the no-action column would then measure something the caption does not describe. On the
    triage table this is what keeps Flu non-critical: its No Action and ICU entries tie at 4.

    Args:
        labels: Diagnosis classes in label-index order.
        actions: Treatment actions in action-index order; must contain NO_ACTION.
        cost: The (K, A) cost table, rows indexed by label and columns by action.

    Returns:
        The qualifying label names, ordered by label index.

    Raises:
        ValueError: `actions` does not contain the no-action option.
    """
    if NO_ACTION not in actions:
        raise ValueError(f"actions must contain {NO_ACTION!r}, got {actions}.")
    no_action = actions.index(NO_ACTION)
    critical = []
    for index, label in enumerate(labels):
        row = cost[index]
        worst = row.max()
        # Uniquely worst: attained by exactly one action, and that action is doing nothing.
        if int((row == worst).sum()) == 1 and bool(row[no_action] == worst):
            critical.append(label)
    return tuple(critical)


def _build_spec(
    name: str,
    labels: tuple[str, ...],
    actions: tuple[str, ...],
    cost: npt.NDArray[np.float64],
    critical_labels: tuple[str, ...],
    max_utility: float | None = None,
) -> CostSpec:
    """Assemble a CostSpec from a cost table, deriving the utility mirror and freezing the arrays.

    Args:
        name: The dataset key.
        labels: Diagnosis classes in label-index order.
        actions: Treatment actions in action-index order.
        cost: The (K, A) cost table, rows indexed by label and columns by action.
        critical_labels: Names of the labels for which the no-action option is a critical mistake.
            Declared explicitly rather than derived, so the spec reads as a statement about the
            decision problem -- but checked against uniquely_worst_no_action_labels, so it cannot
            drift away from the definition when a cost table is edited.
        max_utility: The offset relating cost and utility, for specs defined from a published
            UTILITY table. None (the default) means "cost-primary": the offset is cost.max(), which
            makes the derived utility bottom out at exactly 0. Pass it when a utility table is the
            source of truth, so that a table whose minimum utility is strictly positive round-trips
            instead of silently agreeing only because that minimum happened to be 0.

    Returns:
        The assembled spec, with read-only numpy arrays.

    Raises:
        ValueError: The shapes disagree with the label/action names, a cost is negative, a named
            critical label or the no-action option is missing, the declared critical labels
            disagree with the uniquely-worst-action rule, or max_utility is smaller than the
            largest cost (which would make some utility negative).
    """
    if cost.shape != (len(labels), len(actions)):
        raise ValueError(f"{name}: cost matrix has shape {cost.shape}, expected {(len(labels), len(actions))}.")
    if bool((cost < 0.0).any()):
        raise ValueError(f"{name}: costs must be non-negative, got a minimum of {cost.min()}.")
    if NO_ACTION not in actions:
        raise ValueError(f"{name}: actions must contain {NO_ACTION!r}, got {actions}.")
    missing = [label for label in critical_labels if label not in labels]
    if missing:
        raise ValueError(f"{name}: critical labels {missing} are not among the labels {labels}.")

    implied = uniquely_worst_no_action_labels(labels, actions, cost)
    if tuple(critical_labels) != implied:
        raise ValueError(
            f"{name}: declared critical labels {tuple(critical_labels)} disagree with the labels for "
            f"which {NO_ACTION!r} is the uniquely worst action, {implied}. The critical-mistake rate "
            f"is defined by that rule (Kiyani et al. 2025, Figure 3 caption), so the two must match: "
            f"either the cost table or the declaration is wrong. See uniquely_worst_no_action_labels."
        )

    # cost.max() is the offset a cost-primary spec implies. For a utility-primary spec it is only
    # the right answer when the published utility table's minimum is 0 -- true of Kiyani et al.'s
    # Table 1, and exactly the kind of coincidence that makes a future spec disagree silently.
    if max_utility is None:
        max_utility = float(cost.max())
    elif max_utility < float(cost.max()):
        raise ValueError(
            f"{name}: max_utility {max_utility} is below the largest cost {float(cost.max())}, which "
            f"would make some utility negative. The offset must be at least the largest cost."
        )
    utility = max_utility - cost
    cost.setflags(write=False)
    utility.setflags(write=False)
    return CostSpec(
        name=name,
        labels=labels,
        actions=actions,
        cost=cost,
        utility=utility,
        max_utility=max_utility,
        cost_t=torch.tensor(cost, dtype=torch.float64),
        no_action_index=actions.index(NO_ACTION),
        critical_label_indices=tuple(labels.index(label) for label in critical_labels),
    )


# --------------------------------------------------------------------------------------------
# synthetic_triage: the decision problem of the synthetic triage experiment
# (src/cost_sensitive/cost_sensitive_triage.py).
# --------------------------------------------------------------------------------------------
#
# Designed under a TWO-THRESHOLD rule (learned the hard way on earlier, since-removed cost
# matrices): the no-action dominance threshold D = min over non-no-action columns of
# cost[healthy, a] and the collapse threshold v* = min_a max_y cost must be separated, with a
# wide informative band below D. For any calibrated v >= D the no-action option is elementwise
# dominated and no arm can choose it (every critical-mistake rate is identically 0), and at
# v = v* the rule decides one constant action for every instance and all arms coincide -- so
# with D = v* the metric measures the table, not the method.
#
#   D  = 6  (the Treat Flu column's Healthy entry): for v < 6 the no-action option is live, so
#           the critical-mistake metric measures the method, not the table.
#   v* = 8  (the ICU column's maximum): no constant-action collapse before 8.
#
# Sepsis is the ONLY critical label: its no-action cost 20 is the uniquely worst entry of its
# row. Flu's row deliberately TIES No Action and ICU at 4, so Flu does not qualify under the
# uniquely-worst-action rule -- sending a flu patient home and putting one in the ICU are both
# wasteful, but neither is the catastrophe this experiment counts. Hedging is priced: ICU costs
# 8 on a healthy patient and Treat Flu costs 6, so there is no near-free refuge column for the
# calibration to collapse onto.
#
# The flu row is CHEAP (max 4) on purpose, and this is load-bearing for the calibration's
# selectivity. The constant-ICU policy has realized CVaR exactly 9 (the healthy/ICU entry) at
# every tail level, so a selective policy wins the calibration only if fewer than tau * N
# instances cost 9 or more. With an expensive flu row (an earlier draft used 9/1/9), the
# flu-boundary misroutes alone pushed the >= 9 count past the tail size and the calibrated v
# collapsed to v* for every arm -- the constant-refuge failure every earlier matrix kept
# hitting, reproduced from the other side. With flu errors at 4, only hedged healthy patients
# (9) and missed sepsis (20/14) reach the tail, and input-dependent hedging genuinely beats
# the constant policy.
_TRIAGE_LABELS: tuple[str, ...] = ("Healthy", "Flu", "Sepsis")
_TRIAGE_ACTIONS: tuple[str, ...] = (NO_ACTION, "Treat Flu", "ICU")

_TRIAGE_COST: npt.NDArray[np.float64] = np.array(
    [
        [0.0, 6.0, 8.0],  # Healthy -> No Action (0); every intervention on a healthy patient is costly
        [4.0, 1.0, 4.0],  # Flu     -> Treat Flu (1); home and ICU tie at 4, so Flu is NOT critical
        [20.0, 14.0, 2.0],  # Sepsis -> ICU (2); sent home = 20, the catastrophe the experiment counts
    ]
)

TRIAGE_SPEC: CostSpec = _build_spec(
    name="synthetic_triage",
    labels=_TRIAGE_LABELS,
    actions=_TRIAGE_ACTIONS,
    cost=_TRIAGE_COST,
    critical_labels=("Sepsis",),
)


# The registry: every cost spec, keyed by spec.name, resolved by get_cost_spec_by_name.
_SPECS_BY_NAME: dict[str, CostSpec] = {
    TRIAGE_SPEC.name: TRIAGE_SPEC,
}


def get_cost_spec_by_name(name: str) -> CostSpec:
    """Look up a cost specification by name.

    Args:
        name: The spec name, e.g. "synthetic_triage".

    Returns:
        The registered CostSpec.

    Raises:
        KeyError: No spec has this name. The message lists the available names, because the usual
            cause is a typo in the caller's spec argument.
    """
    try:
        return _SPECS_BY_NAME[name]
    except KeyError:
        available = ", ".join(sorted(_SPECS_BY_NAME))
        raise KeyError(
            f"No cost specification named {name!r}. Available specs: {available}. Add a CostSpec to "
            f"src/cost_sensitive/costs.py to register a new one."
        ) from None
