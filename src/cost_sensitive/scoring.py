"""Cost-table diagnostics for the cost-sensitive evaluation.

Once the streaming driver/re-scorer pipeline of the removed image experiments went, one function
remained in use: the no-action dominance threshold, a structural property of a cost table that the
synthetic triage experiment (src/cost_sensitive/cost_sensitive_triage.py) checks at startup and its
design comments lean on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cost_sensitive.costs import CostSpec


def no_action_dominated_threshold(spec: CostSpec) -> float | None:
    """The smallest truncation threshold v at which the no-action option can never be chosen.

    Our rule minimises the worst-case TRUNCATED cost, so it compares the columns of
    (cost - v)+ rather than of cost. Truncation is a floor at zero, and it compresses large entries
    by the same amount it compresses small ones -- so raising v can make one action's truncated
    column fall elementwise below another's even though neither dominated the other before. Once
    some action's truncated column is <= the no-action column everywhere, the no-action option is
    never a unique argmin and, with ties broken towards the lowest action index, is chosen only when
    it is itself that dominating column. It effectively leaves the action set.

    THIS IS WHY THE CRITICAL-MISTAKE RATE MUST NOT BE READ AT FACE VALUE ABOVE THIS THRESHOLD. The
    rate counts critical-label cases that receive the no-action option. Above this v, EVERY decision
    maker under our rule -- a credal set of any width, or the singleton of any point predictor --
    scores exactly zero, because nothing can pick the action the metric counts. The zero is then a
    property of the cost table and the threshold, not evidence that the method handled uncertainty
    well, and comparing arms on it is comparing constants. The synthetic triage table was designed
    around this: its threshold is v = 6, well below its collapse threshold v* = 8, so there is a
    band of calibrated thresholds in which decisions still vary with the input and the metric
    measures the method.

    The search is over the distinct entries of the cost table plus 0. That set is exact: the ordering
    between two truncated columns can only change where some entry meets the floor, i.e. at v equal
    to a cost value, so dominance is constant between consecutive entries and first appears at one of
    them.

    Args:
        spec: The cost specification whose action set is being checked.

    Returns:
        The smallest threshold at which the no-action option is dominated, or None if no threshold
        up to the largest cost dominates it (the metric is then meaningful at every usable v).
    """
    cost = spec.cost
    no_action = spec.no_action_index
    others = [a for a in range(cost.shape[1]) if a != no_action]
    for v in sorted({0.0} | {float(value) for value in cost.flatten()}):
        truncated = np.clip(cost - v, 0.0, None)
        if any(bool((truncated[:, a] <= truncated[:, no_action]).all()) for a in others):
            return float(v)
    return None
