"""Committed numerical self-tests for the cost-sensitive experiment.

Run with: uv run python src/cost_sensitive/selftest.py

This repo has no pytest infrastructure by convention, so the numerical claims of
src/cost_sensitive (closed-form worst-case cost, calibration, cost table) are checked
here with plain asserts and re-run manually after changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from probly.representation.credal_set.torch import TorchConvexCredalSet, TorchProbabilityIntervalsCredalSet
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution

from cost_sensitive.costs import (
    NO_ACTION,
    TRIAGE_SPEC,
    _build_spec,
    get_cost_spec_by_name,
    uniquely_worst_no_action_labels,
)
from cost_sensitive.evaluate import (
    critical_mistake_rate,
    critical_mistake_rates_by_label,
    cvar_is_degenerate,
    max_nondegenerate_beta,
    metric_bundle,
    nonzero_cost_fraction,
    realized_costs,
)
from cost_sensitive.rules import (
    best_response_actions,
    calibrate_action_var_threshold,
    calibrate_action_var_thresholds,
    cvar_minimax_actions,
    cvar_minimax_point_actions,
    singleton_credal_set,
    worst_case_action_cost,
)
from cost_sensitive.scoring import no_action_dominated_threshold
from metrics import cvar

# TEST FIXTURES, not registered specs. The covid and retina image experiments were removed, but
# their matrices remain the best geometries for exercising the machinery: the covid table has a
# zero-cost diagonal (so the CVaR degeneracy checks have something to fire on) and a worst-case
# tie on its Normal row (so the uniquely-worst-action rule's exclusion branch is covered); the
# retina table is NON-SQUARE (5 labels, 4 actions), which is what catches K-vs-A axis mix-ups.
# The registered synthetic_triage spec has none of these properties, so the fixtures stay as
# pure test data, built through the same factory the real spec uses.
COVID = _build_spec(
    name="covid_fixture",
    labels=("Normal", "Pneumonia", "COVID-19", "Lung Opacity"),
    actions=(NO_ACTION, "Antibiotics", "Quarantine", "Testing"),
    # 10 - the published utility table of Kiyani et al. (ICML 2025), Table 1.
    cost=10.0
    - np.array(
        [
            [10.0, 2.0, 2.0, 4.0],  # Normal
            [0.0, 10.0, 3.0, 7.0],  # Pneumonia
            [0.0, 3.0, 10.0, 8.0],  # COVID-19
            [1.0, 4.0, 4.0, 10.0],  # Lung Opacity
        ]
    ),
    critical_labels=("Pneumonia", "COVID-19", "Lung Opacity"),
    max_utility=10.0,
)
RETINA = _build_spec(
    name="retina_fixture",
    labels=("No DR", "Mild", "Moderate", "Severe", "Proliferative"),
    actions=(NO_ACTION, "Recheck 6mo", "Refer", "Urgent Referral"),
    cost=np.array(
        [
            [0.0, 2.0, 4.0, 7.0],  # grade 0, no DR
            [3.0, 0.0, 3.0, 6.0],  # grade 1, mild
            [6.0, 3.0, 0.0, 3.0],  # grade 2, moderate
            [9.0, 6.0, 2.0, 0.0],  # grade 3, severe
            [10.0, 8.0, 3.0, 0.0],  # grade 4, proliferative
        ]
    ),
    critical_labels=("Moderate", "Severe", "Proliferative"),
)

# Every spec the per-spec structural checks iterate over: the two fixtures plus the registered one.
ALL_SPECS = {spec.name: spec for spec in (COVID, RETINA, TRIAGE_SPEC)}


def check_critical_mistake() -> None:
    """A critical mistake is a critical-label case assigned No Action, and nothing else.

    The covid critical labels are Pneumonia, COVID-19 AND Lung Opacity -- three, not two. Kiyani
    et al. (2025) define a critical mistake in the Figure 3 caption as choosing the WORST action for
    a critical ground-truth label, and Figure 3(b) has three bar groups, each "-> No action". Under
    the paper's Table 1, No Action is the uniquely worst action for exactly those three (costs 10,
    10, 9) and not for Normal (whose worst is a tie between Antibiotics and Quarantine at 8).
    The Section 5 prose "e.g., pneumonia or COVID-19" is an example, not the definition.
    """
    assert COVID.critical_label_indices == (1, 2, 3), (
        f"covid critical labels must be Pneumonia, COVID-19 and Lung Opacity, got {COVID.critical_label_indices}"
    )
    labels = np.array([0, 1, 2, 3, 1, 2])
    actions = np.array([0, 0, 0, 0, 3, 1])
    got = COVID.is_critical_mistake(actions, labels)
    # Index 3 is Lung Opacity given No Action: a critical mistake now, and NOT one before this fix.
    expected = np.array([False, True, True, True, False, False])
    assert np.array_equal(got, expected), f"critical-mistake mask wrong: {got} vs {expected}"
    print("OK: critical-mistake predicate (three covid critical labels)")


def check_critical_labels_follow_the_worst_action_rule() -> None:
    """Every spec's declared critical labels are exactly those where No Action is uniquely worst.

    The rule is the paper's definition of a critical mistake (Figure 3 caption). Asserting it per
    spec rather than pinning two hardcoded tuples is what keeps a future cost-table edit honest: a
    table change that makes some other label's worst action "do nothing" must be accompanied by a
    change to the declaration, and _build_spec now refuses to construct a spec where they disagree.
    """
    for name, spec in ALL_SPECS.items():
        declared = tuple(spec.labels[index] for index in spec.critical_label_indices)
        implied = uniquely_worst_no_action_labels(spec.labels, spec.actions, spec.cost)
        assert declared == implied, f"{name}: declared critical labels {declared} != rule's {implied}"
        # And the rule really is discriminating on this table, not trivially selecting everything.
        assert 0 < len(declared) < len(spec.labels), f"{name}: the rule selected {declared} out of {spec.labels}"
        for index in spec.critical_label_indices:
            row = spec.cost[index]
            assert row[spec.no_action_index] == row.max(), f"{name}: {spec.labels[index]} does not peak at No Action"
            assert int((row == row.max()).sum()) == 1, f"{name}: {spec.labels[index]}'s worst action is a tie"

    # The retina spec includes Moderate on exactly this rule (row [6, 3, 0, 3]): a moderate case
    # left for a year is the worst of its four options. It was previously omitted under the
    # narrower reading "sight-threatening disease", which is a different definition from the one
    # the metric implements.
    assert RETINA.critical_label_indices == (2, 3, 4), (
        f"retina critical labels must be Moderate, Severe and Proliferative, got {RETINA.critical_label_indices}"
    )
    # Grades 0 and 1 must NOT qualify: their worst action is over-referral, not doing nothing.
    for grade in (0, 1):
        assert RETINA.cost[grade].argmax() != RETINA.no_action_index, f"grade {grade} should peak at Urgent Referral"

    # A spec whose declaration contradicts the rule must not build at all.
    try:
        _build_spec(
            name="broken",
            labels=("a", "b"),
            actions=(NO_ACTION, "act"),
            cost=np.array([[0.0, 1.0], [5.0, 0.0]]),
            critical_labels=(),  # label "b" peaks at No Action, so this is wrong
        )
    except ValueError as error:
        assert "uniquely worst" in str(error), f"the error must name the rule it enforces: {error}"
    else:
        raise AssertionError("a spec whose critical labels contradict the worst-action rule must raise")
    print("OK: critical labels follow the uniquely-worst-action rule on every spec")


def check_max_utility_is_not_an_accident() -> None:
    """max_utility comes from the declared utility scale, not from cost.max() by coincidence.

    Kiyani et al.'s Table 1 has a minimum utility of 0, so max_utility and cost.max() agree for the
    covid spec -- a coincidence of that table, not a law. A spec whose utility bottoms out strictly
    above 0 has max_utility > cost.max(), and deriving the offset from the costs would silently
    disagree with the published scale.
    """
    assert np.isclose(COVID.max_utility, 10.0), f"covid max_utility must be the table's 10.0, got {COVID.max_utility}"
    # The covid utility table must round-trip: utility = max_utility - cost, exactly as published.
    expected_utility = np.array(
        [[10.0, 2.0, 2.0, 4.0], [0.0, 10.0, 3.0, 7.0], [0.0, 3.0, 10.0, 8.0], [1.0, 4.0, 4.0, 10.0]]
    )
    assert np.allclose(COVID.utility, expected_utility), f"covid utility does not round-trip:\n{COVID.utility}"

    # A utility-primary spec with a strictly positive minimum utility: cost.max() is 3, but the
    # declared scale is 8, and the utilities must come back on THAT scale.
    utility = np.array([[8.0, 5.0], [6.0, 7.0]])
    spec = _build_spec(
        name="offset",
        labels=("x", "y"),
        actions=(NO_ACTION, "act"),
        cost=8.0 - utility,
        # cost is [[0, 3], [2, 1]]: label "y" peaks at No Action, so the rule selects it and the
        # declaration must say so -- _build_spec refuses any other answer.
        critical_labels=("y",),
        max_utility=8.0,
    )
    assert np.isclose(spec.max_utility, 8.0), f"declared max_utility was overwritten: {spec.max_utility}"
    assert spec.max_utility > spec.cost.max(), "this fixture is meant to have max_utility above cost.max()"
    assert np.allclose(spec.utility, utility), f"utility did not round-trip on the offset scale:\n{spec.utility}"

    # And a declared offset BELOW the largest cost is rejected, since it implies a negative utility.
    try:
        _build_spec(
            name="negative",
            labels=("x",),
            actions=(NO_ACTION, "act"),
            cost=np.array([[0.0, 5.0]]),
            critical_labels=(),  # row [0, 5] peaks at "act", not No Action, so nothing is critical
            max_utility=2.0,
        )
    except ValueError as error:
        assert "max_utility" in str(error), f"unexpected message: {error}"
    else:
        raise AssertionError("a max_utility below the largest cost must raise")
    print("OK: max_utility follows the declared utility scale, not cost.max() by accident")


def _random_hull(batch: int, num_members: int, num_classes: int, rng: np.random.Generator) -> TorchConvexCredalSet:
    """A random convex-hull credal set with `num_members` vertices per instance."""
    probs = rng.dirichlet(np.ones(num_classes), size=(batch, num_members))
    return TorchConvexCredalSet(
        tensor=TorchProbabilityCategoricalDistribution(torch.tensor(probs, dtype=torch.float64))
    )


def _random_intervals(batch: int, num_classes: int, rng: np.random.Generator) -> TorchProbabilityIntervalsCredalSet:
    """A random probability-interval credal set, rescued from the one failure mode the sampler hits.

    The shrink below only fixes sum(lower) > 1. The other unreachable case, sum(upper) < 1, is not
    corrected here because these widths around a simplex centre cannot produce it: upper >= centre
    pointwise, so the upper bounds always sum to at least 1. Reachability is therefore a property of
    this particular construction, not something the helper enforces in general.
    """
    centre = rng.dirichlet(np.ones(num_classes), size=batch)
    width = rng.uniform(0.02, 0.15, size=(batch, num_classes))
    lower = np.clip(centre - width, 0.0, 1.0)
    upper = np.clip(centre + width, 0.0, 1.0)
    # Shrink the lower bounds if they oversubscribe the simplex, so the box always contains
    # at least one distribution (the water-filling assumes a non-empty, reachable box).
    oversubscribed = lower.sum(axis=1, keepdims=True)
    lower = np.where(oversubscribed > 1.0, lower / oversubscribed * 0.95, lower)
    return TorchProbabilityIntervalsCredalSet(
        torch.tensor(lower, dtype=torch.float64), torch.tensor(upper, dtype=torch.float64)
    )


def _oracle_hull_worst_case(hull: TorchConvexCredalSet, cost: np.ndarray, v: float) -> np.ndarray:
    """Brute-force worst case over hull vertices: the sup of a linear objective is at a vertex."""
    vertices = hull.tensor.probabilities.numpy()  # (B, M, K)
    excess = np.clip(cost - v, 0.0, None)  # (K, A)
    return np.einsum("bmk,ka->bma", vertices, excess).max(axis=1)  # (B, A)


def _oracle_interval_worst_case(
    intervals: TorchProbabilityIntervalsCredalSet, cost: np.ndarray, v: float, rng: np.random.Generator
) -> np.ndarray:
    """Two-part oracle for the interval worst case. Read this before trusting the check that uses it.

    The two halves are not equally strong, and the stronger-looking assert is the weaker evidence:

    - Random sampling inside the box is genuinely independent of the implementation, but one-sided:
      a finite sample can only under-estimate a supremum. It is therefore asserted only as
      got >= sampled. It would catch the implementation over-stating the worst case, and nothing else.
    - The greedy loop is an independent IMPLEMENTATION of the same ALGORITHM (an explicit per-action
      numpy loop against the vectorised torch sort/gather/cumsum). It supplies essentially all of the
      tightness the allclose assert checks. So that assert validates the vectorisation - the
      broadcasting, the action axis, the sorted-index bookkeeping - and NOT the correctness of
      water-filling itself. A shared misconception about the algorithm would pass both halves.

    Water-filling's optimality is argued from linearity of the objective in q (see the docstring of
    worst_case_action_cost) and was cross-checked against scipy.optimize.linprog during review; it is
    not established by this helper.
    """
    lower = intervals.lower_bounds.numpy()
    upper = intervals.upper_bounds.numpy()
    excess = np.clip(cost - v, 0.0, None)  # (K, A)
    batch = lower.shape[0]
    best = np.full((batch, excess.shape[1]), -np.inf)
    for _ in range(4000):
        raw = rng.uniform(lower, upper)
        total = raw.sum(axis=1, keepdims=True)
        q = raw / total  # renormalise; may leave the box, so filter below
        inside = np.all((q >= lower - 1e-9) & (q <= upper + 1e-9), axis=1)
        if not inside.any():
            continue
        values = q @ excess  # (B, A)
        best = np.where(inside[:, None], np.maximum(best, values), best)
    # Vertices of the box that are reachable are the real optimisers; include the greedy
    # extreme points per action so the bound is tight enough to be meaningful.
    for action in range(excess.shape[1]):
        order = np.argsort(-excess[:, action])
        q = lower.copy()
        leftover = 1.0 - lower.sum(axis=1)
        for k in order:
            take = np.minimum(upper[:, k] - lower[:, k], np.maximum(leftover, 0.0))
            q[:, k] += take
            leftover -= take
        best[:, action] = np.maximum(best[:, action], q @ excess[:, action])
    return best


def check_worst_case_hull_matches_oracle() -> None:
    """Closed-form worst-case action cost equals the vertex maximum for convex hulls."""
    rng = np.random.default_rng(0)
    for v in (0.0, 1.5, 5.0, 9.5):
        hull = _random_hull(batch=16, num_members=7, num_classes=4, rng=rng)
        got = worst_case_action_cost(hull, COVID.cost_t, v).numpy()
        expected = _oracle_hull_worst_case(hull, COVID.cost, v)
        assert np.allclose(got, expected, atol=1e-9), (
            f"hull worst case mismatch at v={v}: max diff {np.abs(got - expected).max()}"
        )
    print("OK: worst-case action cost on convex hulls matches the vertex oracle")


def check_worst_case_intervals_matches_oracle() -> None:
    """Closed-form worst-case action cost matches greedy water-filling on interval sets."""
    rng = np.random.default_rng(1)
    for v in (0.0, 2.0, 6.0):
        intervals = _random_intervals(batch=12, num_classes=4, rng=rng)
        got = worst_case_action_cost(intervals, COVID.cost_t, v).numpy()
        expected = _oracle_interval_worst_case(intervals, COVID.cost, v, rng)
        assert np.all(got >= expected - 1e-9), f"closed form below the sampled bound at v={v}"
        assert np.allclose(got, expected, atol=1e-6), (
            f"interval worst case not tight at v={v}: max diff {np.abs(got - expected).max()}"
        )
    print("OK: worst-case action cost on probability intervals matches the water-filling oracle")


def check_worst_case_singleton_is_expected_cost() -> None:
    """On a singleton credal set at v=0 the worst case is the plain expected cost."""
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(4), size=(9, 1))  # one vertex per instance
    hull = TorchConvexCredalSet(
        tensor=TorchProbabilityCategoricalDistribution(torch.tensor(probs, dtype=torch.float64))
    )
    got = worst_case_action_cost(hull, COVID.cost_t, 0.0).numpy()
    expected = probs[:, 0, :] @ COVID.cost
    assert np.allclose(got, expected, atol=1e-12), "singleton worst case is not the expected cost"
    print("OK: singleton credal set at v=0 reduces to expected cost")


def check_point_cvar_rule_is_the_truncated_best_response() -> None:
    """The ablation arm really computes argmin_a E_p[(c(y, a) - v)+], not something else.

    This is the arm that decides whether a credal win can be attributed to the credal set, so its
    objective is pinned directly against an independent numpy evaluation of the intended formula
    rather than only against the code path it is built from.

    Three things are checked, in increasing strength:

    1. singleton_credal_set produces exactly one vertex per instance, holding the input row. A
       silently broadcast or transposed wrap would still run and still yield actions.
    2. worst_case_action_cost on that set equals sum_y p(y) (c[y, a] - v)+ computed in numpy, at
       several v including v = 0 (where the truncation is inert and the objective collapses to the
       plain expected cost, so the arm must agree with best_response_actions) and v inside the
       range of the cost entries (where it must NOT).
    3. The chosen actions equal the argmin of that independent objective.

    The v > 0 case is the load-bearing one: a truncation that was dropped, applied to the wrong
    axis, or clamped after instead of before the expectation would still pass at v = 0.
    """
    rng = np.random.default_rng(11)
    probs_np = rng.dirichlet(np.ones(4), size=32)
    probs = torch.tensor(probs_np, dtype=torch.float64)

    hull = singleton_credal_set(probs)
    vertices = hull.tensor.probabilities
    assert vertices.shape == (32, 1, 4), f"singleton set must be (B, 1, K), got {tuple(vertices.shape)}"
    assert torch.allclose(vertices[:, 0, :], probs), "the single vertex is not the input distribution"

    for v in (0.0, 1.5, 3.0, 7.5):
        # The intended objective, written out from the definition and nothing else.
        expected_objective = probs_np @ np.clip(COVID.cost - v, 0.0, None)  # (B, A)
        got_objective = worst_case_action_cost(hull, COVID.cost_t, v).numpy()
        assert np.allclose(got_objective, expected_objective, atol=1e-12), (
            f"singleton worst case is not the truncated expected cost at v={v}: "
            f"max diff {np.abs(got_objective - expected_objective).max()}"
        )
        got_actions = cvar_minimax_point_actions(probs, COVID.cost_t, v).numpy()
        assert np.array_equal(got_actions, expected_objective.argmin(axis=1)), (
            f"cvar_minimax_point_actions is not the argmin of the truncated expected cost at v={v}"
        )

    # At v = 0 the truncation is inert (the costs are non-negative), so the ablation must coincide with
    # the risk-neutral best response. The arm is only a DIFFERENT decision maker because v > 0.
    at_zero = cvar_minimax_point_actions(probs, COVID.cost_t, 0.0)
    assert torch.equal(at_zero, best_response_actions(probs, COVID.cost_t)), (
        "at v=0 the point cvar rule must reduce to the risk-neutral best response"
    )

    # ... and it must genuinely diverge from best response at a v the costs actually reach, or the
    # ablation would be a relabelled copy of the base baseline and could never isolate the rule.
    diverged = any(
        not torch.equal(cvar_minimax_point_actions(probs, COVID.cost_t, v), best_response_actions(probs, COVID.cost_t))
        for v in (1.5, 3.0, 6.0, 7.5)
    )
    assert diverged, "the point cvar rule never differs from best response; the truncation is inert"

    # The construction must also compose with the calibration, which is the other half of the arm:
    # the same calibrate_action_var_threshold call, fed singleton sets, must return a usable v.
    targets = [torch.tensor(rng.integers(0, 4, size=32), dtype=torch.long)]
    v_star = calibrate_action_var_threshold([hull], targets, beta=0.25, cost=COVID.cost_t, num_grid=41)
    assert 0.0 <= v_star <= float(COVID.cost.max()), f"calibration on singleton sets returned v={v_star} out of range"
    print(f"OK: point cvar rule is the truncated best response on a singleton set (calibrated v={v_star:.3f})")


def check_action_rules_pick_the_minimiser() -> None:
    """Both action rules return the cost-minimising action on hand-computed cases.

    Every expected action here is non-zero, so an argmin/argmax flip or an off-by-one in the
    action axis fails rather than coincidentally landing on index 0. The credal case is chosen
    so the worst-case answer differs from the barycentre answer, which pins cvar_minimax_actions
    to the worst case rather than the mean.
    """
    # Point predictions: the expected cost under a certain diagnosis is that row of the cost table.
    probs = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],  # certain Pneumonia -> row [10, 0, 7, 3] -> Antibiotics
            [0.0, 0.0, 1.0, 0.0],  # certain COVID-19 -> row [10, 7, 0, 2] -> Quarantine
            [0.0, 0.0, 0.0, 1.0],  # certain Lung Opacity -> row [9, 6, 6, 0] -> Testing
            [0.5, 0.5, 0.0, 0.0],  # mixture -> [5, 4, 7.5, 4.5] -> Antibiotics
        ],
        dtype=torch.float64,
    )
    expected_best_response = torch.tensor([1, 2, 3, 1])
    got_best_response = best_response_actions(probs, COVID.cost_t)
    assert torch.equal(got_best_response, expected_best_response), (
        f"best_response_actions wrong: {got_best_response.tolist()} vs {expected_best_response.tolist()}"
    )

    # Credal case: the hull of Normal and COVID-19. Worst case per action is the elementwise max of
    # rows 0 and 2, max([0, 8, 8, 6], [10, 7, 0, 2]) = [10, 8, 8, 6], so Testing (index 3) wins.
    # The barycentre would give [5, 7.5, 4, 4] and pick Quarantine (index 2), so this case fails if
    # the rule averages instead of taking the worst case.
    vertices = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], dtype=torch.float64)
    hull = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))
    hull_costs = worst_case_action_cost(hull, COVID.cost_t, 0.0)
    assert torch.allclose(hull_costs, torch.tensor([[10.0, 8.0, 8.0, 6.0]], dtype=torch.float64)), (
        f"hull worst-case costs wrong: {hull_costs.tolist()}"
    )
    assert torch.equal(cvar_minimax_actions(hull, COVID.cost_t, 0.0), torch.tensor([3])), (
        "cvar_minimax_actions wrong on hull"
    )

    # Interval case, hand-filled at v=0: lower [0, 0, 0.5, 0], upper [0, 0, 1, 1]. Only classes
    # COVID-19 and Lung Opacity have headroom, and the leftover mass is 0.5. Pouring it onto the
    # costliest reachable class per action gives [10, 7, 3, 2], so Testing (index 3) wins.
    intervals = TorchProbabilityIntervalsCredalSet(
        torch.tensor([[0.0, 0.0, 0.5, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float64),
    )
    interval_costs = worst_case_action_cost(intervals, COVID.cost_t, 0.0)
    assert torch.allclose(interval_costs, torch.tensor([[10.0, 7.0, 3.0, 2.0]], dtype=torch.float64)), (
        f"interval worst-case costs wrong: {interval_costs.tolist()}"
    )
    assert torch.equal(cvar_minimax_actions(intervals, COVID.cost_t, 0.0), torch.tensor([3])), (
        "cvar_minimax_actions wrong on intervals"
    )

    # The rule must stay consistent with the primitive it is built on, on non-degenerate input.
    rng = np.random.default_rng(3)
    for v in (0.0, 3.0):
        random_hull = _random_hull(batch=16, num_members=5, num_classes=4, rng=rng)
        costs = worst_case_action_cost(random_hull, COVID.cost_t, v)
        assert torch.equal(cvar_minimax_actions(random_hull, COVID.cost_t, v), costs.argmin(dim=-1)), (
            f"cvar_minimax_actions disagrees with the argmin of worst_case_action_cost at v={v}"
        )
    print("OK: action rules pick the cost-minimising action")


def check_unreachable_interval_box_raises() -> None:
    """An interval box containing no distribution is rejected instead of silently mis-scored."""
    oversubscribed = TorchProbabilityIntervalsCredalSet(
        torch.tensor([[0.5, 0.5, 0.5, 0.0]], dtype=torch.float64),  # lower bounds sum to 1.5
        torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float64),
    )
    try:
        worst_case_action_cost(oversubscribed, COVID.cost_t, 0.0)
    except ValueError as error:
        assert "lower bounds" in str(error), f"unexpected message for oversubscribed box: {error}"
    else:
        raise AssertionError("a box with lower bounds summing above 1 must raise")

    undersubscribed = TorchProbabilityIntervalsCredalSet(
        torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.1, 0.1, 0.1, 0.1]], dtype=torch.float64),  # upper bounds sum to 0.4
    )
    try:
        worst_case_action_cost(undersubscribed, COVID.cost_t, 0.0)
    except ValueError as error:
        assert "upper bounds" in str(error), f"unexpected message for undersubscribed box: {error}"
    else:
        raise AssertionError("a box with upper bounds summing below 1 must raise")

    # A valid box must not trip the guard, including the exactly-tight case sum(lower) = sum(upper) = 1.
    tight = TorchProbabilityIntervalsCredalSet(
        torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float64),
        torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float64),
    )
    got = worst_case_action_cost(tight, COVID.cost_t, 0.0)
    expected = torch.tensor([[7.25, 5.25, 5.25, 2.75]], dtype=torch.float64)  # the uniform mean of the covid cost table
    assert torch.allclose(got, expected), f"tight box mis-scored: {got.tolist()}"
    print("OK: unreachable probability-interval boxes raise")


def check_realized_costs() -> None:
    """Realized cost indexes the cost matrix at (true label, chosen action)."""
    targets = np.array([0, 1, 2, 3])
    actions = np.array([0, 3, 0, 1])
    got = realized_costs(actions, targets, COVID)
    expected = np.array([0.0, 3.0, 10.0, 6.0])
    assert np.allclose(got, expected), f"realized costs wrong: {got} vs {expected}"
    print("OK: realized costs index the cost matrix correctly")


def check_metric_bundle() -> None:
    """The metric bundle reports mean cost, CVaR of cost, and the critical-mistake rate."""
    # Four instances; costs 0, 3, 10, 6 as above. One critical mistake (COVID-19 -> No Action).
    targets = np.array([0, 1, 2, 3])
    actions = np.array([0, 3, 0, 1])
    bundle = metric_bundle(actions, targets, beta=0.5, spec=COVID)
    assert np.isclose(bundle["mean_cost"], 4.75), f"mean cost wrong: {bundle['mean_cost']}"
    # CVaR_0.5 averages the worst int(0.5 * 4) = 2 costs: (10 + 6) / 2 = 8. metrics.cvar
    # truncates the tail size rather than rounding up, so int (not ceil) is the exact rule.
    assert np.isclose(bundle["cvar_cost"], 8.0), f"CVaR wrong: {bundle['cvar_cost']}"
    # Three critical-label instances (Pneumonia, COVID-19, Lung Opacity), one of which got No
    # Action (the COVID-19 case), so the CONDITIONAL rate is 1/3. The marginal rate over all four
    # instances would be 1/4, which is close enough here that the dedicated conditional-vs-marginal
    # fixture below, not this one, is what pins the distinction.
    assert np.isclose(bundle["critical_mistake_rate"], 1.0 / 3.0), (
        f"critical rate wrong: {bundle['critical_mistake_rate']}"
    )
    print("OK: metric bundle (mean, CVaR, critical-mistake rate)")


def check_critical_mistake_rate_is_conditional() -> None:
    """The reported rate conditions on the true label, and is not the marginal rate over all instances.

    Pinned on a sample where the two answers differ by a large factor, so a regression back to
    .mean() over every instance cannot pass. Ten instances: two are critical (one Pneumonia, one
    COVID-19) and eight are Normal. The Pneumonia case is sent home with No Action; the COVID-19
    case gets Testing. So:
      conditional  = 1 critical mistake / 2 critical-label instances  = 0.5
      marginal     = 1 critical mistake / 10 instances                = 0.1
    The eight non-critical instances are also given No Action, which is deliberate: a No-Action
    count over the whole sample would be 9/10, so an implementation that forgot to mask by label
    at all lands on neither number.

    The filler is Normal (label 0), the one covid label that is NOT critical -- Lung Opacity is,
    since No Action is its uniquely worst response too. Using it as filler would put four more
    critical mistakes in the numerator and destroy the factor-of-five contrast this fixture exists
    to create.
    """
    targets = np.array([1, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    actions = np.array([0, 3, 0, 0, 0, 0, 0, 0, 0, 0])

    got = critical_mistake_rate(actions, targets, COVID)
    assert np.isclose(got, 0.5), f"critical-mistake rate is not the conditional 0.5: {got}"
    marginal = float(COVID.is_critical_mistake(actions, targets).mean())
    assert np.isclose(marginal, 0.1), f"test fixture is wrong: marginal should be 0.1, got {marginal}"
    assert not np.isclose(got, marginal), "conditional and marginal rates must differ on this fixture"
    assert np.isclose(metric_bundle(actions, targets, beta=0.5, spec=COVID)["critical_mistake_rate"], got), (
        "metric_bundle does not report the conditional rate"
    )

    # A perfect method on the critical labels scores 0 even though most instances get No Action
    # (which is the correct, cost-0 action for the eight Normal cases).
    safe_actions = np.array([1, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.isclose(critical_mistake_rate(safe_actions, targets, COVID), 0.0), "no critical mistake must score 0"

    # Every critical case mishandled scores 1, which the marginal version could never reach here.
    worst_actions = np.zeros(10, dtype=np.int64)
    assert np.isclose(critical_mistake_rate(worst_actions, targets, COVID), 1.0), "all critical mistakes must score 1"

    # No critical-label instance at all: 0/0 is reported as 0.0, never NaN, or the seed mean and
    # the figure bar would silently blank out.
    # Normal only: the sole covid label for which No Action is not the worst action.
    no_critical = np.array([0, 0, 0, 0])
    empty_rate = critical_mistake_rate(np.zeros(4, dtype=np.int64), no_critical, COVID)
    assert not np.isnan(empty_rate), "an empty critical denominator must not produce NaN"
    assert empty_rate == 0.0, f"an empty critical denominator must report 0.0, got {empty_rate}"
    print("OK: critical-mistake rate is conditional on the true label (and 0/0 -> 0.0, not NaN)")


def check_critical_mistake_rates_by_label() -> None:
    """The per-label breakdown conditions on each critical label separately, and pools to the old rate.

    This is what Kiyani et al. (2025) Figure 3b plots: one bar group per critical ground-truth
    label, not a single pooled bar. The fixture is built so the pooled number HIDES a failure -- the
    method sends home every COVID-19 case but handles the other two critical labels perfectly -- and
    COVID-19 is deliberately the rarest of the three, which is precisely the situation where a
    base-rate weighted average flatters a method.
    """
    # 2 Pneumonia, 1 COVID-19, 7 Lung Opacity, 10 Normal. Only the single COVID-19 case is mishandled.
    targets = np.array([1, 1] + [2] + [3] * 7 + [0] * 10)
    actions = np.array([1, 1] + [0] + [3] * 7 + [0] * 10)

    rates = critical_mistake_rates_by_label(actions, targets, COVID)
    assert list(rates) == ["Pneumonia", "COVID-19", "Lung Opacity"], f"unexpected keys or order: {list(rates)}"
    assert np.isclose(rates["COVID-19"], 1.0), f"every COVID-19 case was sent home, expected 1.0: {rates['COVID-19']}"
    assert np.isclose(rates["Pneumonia"], 0.0), f"no Pneumonia case was sent home: {rates['Pneumonia']}"
    assert np.isclose(rates["Lung Opacity"], 0.0), f"no Lung Opacity case was sent home: {rates['Lung Opacity']}"

    # The pooled rate is 1 mistake / 10 critical instances = 0.1, which reads as a fine method while
    # the breakdown shows a 100% failure on COVID-19. That gap is the whole reason for this function.
    pooled = critical_mistake_rate(actions, targets, COVID)
    assert np.isclose(pooled, 0.1), f"pooled rate should be 1/10: {pooled}"
    assert not np.isclose(pooled, rates["COVID-19"]), "the fixture must make pooling hide the failure"

    # Pooling the breakdown back with the label base rates must reproduce the pooled rate exactly,
    # which pins the two as the same measurement at different granularities.
    counts = {COVID.labels[i]: int((targets == i).sum()) for i in COVID.critical_label_indices}
    recombined = sum(rates[label] * count for label, count in counts.items()) / sum(counts.values())
    assert np.isclose(recombined, pooled), f"base-rate weighted breakdown {recombined} != pooled {pooled}"

    # A critical label absent from the sample reports 0.0, never NaN: a NaN would propagate into the
    # seed mean and blank a bar of the figure with no error at all.
    only_pneumonia = np.array([1, 1])
    sparse = critical_mistake_rates_by_label(np.array([0, 1]), only_pneumonia, COVID)
    assert set(sparse) == {"Pneumonia", "COVID-19", "Lung Opacity"}, "every critical label must be present as a key"
    assert np.isclose(sparse["Pneumonia"], 0.5), f"1 of 2 Pneumonia cases sent home: {sparse['Pneumonia']}"
    for absent in ("COVID-19", "Lung Opacity"):
        assert sparse[absent] == 0.0 and not np.isnan(sparse[absent]), f"{absent} with no instances must be 0.0"

    # And it works on the non-square retina spec, whose three critical grades are 2, 3 and 4.
    retina_targets = np.array([2, 3, 4, 0])
    retina_rates = critical_mistake_rates_by_label(np.array([0, 3, 0, 0]), retina_targets, RETINA)
    assert list(retina_rates) == ["Moderate", "Severe", "Proliferative"], f"retina keys wrong: {list(retina_rates)}"
    assert np.isclose(retina_rates["Moderate"], 1.0) and np.isclose(retina_rates["Proliferative"], 1.0), (
        f"both sent-home grades must score 1.0: {retina_rates}"
    )
    assert np.isclose(retina_rates["Severe"], 0.0), f"the urgently referred severe case must score 0: {retina_rates}"
    print("OK: per-label critical-mistake rates (and pooling hides what they show)")


def check_cvar_degeneracy_predicate() -> None:
    """The degeneracy predicate fires exactly when CVaR_beta stops carrying tail information.

    The cost matrix has a zero diagonal, so metrics.cvar's tail fills with zeros once it is at
    least as large as the number of non-zero costs, and CVaR collapses to mean / beta. Both sides
    of that boundary are pinned here, and each assertion is cross-checked against what metrics.cvar
    ACTUALLY returns rather than only against the predicate's own arithmetic -- the claim being made
    is about the metric, not about the inequality.

    N = 1000 throughout, so int(beta * N) is exact and the boundary lands on a whole instance.
    """
    n = 1000
    rng = np.random.default_rng(21)

    def costs_with(nonzero: int) -> np.ndarray:
        c = np.zeros(n)
        c[:nonzero] = rng.choice([2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 10.0], size=nonzero)
        return c

    # --- degenerate: tail strictly larger than the non-zero count ---
    degenerate = costs_with(50)  # 5% non-zero, tail at beta=0.1 is 100
    assert cvar_is_degenerate(degenerate, 0.1), "50 non-zero costs in a tail of 100 must be degenerate"
    assert np.isclose(cvar(degenerate, 0.1), degenerate.mean() / 0.1), (
        "the degenerate case must actually equal mean / beta, or the predicate is testing the wrong thing"
    )

    # --- the EXACT-EQUALITY edge: tail == non-zero count is still degenerate ---
    edge = costs_with(100)  # 10% non-zero, tail at beta=0.1 is exactly 100
    assert int(0.1 * n) == int((edge > 0).sum()) == 100, "fixture is not on the equality boundary"
    assert cvar_is_degenerate(edge, 0.1), "tail exactly equal to the non-zero count must count as degenerate"
    assert np.isclose(cvar(edge, 0.1), edge.mean() / 0.1), "the equality edge must still equal mean / beta"

    # --- non-degenerate: one more non-zero cost than the tail can hold ---
    non_degenerate = costs_with(101)
    assert not cvar_is_degenerate(non_degenerate, 0.1), "101 non-zero costs in a tail of 100 must NOT be degenerate"
    assert not np.isclose(cvar(non_degenerate, 0.1), non_degenerate.mean() / 0.1), (
        "the non-degenerate case must differ from mean / beta"
    )

    # --- the same costs become non-degenerate at a smaller beta ---
    assert cvar_is_degenerate(degenerate, 0.1), "precondition"
    assert not cvar_is_degenerate(degenerate, 0.02), "5% non-zero must be a real tail at beta=0.02"
    assert not np.isclose(cvar(degenerate, 0.02), degenerate.mean() / 0.02), "beta=0.02 must measure a genuine tail"

    # --- the separate hole: int(beta * N) == 0 makes metrics.cvar return the plain mean ---
    # numpy's sorted[-0:] is the WHOLE array, not an empty one. A predicate testing only
    # tail_size >= nnz would call this NON-degenerate, which is exactly backwards.
    tiny = np.array([0.0, 0.0, 0.0, 10.0])
    assert int(0.1 * tiny.size) == 0, "fixture does not exercise the zero-tail case"
    assert np.isclose(cvar(tiny, 0.1), tiny.mean()), "metrics.cvar on a zero-size tail returns the plain mean"
    assert cvar_is_degenerate(tiny, 0.1), "a zero-size tail must be reported as degenerate"

    # --- a perfect method (no non-zero cost) is degenerate at every beta ---
    perfect = np.zeros(n)
    assert cvar_is_degenerate(perfect, 0.02), "all-zero costs have no tail at any beta"
    assert max_nondegenerate_beta(perfect) == 0.0, "no beta can measure a tail that does not exist"

    # --- max_nondegenerate_beta is the exclusive bound the predicate implies ---
    for nonzero in (37, 100, 233):
        c = costs_with(nonzero)
        bound = max_nondegenerate_beta(c)
        assert np.isclose(bound, nonzero / n), f"bound should be nnz/N, got {bound} for {nonzero}"
        assert cvar_is_degenerate(c, bound), "the bound itself is EXCLUSIVE, so beta = bound is degenerate"
        assert not cvar_is_degenerate(c, bound - 1e-9), "just below the bound must be non-degenerate"

    assert nonzero_cost_fraction(costs_with(250)) == 0.25, "non-zero fraction wrong"
    assert nonzero_cost_fraction(np.zeros(0)) == 0.0, "empty cost vector must not divide by zero"
    print("OK: CVaR degeneracy predicate fires exactly at the zero-padding boundary")


def check_degenerate_regime_calibration_collapses_to_v_zero() -> None:
    """In the degenerate regime the calibration must return v = 0, i.e. plain Gamma-minimax.

    The claim being tested: when the non-zero costs are too few to fill the tail, there is nothing
    beyond the mean for the truncation to hedge against, so the Rockafellar-Uryasev objective
    F(v) = v + (1 / (beta n)) sum_i (c_i(v) - v)+ should be increasing in v from 0 and the grid
    search should land on v = 0. Holding the actions fixed, dF/dv = 1 - #{c_i > v} / (beta n), and
    #{c_i > 0} = nnz < beta n is exactly the degeneracy condition, so the slope at 0+ is positive.

    That argument assumes the actions do not move with v, which they do (the rule's argmin depends
    on v), so it is a prediction rather than a proof -- hence this check on an actual fixture.

    The fixture is a near-certain credal set: both vertices concentrate on the true label, so the
    rule is right on most instances and the realized costs are mostly the zero diagonal. beta = 0.1
    with n = 200 gives a tail of 20, against about 10 non-zero costs, so the regime is degenerate.
    """
    batch = 100
    # Both vertices near the Normal corner: the rule confidently takes the cost-0 action.
    vertices = np.full((batch, 2, 4), 0.01)
    vertices[:, 0, 0] = 0.97
    vertices[:, 1, 0] = 0.97
    vertex_tensor = torch.tensor(vertices, dtype=torch.float64)
    hulls = [TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertex_tensor)) for _ in range(2)]
    # 95% Normal (which the rule gets right, cost 0), 5% not (the only non-zero costs).
    labels = np.zeros(batch, dtype=np.int64)
    labels[:5] = 2  # COVID-19 cases the confident rule will mishandle
    targets = [torch.tensor(labels), torch.tensor(labels)]

    beta = 0.1
    v = calibrate_action_var_threshold(hulls, targets, beta=beta, cost=COVID.cost_t, num_grid=101)

    actions = cvar_minimax_actions(hulls[0], COVID.cost_t, v).numpy()
    realized = COVID.cost[np.concatenate([labels, labels]), np.concatenate([actions, actions])]
    assert cvar_is_degenerate(realized, beta), "fixture is not in the degenerate regime; the check is vacuous"

    assert np.isclose(v, 0.0), f"degenerate regime should calibrate to v=0 (Gamma-minimax), got v={v}"

    # ... and v = 0 really is Gamma-minimax on the untruncated expected cost, so the rule has
    # collapsed to the risk-neutral-over-the-credal-set decision maker, as the theory predicts.
    gamma_minimax = (
        np.einsum("bmk,ka->bma", hulls[0].tensor.probabilities.numpy(), COVID.cost).max(axis=1).argmin(axis=1)
    )
    assert np.array_equal(actions, gamma_minimax), "at the calibrated v the rule is not Gamma-minimax"
    print(f"OK: degenerate regime calibrates to v={v:.1f} and collapses to Gamma-minimax")


def _ambiguous_hull(batch: int) -> TorchConvexCredalSet:
    """A credal set undecided between Pneumonia and COVID-19 for every instance.

    The two vertices sit at the near-pure corners of the Pneumonia/COVID-19 face of the simplex,
    so the set is wide along exactly the axis the cost matrix punishes: Antibiotics costs 7 under
    COVID-19 and Quarantine costs 7 under Pneumonia, while Testing costs at most 3 under either.
    """
    vertices = np.zeros((batch, 2, 4))
    vertices[:, 0, 1] = 0.98
    vertices[:, 0, 2] = 0.02
    vertices[:, 1, 1] = 0.02
    vertices[:, 1, 2] = 0.98
    return TorchConvexCredalSet(
        tensor=TorchProbabilityCategoricalDistribution(torch.tensor(vertices, dtype=torch.float64))
    )


def _brute_force_objective(
    rep_outs: list[TorchConvexCredalSet], targets: list[torch.Tensor], beta: float, grid: np.ndarray
) -> np.ndarray:
    """Independent numpy evaluation of F(v) = v + (1 / (beta n)) sum_i (cost(a_i(v), y_i) - v)+ on a grid."""
    total = sum(int(t.numel()) for t in targets)
    values = []
    for v in grid:
        excess = 0.0
        for rep_out, batch_targets in zip(rep_outs, targets, strict=True):
            actions = cvar_minimax_actions(rep_out, COVID.cost_t, float(v)).numpy()
            realized = COVID.cost[batch_targets.numpy(), actions]
            excess += float(np.clip(realized - v, 0.0, None).sum())
        values.append(v + excess / (beta * total))
    return np.array(values)


def check_calibration_recovers_known_optimum() -> None:
    """Calibration returns the analytically derived minimiser on a val set built to have one.

    Construction: 64 instances, every one ambiguous between Pneumonia and COVID-19 (see
    _ambiguous_hull), delivered as two batches so the per-batch accumulation and the pooled sample
    size are both exercised. beta = 0.25, so beta * n = 16. The credal set is identical for every
    instance, so the chosen action is too, and the labels alone shape the cost sample. The labels
    include some Normal and Lung Opacity cases the credal set does not cover, which is what gives
    the realized cost under Testing a spread (6 / 3 / 2 / 0 for the four labels) rather than the two
    values the credal face alone would produce. That spread matters: it puts the optimum strictly
    inside the region where the truncation is still active, so the leading v term of the objective
    is load-bearing. Without it the minimiser would sit exactly where the excess first vanishes and
    an objective missing the v term would agree by accident.

    The rule's action is a step function of v, because the truncated worst case (cost - v)+ collapses
    action by action as v passes each cost entry:
      v < 7:       Testing (index 3) alone has worst-case truncated cost 0.
      7 <= v < 10: Antibiotics, Quarantine and Testing all reach 0 and the argmin tie breaks to the
                   lowest index, Antibiotics.
      v >= 10:     every action reaches 0 -> No Action.

    On v < 7 the realized costs are 8 x 6, 24 x 3, 24 x 2, 8 x 0, so F(v) = v + (1/16) sum (c_i - v)+
    is piecewise linear with slope 1 - #{c_i > v} / 16:
      v in [0, 2): 56 above         -> slope -2.5 (the 8 Lung Opacity zeros are not above v).
      v in [2, 3): 32 above         -> slope -1.
      v in [3, 6): 8 above          -> slope +0.5.
      v in [6, 7): none above       -> slope +1.
    The unique minimiser is therefore v = 3, with F(3) = 3 + 8 * 3 / 16 = 4.5. Neither later regime
    competes: at v = 7 the rule switches to Antibiotics and F(7) = 7 + 8 * (8 - 7) / 16 = 7.5, and
    F(10) = 10.

    v = 3 is an entry of the cost table, so it is one of the EXACT candidates the calibration now
    always evaluates, and the assertions below hold for any num_grid at all -- including num_grid=0,
    which is asserted separately in check_calibration_uses_exact_cost_candidates. The grid of 41
    points on [0, 10] happens to contain 3.0 too, which is what let the old grid-only search find it.
    """
    beta, num_grid = 0.25, 41
    hulls = [_ambiguous_hull(32), _ambiguous_hull(32)]
    # Per batch: 4 Normal, 12 Pneumonia, 12 COVID-19, 4 Lung Opacity -> 8 / 24 / 24 / 8 overall.
    half = torch.tensor([0] * 4 + [1] * 12 + [2] * 12 + [3] * 4, dtype=torch.long)
    targets = [half.clone(), half.clone()]

    v = calibrate_action_var_threshold(hulls, targets, beta=beta, cost=COVID.cost_t, num_grid=num_grid)
    assert np.isclose(v, 3.0), f"calibration missed the analytic optimum v=3: got {v}"

    # The chosen threshold must actually put the rule in the risk-averse regime, i.e. Testing for
    # every instance. This is the behavioural claim; the value of v alone would not pin it down.
    actions = cvar_minimax_actions(hulls[0], COVID.cost_t, v).numpy()
    testing = COVID.actions.index("Testing")
    assert np.all(actions == testing), f"calibrated v={v} does not select Testing: {np.unique(actions)}"

    # Cross-check the search itself against an independent numpy evaluation of the same objective:
    # the returned v must be the grid argmin, and its objective value the analytic 3.0.
    grid = np.linspace(0.0, float(COVID.cost.max()), num_grid)
    objectives = _brute_force_objective(hulls, targets, beta, grid)
    assert np.isclose(grid[objectives.argmin()], 3.0), f"brute-force argmin is not 3: {grid[objectives.argmin()]}"
    assert np.isclose(objectives.min(), 4.5), f"objective at the optimum is not 4.5: {objectives.min()}"
    assert np.isclose(objectives[np.isclose(grid, v)][0], objectives.min()), "returned v is not the grid minimiser"

    # The calibration objective is Rockafellar-Uryasev, whose minimum over v is the empirical CVaR;
    # the driver reports metrics.cvar, the mean of the worst int(beta * n) costs. Here beta * n = 16
    # is an integer, so the two definitions must agree exactly. They diverge only through that
    # rounding, which is why this equality is asserted rather than merely bounded.
    realized = COVID.cost[torch.cat(targets).numpy(), np.concatenate([actions, actions])]
    assert np.isclose(cvar(realized, beta), 4.5), f"metrics.cvar disagrees with the R-U optimum: {cvar(realized, beta)}"
    print(f"OK: calibration recovers the analytic optimum (v={v:.3f}, F(v)={objectives.min():.3f})")


def check_calibration_uses_exact_cost_candidates() -> None:
    """The calibration finds the integer optimum with NO grid at all, and a grid alone would miss it.

    The bug this pins: holding the actions fixed, F(v) = v + (1/(beta n)) sum_i (c_i - v)+ is
    piecewise linear with kinks exactly at the realized cost values, so its minimiser is always a
    cost value -- but every entry of both cost tables is an integer, and linspace(0, 10, 200) has
    step 10/199 with 199 prime, so it contains no interior integer whatsoever. The configs ran at
    num_grid = 200 and calibrated to v = 4.0201 where the true minimiser was 4.0. Evaluating F on
    the exact candidate set removes the grid as a source of that error entirely.

    Uses the same fixture as check_calibration_recovers_known_optimum, whose analytic optimum is
    v = 3 with F(3) = 4.5.
    """
    beta = 0.25
    hulls = [_ambiguous_hull(32), _ambiguous_hull(32)]
    half = torch.tensor([0] * 4 + [1] * 12 + [2] * 12 + [3] * 4, dtype=torch.long)
    targets = [half.clone(), half.clone()]

    # num_grid=0 disables the residual grid, so ONLY the exact candidates are searched. The analytic
    # optimum must still come out exactly: this is the check that the exact set is doing the work.
    for num_grid in (0, 1):
        v_exact = calibrate_action_var_threshold(hulls, targets, beta=beta, cost=COVID.cost_t, num_grid=num_grid)
        assert np.isclose(v_exact, 3.0), f"exact candidates alone must recover v=3 at num_grid={num_grid}: {v_exact}"

    # The returned v must be exactly 3.0, not merely close: with the exact candidates it is a cost
    # table entry, so any floating drift means the candidate set is not being used as intended.
    v = calibrate_action_var_threshold(hulls, targets, beta=beta, cost=COVID.cost_t, num_grid=200)
    assert v == 3.0, f"calibration must return the cost value 3.0 exactly, got {v!r}"

    # The regression itself: a grid of 200 on [0, 10] contains no interior integer, so a grid-ONLY
    # search cannot reach 3.0 and lands beside it. This is what the old implementation did.
    grid_only = np.linspace(0.0, 10.0, 200)
    interior_integers = [x for x in grid_only[1:-1] if float(x).is_integer()]
    assert not interior_integers, f"the 200-point grid was expected to miss every interior integer: {interior_integers}"
    objectives = _brute_force_objective(hulls, targets, beta, grid_only)
    grid_best = float(grid_only[objectives.argmin()])
    assert not np.isclose(grid_best, 3.0, atol=1e-9), f"the grid-only search should miss 3.0, got {grid_best}"
    assert abs(grid_best - 3.0) < 0.1, f"the grid-only miss should be a near miss, not a different regime: {grid_best}"

    # ... and the exact answer is genuinely better than the best the grid could do, so this is a
    # real improvement in the objective and not just a prettier number.
    exact_objective = _brute_force_objective(hulls, targets, beta, np.array([3.0]))[0]
    assert exact_objective <= objectives.min(), f"F(3)={exact_objective} should beat the grid best {objectives.min()}"
    assert np.isclose(exact_objective, 4.5), f"F(3) must be the analytic 4.5, got {exact_objective}"

    # Every distinct cost entry is a candidate, on the non-square retina table too: its minimiser
    # must likewise be a table value rather than something between two grid points.
    rng = np.random.default_rng(17)
    retina_targets = [torch.tensor(rng.integers(0, 5, size=64), dtype=torch.long)]
    retina_hulls = [_random_hull(batch=64, num_members=4, num_classes=5, rng=rng)]
    v_retina = calibrate_action_var_threshold(retina_hulls, retina_targets, beta=0.25, cost=RETINA.cost_t, num_grid=0)
    assert v_retina in set(RETINA.cost.ravel().tolist()) | {0.0}, (
        f"with no grid the retina calibration must return a cost table entry, got {v_retina}"
    )
    print("OK: calibration evaluates the exact cost-value candidates (v=3.0 exactly, with no grid)")


def check_v_zero_is_gamma_minimax() -> None:
    """At v = 0 the rule minimises the worst-case *expected* cost (Gamma-minimax)."""
    rng = np.random.default_rng(4)
    hull = _random_hull(batch=20, num_members=5, num_classes=4, rng=rng)
    got = cvar_minimax_actions(hull, COVID.cost_t, 0.0).numpy()
    vertices = hull.tensor.probabilities.numpy()
    expected = np.einsum("bmk,ka->bma", vertices, COVID.cost).max(axis=1).argmin(axis=1)
    assert np.array_equal(got, expected), "v=0 does not reduce to Gamma-minimax on expected cost"
    print("OK: v=0 reduces to Gamma-minimax on expected cost")


def check_no_action_dominance_threshold() -> None:
    """Above the reported threshold, NO credal set can make our rule choose the no-action option.

    This is the guard on the headline critical-mistake metric. The metric counts critical-label
    cases that receive the no-action option, so it silently becomes a constant zero -- for every
    method, however good or bad its uncertainty estimates -- once truncation puts some other action's
    column elementwise below the no-action column. The threshold function is what lets the re-scorer
    and the comparison figure warn instead of reporting that zero as a result.

    Two things are checked. The reported value is pinned per spec, because it is a property of a
    frozen cost table and a change in it means the table moved. And the CLAIM behind the value is
    verified directly against the rule, on random credal sets: just below the threshold some
    no-action decision survives, just at it none does. Pinning the number alone would not catch a
    sign error in the dominance test itself.
    """
    expected = {
        COVID.name: 6.0,
        TRIAGE_SPEC.name: 6.0,
    }
    for name, want in expected.items():
        spec = ALL_SPECS[name]
        got = no_action_dominated_threshold(spec)
        # None means no finite threshold dominates the no-action option; for these specs one exists,
        # and the narrowing keeps the float-typed uses below (the rule call and the `< got` witness).
        assert got is not None, f"{name}: no-action is never dominated, but a finite threshold {want} was expected"
        assert got == want, f"{name}: no-action dominance threshold moved from {want} to {got}"

        rng = np.random.default_rng(11)
        # Wide random hulls, so the rule has every opportunity to pick the no-action option: if it
        # never does at v = threshold, that is the dominance and not a lack of candidates.
        hull = _random_hull(batch=400, num_members=6, num_classes=len(spec.labels), rng=rng)
        at = cvar_minimax_actions(hull, spec.cost_t, got).numpy()
        assert not (at == spec.no_action_index).any(), (
            f"{name}: {spec.actions[spec.no_action_index]!r} was still chosen at v = {got}, so it is "
            f"not dominated there and the threshold is wrong."
        )

        # Minimality, tested with the RIGHT witness. Just below the threshold the option must still
        # be reachable, or the reported value is not the smallest such v and every warning built on
        # it fires too late. The witness must be a point mass, not another random hull: dominance is
        # sufficient for unreachability but not necessary, so a random hull can easily fail to pick
        # the no-action option well below the threshold without that saying anything about
        # dominance. A singleton credal set on the label whose no-action cost is smallest is the
        # distribution most favourable to the option, so if ANY credal set can make our rule choose
        # it, that one can.
        below = sorted(v for v in {0.0} | {float(x) for x in spec.cost.flatten()} if v < got)[-1]
        favourable = torch.zeros(1, len(spec.labels), dtype=torch.float64)
        favourable[0, int(spec.cost[:, spec.no_action_index].argmin())] = 1.0
        chosen = int(cvar_minimax_point_actions(favourable, spec.cost_t, below)[0])
        assert chosen == spec.no_action_index, (
            f"{name}: even the most favourable point mass chose {spec.actions[chosen]!r} rather than "
            f"{spec.actions[spec.no_action_index]!r} at v = {below}, below the reported threshold "
            f"{got}, so {got} is not the smallest threshold at which the option leaves the action set."
        )
    print("OK: the no-action dominance threshold matches the rule's actual behaviour")


def check_synthetic_triage_spec() -> None:
    """Pin the structural invariants the synthetic triage experiment's design depends on.

    The triage matrix (src/cost_sensitive/cost_sensitive_triage.py) was designed under the
    two-threshold rule the covid_xray_v4 block states: the no-action dominance threshold D and
    the collapse threshold v* = min_a max_y cost must be separated, with a wide band below D in
    which the critical-mistake metric measures the method. D = 6 is pinned in
    check_no_action_dominance_threshold with the other specs; here the remaining invariants are
    pinned, because a casual edit to the table would silently move the experiment out of its
    informative regime while every number still computed.
    """
    spec = get_cost_spec_by_name("synthetic_triage")
    collapse = float(spec.cost.max(axis=0).min())
    assert collapse == 8.0, f"synthetic_triage: collapse threshold v* moved from 8 to {collapse}"
    assert spec.critical_label_indices == (2,), (
        f"synthetic_triage: critical labels moved from (Sepsis,) to "
        f"{tuple(spec.labels[i] for i in spec.critical_label_indices)}. Sepsis-only is load-bearing: "
        f"the catastrophe count the experiment reports is 'sepsis sent home', nothing else."
    )
    # The Flu row's deliberate 4-4 tie is what keeps Flu non-critical; a well-meaning "fix" of
    # either entry would add Flu to the critical set and change what the headline metric counts.
    assert spec.cost[1, spec.no_action_index] == spec.cost[1, 2] == 4.0, (
        "synthetic_triage: the Flu row's No Action / ICU tie at 4 is broken; Flu would become (or "
        "stop being excluded as) a critical label under the uniquely-worst-action rule."
    )
    print("OK: the synthetic triage spec's structural invariants hold (v* = 8, Sepsis-only critical)")


def check_fixed_v_scoring_bypasses_calibration() -> None:
    """--fixed-v scores at the threshold it is given, and at v = 0 the ablation IS best response.

    The fixed-threshold path exists so the credal set is the only difference between arms, and the
    v = 0 coincidence is the property that makes the resulting figure interpretable: the supremum
    over a singleton credal set is an expectation, so a point predictor under our rule at v = 0
    decides exactly as the risk-neutral best response does. If that ever stopped holding, the
    ablation bar would silently stop being the reference the figure claims it is.
    """
    rng = np.random.default_rng(23)
    probs = torch.tensor(rng.dirichlet(np.ones(4), size=300), dtype=torch.float64)
    ours_at_zero = cvar_minimax_point_actions(probs, COVID.cost_t, 0.0).numpy()
    risk_neutral = best_response_actions(probs, COVID.cost_t).numpy()
    assert np.array_equal(ours_at_zero, risk_neutral), (
        "our rule on a singleton credal set at v = 0 no longer coincides with best response, so the "
        "comparison figure's ablation bar is not the reference it is labelled as."
    )
    print("OK: at v = 0 our rule on a point predictor coincides with best response")


def check_non_square_spec_runs_end_to_end() -> None:
    """The 5x4 RetinaMNIST spec works through every rule and the metric bundle.

    This is the check the whole dataset generalisation turns on. Every other check in this file
    uses the 4x4 COVID matrix, where a transposed axis, a reduction over the wrong dimension or a
    K-vs-A mix-up is invisible because the two lengths coincide. Here K = 5 and A = 4, so any such
    slip is a shape error rather than a wrong number.
    """
    cost_np, cost_t = RETINA.cost, RETINA.cost_t
    num_labels, num_actions = cost_np.shape

    # --- point predictions: shapes and hand-computed argmins ---
    probs = torch.eye(num_labels, dtype=torch.float64)  # one certain distribution per grade
    expected_cost = probs.numpy() @ cost_np
    assert expected_cost.shape == (num_labels, num_actions), "expected-cost shape is not (B, A)"
    got_br = best_response_actions(probs, cost_t)
    # Certain grade g -> the zero-cost action of row g: 0, 1, 2, 3, 3.
    assert torch.equal(got_br, torch.tensor([0, 1, 2, 3, 3])), f"best response on the 5x4 spec wrong: {got_br.tolist()}"

    # --- singleton credal sets: the ablation arm on a non-square table ---
    for v in (0.0, 2.5, 6.0):
        objective = probs.numpy() @ np.clip(cost_np - v, 0.0, None)
        got = worst_case_action_cost(singleton_credal_set(probs), cost_t, v).numpy()
        assert got.shape == (num_labels, num_actions), f"worst-case shape is not (B, A) at v={v}: {got.shape}"
        assert np.allclose(got, objective, atol=1e-12), f"truncated expected cost wrong on the 5x4 spec at v={v}"
        assert np.array_equal(cvar_minimax_point_actions(probs, cost_t, v).numpy(), objective.argmin(axis=1)), (
            f"cvar_minimax_point_actions wrong on the 5x4 spec at v={v}"
        )

    # --- convex hull: worst case must beat the barycentre, on a case where they disagree ---
    # Hull of grade 0 ([0, 2, 4, 7]) and grade 2 ([6, 3, 0, 3]). The elementwise max is
    # [6, 3, 4, 7], so the worst case picks Recheck 6mo (index 1). The barycentre is
    # [3, 2.5, 2, 5] and would pick Refer (index 2), so an averaging bug fails here.
    vertices = torch.zeros((1, 2, num_labels), dtype=torch.float64)
    vertices[0, 0, 0] = 1.0
    vertices[0, 1, 2] = 1.0
    hull = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))
    hull_costs = worst_case_action_cost(hull, cost_t, 0.0)
    assert torch.allclose(hull_costs, torch.tensor([[6.0, 3.0, 4.0, 7.0]], dtype=torch.float64)), (
        f"hull worst-case costs wrong on the 5x4 spec: {hull_costs.tolist()}"
    )
    assert torch.equal(cvar_minimax_actions(hull, cost_t, 0.0), torch.tensor([1])), (
        "cvar_minimax_actions on the 5x4 spec did not take the worst case"
    )

    # --- random hulls and interval boxes against the oracles, at K = 5 ---
    rng = np.random.default_rng(31)
    for v in (0.0, 3.0, 7.5):
        random_hull = _random_hull(batch=12, num_members=6, num_classes=num_labels, rng=rng)
        got = worst_case_action_cost(random_hull, cost_t, v).numpy()
        assert got.shape == (12, num_actions), f"hull worst-case shape wrong at v={v}: {got.shape}"
        assert np.allclose(got, _oracle_hull_worst_case(random_hull, cost_np, v), atol=1e-9), (
            f"5x4 hull worst case disagrees with the vertex oracle at v={v}"
        )
        assert torch.equal(cvar_minimax_actions(random_hull, cost_t, v), torch.tensor(got.argmin(axis=1))), (
            f"5x4 cvar_minimax_actions is not the argmin of the primitive at v={v}"
        )
    for v in (0.0, 2.0, 5.0):
        intervals = _random_intervals(batch=10, num_classes=num_labels, rng=rng)
        got = worst_case_action_cost(intervals, cost_t, v).numpy()
        assert got.shape == (10, num_actions), f"interval worst-case shape wrong at v={v}: {got.shape}"
        assert np.allclose(got, _oracle_interval_worst_case(intervals, cost_np, v, rng), atol=1e-6), (
            f"5x4 interval worst case disagrees with the water-filling oracle at v={v}"
        )

    # --- calibration accepts the non-square table and returns a threshold inside its range ---
    calibration_targets = [torch.tensor(rng.integers(0, num_labels, size=12), dtype=torch.long)]
    v_star = calibrate_action_var_threshold(
        [_random_hull(batch=12, num_members=4, num_classes=num_labels, rng=rng)],
        calibration_targets,
        beta=0.25,
        cost=cost_t,
        num_grid=21,
    )
    assert 0.0 <= v_star <= float(cost_np.max()), f"5x4 calibration returned v={v_star} outside [0, 10]"

    # --- metric bundle on the 5x4 spec ---
    # One instance per grade, all sent home: costs 0, 3, 6, 9, 10. Both critical grades (3, 4)
    # got No Action, so the conditional rate is 2/2 = 1. CVaR_0.4 averages the worst
    # int(0.4 * 5) = 2 costs, (10 + 9) / 2 = 9.5, and the mean is 28 / 5 = 5.6.
    targets = np.arange(num_labels)
    actions = np.zeros(num_labels, dtype=np.int64)
    assert np.allclose(realized_costs(actions, targets, RETINA), [0.0, 3.0, 6.0, 9.0, 10.0]), (
        "realized costs wrong on the 5x4 spec"
    )
    bundle = metric_bundle(actions, targets, beta=0.4, spec=RETINA)
    assert np.isclose(bundle["mean_cost"], 5.6), f"5x4 mean cost wrong: {bundle['mean_cost']}"
    assert np.isclose(bundle["cvar_cost"], 9.5), f"5x4 CVaR wrong: {bundle['cvar_cost']}"
    assert np.isclose(bundle["critical_mistake_rate"], 1.0), (
        f"5x4 critical rate wrong: {bundle['critical_mistake_rate']}"
    )
    assert np.isclose(critical_mistake_rate(actions, targets, RETINA), 1.0), "5x4 conditional rate wrong"

    # The spec argument must actually reach the arithmetic: the same inputs scored against the
    # COVID table give a different answer, so a helper that ignored its spec would fail here.
    covid_bundle = metric_bundle(actions[:4], targets[:4], beta=0.4, spec=COVID)
    assert not np.isclose(covid_bundle["mean_cost"], bundle["mean_cost"]), (
        "the covid and retina specs score identically; the spec argument is not being used"
    )
    print("OK: the non-square 5x4 spec runs through every rule and the metric bundle")


def check_registry_lookup() -> None:
    """get_cost_spec_by_name resolves the registered spec and fails loudly, and helpfully, otherwise."""
    assert get_cost_spec_by_name("synthetic_triage") is TRIAGE_SPEC, "synthetic_triage must resolve to TRIAGE_SPEC"

    try:
        get_cost_spec_by_name("no_such_spec")
    except KeyError as error:
        message = str(error)
        assert "no_such_spec" in message, f"the error must name the spec asked for: {message}"
        assert "synthetic_triage" in message, f"the error must list the available specs: {message}"
    else:
        raise AssertionError("an unregistered spec name must raise KeyError")
    print("OK: registry lookup resolves the registered spec and lists the registry on failure")


def check_cost_tables_are_read_only() -> None:
    """Every spec's numpy tables reject in-place writes.

    The driver evaluates many methods in one process and the rules truncate costs, so an in-place
    write into a shared table would corrupt every method scored afterwards with no error at all.
    Asserted per spec, not only for one of them, because a new spec built by hand rather
    than through the factory would miss the freeze.
    """
    for name, spec in ALL_SPECS.items():
        assert not spec.cost.flags.writeable, f"{name}: cost table is writeable"
        assert not spec.utility.flags.writeable, f"{name}: utility table is writeable"
        try:
            spec.cost[0, 0] = 99.0
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name}: writing into the cost table must raise")
        # Fancy indexing still returns copies, so ordinary downstream use is unaffected.
        sample = spec.cost[np.array([0, 1]), np.array([0, 1])]
        assert sample.flags.writeable, f"{name}: fancy-indexed copies should stay writeable"
        # utility is the exact mirror of cost, so the two cannot drift.
        assert np.allclose(spec.utility, spec.max_utility - spec.cost), f"{name}: utility is not max_utility - cost"
        # >=, not ==: max_utility equals cost.max() only when the utility scale bottoms out at 0,
        # which is true of both current specs but is a property of their tables, not a law. Anything
        # below cost.max() would imply a negative utility and is rejected by _build_spec.
        assert spec.max_utility >= spec.cost.max(), (
            f"{name}: max_utility {spec.max_utility} is below the largest cost {spec.cost.max()}"
        )
        assert torch.allclose(spec.cost_t, torch.tensor(spec.cost, dtype=torch.float64)), (
            f"{name}: the torch mirror disagrees with the numpy table"
        )
    print("OK: every spec's cost and utility tables are read-only")


def check_multi_beta_calibration_matches_one_beta_at_a_time() -> None:
    """Sweeping betas in one call returns exactly what calibrating each level separately would.

    This is the claim the whole beta sweep rests on. calibrate_action_var_thresholds evaluates the
    Rockafellar-Uryasev excess ONCE per candidate v and reuses it at every level, on the argument
    that excess(v) = sum_i (cost(a_i(v), y_i) - v)+ does not mention beta -- beta enters only as the
    scalar 1 / (beta n) multiplying it. If that were wrong, the sweep would produce a plausible
    figure in which every level but one was calibrated against the wrong tail, with nothing to say
    so. So the amortised answers are pinned against the singular function, level by level, and
    EXACTLY: both search the same candidate set with the same tie-breaking, so any difference at all
    is a bug rather than floating-point drift.

    The fixture is the ambiguous-hull validation set from check_calibration_recovers_known_optimum,
    whose optimum at beta = 0.25 is the analytic v = 3, extended over levels that span the regimes:
    a tiny beta (where the linear v term dominates and drives v down), the analytic level, and
    beta = 1 (where the objective is the plain mean excess).
    """
    betas = [0.02, 0.1, 0.25, 0.5, 1.0]
    hulls = [_ambiguous_hull(32), _ambiguous_hull(32)]
    half = torch.tensor([0] * 4 + [1] * 12 + [2] * 12 + [3] * 4, dtype=torch.long)
    targets = [half.clone(), half.clone()]

    swept = calibrate_action_var_thresholds(hulls, targets, betas, cost=COVID.cost_t, num_grid=41)
    assert len(swept) == len(betas), f"one threshold per beta expected, got {len(swept)} for {len(betas)} betas"
    for beta, v_swept in zip(betas, swept, strict=True):
        v_single = calibrate_action_var_threshold(hulls, targets, beta=beta, cost=COVID.cost_t, num_grid=41)
        assert v_swept == v_single, f"sweep and single calibration disagree at beta={beta}: {v_swept} vs {v_single}"

    # The known level must still land on its analytic optimum when reached through the sweep, or the
    # agreement above could be two implementations sharing one mistake.
    assert np.isclose(swept[betas.index(0.25)], 3.0), f"the sweep missed the analytic v=3 at beta=0.25: {swept}"

    # The sweep must actually SEPARATE the levels on this fixture, or it would pass vacuously with a
    # calibration that ignored beta entirely -- which is precisely the bug being ruled out.
    assert len(set(swept)) > 1, f"every level calibrated to the same v; the fixture cannot detect the bug: {swept}"

    # Order is by argument, not sorted by beta or by v: a consumer zips these against its own beta
    # list, so a silent reordering would mislabel every point of the sweep figure.
    reversed_betas = list(reversed(betas))
    reversed_swept = calibrate_action_var_thresholds(hulls, targets, reversed_betas, cost=COVID.cost_t, num_grid=41)
    assert reversed_swept == list(reversed(swept)), f"result order does not follow the betas argument: {reversed_swept}"

    # Duplicates are permitted and agree with themselves; an empty list and an out-of-range level
    # are errors rather than a silently empty or nonsensical sweep.
    assert calibrate_action_var_thresholds(hulls, targets, [0.25, 0.25], cost=COVID.cost_t, num_grid=41) == [3.0, 3.0]
    for bad in ([], [0.0], [0.25, 1.5], [-0.1]):
        try:
            calibrate_action_var_thresholds(hulls, targets, bad, cost=COVID.cost_t, num_grid=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"betas={bad} must raise")
    print(f"OK: multi-beta calibration matches one-beta-at-a-time (betas {betas} -> v {swept})")


def main() -> None:
    """Run every check; raises AssertionError on the first failure."""
    check_critical_mistake()
    check_critical_labels_follow_the_worst_action_rule()
    check_max_utility_is_not_an_accident()
    check_worst_case_hull_matches_oracle()
    check_worst_case_intervals_matches_oracle()
    check_worst_case_singleton_is_expected_cost()
    check_point_cvar_rule_is_the_truncated_best_response()
    check_action_rules_pick_the_minimiser()
    check_unreachable_interval_box_raises()
    check_realized_costs()
    check_metric_bundle()
    check_critical_mistake_rate_is_conditional()
    check_critical_mistake_rates_by_label()
    check_cvar_degeneracy_predicate()
    check_calibration_recovers_known_optimum()
    check_calibration_uses_exact_cost_candidates()
    check_multi_beta_calibration_matches_one_beta_at_a_time()
    check_degenerate_regime_calibration_collapses_to_v_zero()
    check_v_zero_is_gamma_minimax()
    check_no_action_dominance_threshold()
    check_synthetic_triage_spec()
    check_fixed_v_scoring_bypasses_calibration()
    check_non_square_spec_runs_end_to_end()
    check_registry_lookup()
    check_cost_tables_are_read_only()
    print("\nAll cost-sensitive self-tests passed.")


if __name__ == "__main__":
    main()
