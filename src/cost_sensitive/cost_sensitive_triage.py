"""Synthetic cost-sensitive triage: credal RL multinomial vs an MLE, hedging both uncertainties.

A 2-D triage problem (Healthy / Flu / Sepsis -> No Action / Treat Flu / ICU, spec
``synthetic_triage`` in cost_sensitive/costs.py) with the two uncertainty channels the
cost-sensitive goal names, each under its own knob:

- ALEATORIC: "silent sepsis" -- a sepsis component supported inside the Healthy cluster, so
  even the Bayes posterior is genuinely mixed there. Knob: its share ``s_silent`` of the
  sepsis mixture. Irreducible by data.
- EPISTEMIC: a deployment subgroup (region B) whose sepsis presentation is missing from the
  training archive. Knob: the keep fraction ``f_b`` of Sepsis-B training rows. Reducible by
  data, which is the diagnostic signature: the epistemic increment vanishes as f_b -> 1 (the
  aleatoric channel remains, so the arms converge in the increment, not to each other).

The MLE (a small MLP) sees Healthy-B but (almost) no Sepsis-B, so its softmax saturates to
Healthy across region B; a saturated-wrong point prediction assigns ~zero truncated
worst-case cost to the catastrophe at EVERY threshold v, so no rule on the singleton can
hedge it (the two-region PoC's provable failure, now in action space). The credal RL
multinomial arm collects ~zero evidence mass at Sepsis-B, its set is near-vacuous, and the
worst-case rule prices No Action at ~(20 - v) and hedges into the ICU. Silent sepsis is
hedged through the kernel vote's honest local class mix where the overfit MLP saturates.

Run:  uv run python src/cost_sensitive/cost_sensitive_triage.py [--smoke] [--plot-only]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.font_manager import FontProperties

import joblib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from matplotlib.transforms import ScaledTranslation
from probly.method.credal_wrapper import credal_wrapper
from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet
from probly.representer import representer
from sqwash import SuperquantileSmoothReducer
from torch.utils.data import DataLoader, TensorDataset

from cost_sensitive.costs import get_cost_spec_by_name
from cost_sensitive.evaluate import metric_bundle, realized_costs
from cost_sensitive.rules import (
    best_response_actions,
    calibrate_action_var_thresholds,
    cvar_minimax_actions,
    singleton_credal_set,
)
from cost_sensitive.scoring import no_action_dominated_threshold

# The paper figures' BAR GEOMETRY is imported from the runtime-bars figure rather than restated, so
# the two cannot drift: same rounded bars, same tint for the lesser version of a hue, same y-number
# nudge. These names are private to that module, which is the price of it being the one place that
# geometry is defined -- copying them here is what the import exists to prevent.
from experiments.plot_runtime_bars import (  # noqa: PLC2701
    _BAR_RADIUS_IN,
    _TEST_TINT,
    _TICK_NUDGE_PT,
    _axes_aspect,
    _lighten,
    _rounded_bar,
)
from methods import Exp3Sampler, IndexedDataset
from methods.credal_rl_multinomial import (
    compute_evidence,
    compute_multinomial_bounds,
    fit_evidence_reference,
    select_whitening,
)
from paths import PLOTS_PATH, RESULTS_PATH

# Everything about type, strokes and method colours comes from the paper's shared style module: the
# three Fira faces, the Sizes every font size and stroke width derives from, one colour per method.
from plotting.paper_style import (
    BASELINE_BLUE,
    COLUMN_WIDTH,
    FP_LIGHT,
    FP_REGULAR,
    FP_SEMIBOLD,
    METHODS,
    OUR_RULE,
    OURS,
    TEXT_WIDTH,
    Sizes,
    band_style,
    legend_below,
    line_style,
    ours_label,
    render_for_print,
    save,
    set_xlabel,
    set_ylabel,
    style_axes,
    use_fira_mathtext,
)

SPEC = get_cost_spec_by_name("synthetic_triage")
DOMINANCE_V = no_action_dominated_threshold(SPEC)  # 6.0, pinned by the selftest

# Config lives in a namespace (not module constants) so --smoke can shrink it in place. Workers
# receive it as an explicit argument: loky re-imports this module, so parent-side mutations of a
# module global would be silently lost (the two_region_cvar lesson).
C = SimpleNamespace(
    seeds=list(range(1, 11)),
    # The two knob sweeps. Epistemic: f_b at the headline silent share. Aleatoric: s_silent at
    # f_b = 1 (NO epistemic gap, so any win there is the aleatoric channel alone).
    f_b_sweep=[0.0, 0.05, 0.25, 0.5, 1.0],
    silent_sweep=[0.0, 0.125, 0.25, 0.5],
    headline_f_b=0.05,
    headline_s_silent=0.25,
    n_train=4000,
    n_val=2000,
    n_test=8000,
    taus=[0.01, 0.025, 0.05, 0.1, 0.2],
    headline_tau=0.1,
    # MLP (two_region_cvar conventions).
    hidden=64,
    epochs=150,
    batch_size=128,
    lr=1e-3,
    sqwash_smoothing=1.0,
    # CreWra ensemble size (configs/method/credal_wrapper.yaml default). Its interval width is
    # an estimator artifact of the member spread (E[max - min] ~ 3.08 sigma at 10 draws).
    n_members=10,
    # CreRL multinomial. num_neighbors stays at the PIPELINE DEFAULT 200 (the covid
    # experiments' value): the neighbourhood size is part of the method, not an experiment
    # knob; only the bandwidth is tuned, as in the covid bandwidth sweep.
    # bandwidth_anchor = num_neighbors ("neighborhood") on purpose: the reference size varies
    # across cells by design, which is exactly the regime the legacy rank-1 anchor mis-scales
    # (see fit_evidence_reference).
    num_neighbors=200,
    # 0.30 at the 200-NN anchor reproduces the ABSOLUTE evidence-mass profile the
    # 100-neighbour draft was tuned to (healthy ~8, Sepsis-B ~0.2). The absolute level
    # matters, not only the contrast: 0.25 gave a sharper 55x contrast but ~2.5x lower masses,
    # the mid-mass components' sets widened past the hedging threshold, and calibration
    # collapsed to the blanket policy again (measured in the probe; the same knife edge as the
    # alpha tuning).
    bandwidth_mult=0.30,
    # alpha 0.95, the paper-wide operating point (the OOD and shift experiments present the
    # method at 0.95). The credal radius is -log(alpha) / mass and the nuisance floor caps
    # in-distribution mass at ~5-18, so large alphas are required at all: 0.5 hedged ~13% of
    # HEALTHY patients into the ICU and collapsed calibration to the blanket policy, 0.8 sat
    # on a knife edge between v = 4 and v = 8, 0.9 was stable, and 0.95 (probed at 3 seeds
    # over f_b {0, 0.05, 1}) keeps v = 4 everywhere, lowers CVaR by 0.2-0.5 -- converging to
    # the point-rule ablation at full data, as it should when the posterior is trustworthy --
    # and still hedges region B (mass ~0.2 => radius ~0.26, wide enough at v = 4), at a
    # moderate catastrophe increase (headline ~40 vs ~26). The "alpha activates once mass
    # drops" effect the covid epistemic sweep predicted is the design's main dial.
    alpha=0.95,
    ridge=1e-2,
    # The lift must be LARGE relative to the standardized data radius (~6). The method
    # L2-normalises internally, and with a small lift the embedding becomes angle-dominated:
    # Healthy-B and Sepsis-B are nearly collinear from the origin (146 vs 142 degrees), so a
    # small-lift normalisation collapsed them onto one direction and Sepsis-B COLLECTED mass
    # from Healthy-B (measured: mass 10-15 at lift 3 vs 0.19 at lift 20, with in-distribution
    # mass unchanged). At lift L the map is x -> x / sqrt(|x|^2 + L^2) ~ x / L: near-affine,
    # so kernel distances reflect the true geometry.
    lift=20.0,
    # Whitening is fixed OFF (exponent 0, pure rotation), not "auto": these features are the
    # observation space itself, and whitening exists to repair encoder-scale artifacts. Here
    # the within-class covariance is dominated by the cluster layout plus low-variance nuisance
    # dims, so full whitening AMPLIFIES the nuisance directions to unit scale and blurs the
    # silent pocket (measured: silent-pocket sepsis vote 0.54 at gamma 0 vs 0.32 at gamma 1;
    # the auto health check does not see this because global vote accuracy barely moves).
    whitening=0.0,
    calib_grid=100,
    map_seed=1,
    map_grid=200,
    n_jobs=-1,
    # DGP: deployment prior Healthy 0.55 (A 0.85 / B 0.15), Flu 0.25, Sepsis 0.20 (overt
    # 1 - s_silent - b_share, silent s_silent, B b_share).
    p_healthy=0.55,
    p_flu=0.25,
    p_sepsis=0.20,
    healthy_b_share=0.15,
    sepsis_b_share=0.15,
    # Nuisance dimensions, identical N(0, sigma_nuisance) for EVERY component, so they carry no
    # label information and the Bayes-optimal decision is unchanged. They exist to make the
    # epistemic gap PERSIST under partial data: in 2-D a handful of archived Sepsis-B rows let
    # the MLP carve out the whole compact cluster (the first full run's gap collapsed by
    # f_b = 0.25, per-seed bimodal 28-vs-270). With few samples in 8-D the MLP cannot know the
    # nuisance coordinates of those samples are irrelevant, so its learned sepsis region hugs
    # them in all dimensions and subgroup patients with other nuisance values still get sent
    # home -- the high-dimensional small-sample mechanism of the covid rare-class experiment,
    # reproduced deliberately.
    # NOTE: sigma_nuisance only has an effect because standardiser() uses a GLOBAL std -- under
    # per-dimension standardisation it is exactly compensated and silently becomes a no-op (see
    # standardiser). With the global std, raw sigma 1.0 equals the within-cluster signal noise:
    # enough to defeat the MLP's few-sample generalisation in region B, small against the
    # region separations (5-13), so the kernel's contrast survives.
    dim_nuisance=6,
    sigma_nuisance=1.0,
)

COMPONENTS = ("healthy_a", "healthy_b", "flu", "sepsis_overt", "sepsis_silent", "sepsis_b")
COMPONENT_LABEL = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)  # component -> class label
# Silent sepsis is a COMPACT POCKET on the healthy cluster's edge, not a copy of the healthy
# blob: dead-centre placement makes blanket ICU (CVaR exactly 9) beat every selective policy,
# so calibration collapses to v = v* and the zero catastrophe counts measure the table -- the
# first smoke run failed exactly that way. On the edge, the ambiguous region is small: hedging
# it sacrifices few healthy patients, so the calibrated optimum stays selective (v < D).
# Sepsis-B is BROAD (signal std 1.5 vs healthy-B's 0.8): the subgroup's presentation is
# diverse, so the few archived rows a partial sweep keeps cover only a sliver of it. Together
# with the nuisance dimensions this is what keeps the epistemic gap open at f_b = 0.25-0.5.
COMPONENT_MEAN = np.array([[0.0, 0.0], [-6.0, 4.0], [3.0, 0.0], [3.0, 3.0], [-1.8, 1.8], [-9.0, 7.0]], dtype=np.float64)
COMPONENT_STD = np.array([1.0, 0.8, 1.0, 0.8, 0.5, 1.5], dtype=np.float64)

ARM_ORDER = (
    "mle|best_response",
    "sqwash|best_response",
    "adacvar|best_response",
    "mle|cvar_minimax_point",
    "crewra|cvar_minimax",
    "crerl|cvar_minimax",
)
ARM_LABELS = {
    "mle|best_response": "MLE",
    "sqwash|best_response": "SQwash",
    "adacvar|best_response": "AdaCVaR",
    "mle|cvar_minimax_point": "MLE + ours",
    "crewra|cvar_minimax": "CreWra + ours",
    "crerl|cvar_minimax": "CreRL + ours",
}
ARM_COLORS = {
    "mle|best_response": "tab:red",
    "sqwash|best_response": "tab:purple",
    "adacvar|best_response": "tab:cyan",
    "mle|cvar_minimax_point": "tab:orange",
    "crewra|cvar_minimax": "tab:brown",
    "crerl|cvar_minimax": "tab:blue",
}
RESULTS_FILE = "cost_sensitive_triage.pkl"

# Which arms carry OUR decision rule. The paper figure puts these in their own panel, mirroring the
# runtime figure's split: baselines on the left in blue, ours on the right in crimson, so the
# comparison reads as "the field" vs "us" rather than as six peers in a row. ARM_COLORS above is
# the per-method palette the diagnostic sweeps still use; the paper figure deliberately overrides it
# with the two-hue opposition, because there the group is the message, not the individual method.
OURS_ARMS = ("mle|cvar_minimax_point", "crewra|cvar_minimax", "crerl|cvar_minimax")
# The paper figure's PANELS, in reading order. Three groups, not two: the middle one holds the arms
# that apply OUR decision rule to somebody else's uncertainty (the MLE singleton, the CreWra
# ensemble box), so the rule is held fixed across the middle and right panels and only the set
# construction varies. That is exactly the R4 claim -- the credal set's own contribution, not the
# rule's -- made a fact of the layout instead of a sentence in the caption.
# SQwash sits LAST in the argmax panel, after AdaCVaR, so that the two arms the panel's badge
# describes stand together on its left. Under aligned-tau training SQwash is no longer a
# catastrophe-heavy arm (29 sepsis-sent-home per split at the headline cell, against MLE's 238 and
# AdaCVaR's 224 -- fewer even than ours), so leaving it between them would put a counterexample in
# the middle of the group the badge generalises over. The order is presentational only; ARM_ORDER
# above still drives every diagnostic figure and the verification table.
PAPER_PANELS: tuple[tuple[str, ...], ...] = (
    ("mle|best_response", "adacvar|best_response", "sqwash|best_response"),
    ("mle|cvar_minimax_point", "crewra|cvar_minimax"),
    ("crerl|cvar_minimax",),
)
# The figure in the paper that draws the cell the line curves shade, as the tag in that band names
# it: the appendix bar figure at 6 recorded sepsis cases, panel (a) of its pair. A figure NUMBER in
# a figure goes stale whenever the paper's figures move, so it lives here, once.
LINE_BAR_FIGURE_TAG = "Fig. 17a"
# One sticker per panel, read together as the 2x2 the experiment is built to show: catastrophes
# high or low, average cost high or low. The baselines are CHEAP on average and catastrophic in the
# tail; the rule on somebody else's uncertainty buys the tail back but pays ~3x the average cost;
# ours buys the tail back at close to the baselines' average cost.
#
# The right panel says "few", not "none". Ours commits ~37 catastrophic decisions per 8000-patient
# test split at the headline cell -- far fewer than MLE's ~238, but not zero. "No catastrophic
# decisions" would be a claim the figure directly under it refutes: the dark block is visibly
# non-empty in every panel.
PAPER_STICKERS: tuple[str, ...] = (
    "many catastrophic decisions",
    "few catastrophes,\nbut high mean cost",
    "few catastrophes,\nat low mean cost",
)
# Which SLOTS of its panel each badge is centred over, or None for the whole panel. The argmax badge
# covers slots 0-1 (MLE, AdaCVaR) and deliberately NOT slot 2: aligned-tau SQwash commits ~29
# catastrophic decisions at the headline cell, fewer than ours' ~37, so a badge centred over all
# three would assert "many" of an arm that has few. The badge is an annotation over the bars it
# describes, and its POSITION is the scope of its claim -- centring it would silently widen that
# scope back to the arm it excludes. This is why it sits left of the panel centre; it matches the
# submitted main paper.
PAPER_STICKER_SLOTS: tuple[tuple[int, ...] | None, ...] = ((0, 1), None, None)
# The stickers above are measured at the HEADLINE cell (f_b = 5%) and do not survive a move along
# the sweep, so a render at another cell must bring its own. At f_b = 25% the headline's "but high
# average cost" becomes false -- the CVaR-minimax arms fall from ~4.6 to ~1.8, against baselines of
# 1.3-2.5 -- and the point ablation still commits ~83 catastrophic decisions against MLE's ~101, so
# "few catastrophes" no longer describes that panel either.
#
# The catastrophe clauses compare against MLE and AdaCVaR, the two arms the left badge names: 83
# and 36 against their 101 and 120 for the middle panel (fewer), and ours' 26 against both (fewest).
#
# "at low mean cost" is worded to match the headline figure's badges rather than to rank the arms,
# and it is the loosest of the claims on either figure. At 25% the whole row is compressed into
# 1.33-2.49 -- MLE 1.33, AdaCVaR 1.62, the CVaR-minimax arms 1.80 and 1.83, ours 1.87, SQwash 2.49
# -- so "low" here means low on the cost scale the shared axis shows (all six bars sit in the
# bottom third of a 0-6.4 axis), NOT lowest among the arms: ours is in fact the second highest of
# the six, under SQwash alone. The claim carries at the headline cell, where the middle panel pays
# ~4.6 and makes ours' ~1.9 low by comparison; at 25% that contrast is gone because the premium is
# gone, which is the finding this figure exists to show.
PAPER_STICKERS_FB25: tuple[str, ...] = (
    "many catastrophic decisions",
    "fewer catastrophes\nat low mean cost",
    "fewest catastrophes\nat low mean cost",
)
# What a row of the paper figure can show; see fig_paper's `rows` for what each one draws. Each kind
# maps to the QUANTITY it measures, which is what decides whether two rows share a y scale: "cvar"
# and "tail" are the same number drawn two ways (solid, then stacked), as are "mean_bar" and "mean",
# and rows of one quantity must share a scale or the same value renders at two heights. Rows of
# DIFFERENT quantities must not: mean cost is ~4 against CVaR's ~11, and a shared scale flattens it.
ROW_QUANTITY = {"cvar": "cvar", "tail": "cvar", "mean_bar": "mean", "mean": "mean"}
ROW_KINDS = tuple(ROW_QUANTITY)
# The kinds drawn as SOLID bars (with the +-1 se whisker and the per-seed dots), and the record
# metric each one aggregates. The stacked kinds are keyed off the same metric for their y scale.
ROW_METRIC = {"cvar": "cvar_cost", "tail": "cvar_cost", "mean_bar": "mean_cost", "mean": "mean_cost"}
SOLID_KINDS = ("cvar", "mean_bar")
# The headline arm inside that panel: full crimson and a semibold tick label, exactly the
# emphasis the runtime figure gives CreWare. The other two are ABLATIONS of it (the point-prediction
# singleton and the ensemble set), so they take the same hue at the tint the runtime figure uses for
# its second quantity -- one colour at two intensities, saying "same family, lesser version".
FLAGSHIP_ARM = "crerl|cvar_minimax"

# The two hues of the bar figures, both from the paper's shared style: every baseline wears the one
# blue, ours wears CreWare's red.
OURS_COLOR = METHODS[OURS].color
# The baselines of Figure 1: every arm but ours recedes into one gray, the value of plot_runtime_bars'
# _BASELINE_GRAY. Set this to BASELINE_BLUE for the render in which they wear the appendix bars' blue.
INTRO_BASELINE_COLOR = "#9b9b9b"
# The width the paper includes each figure at. A figure's type is sized so that it PRINTS like the
# reference figure's at this width (paper_style.render_for_print), whatever width it is drawn at.
INTRO_PRINT_WIDTH = 0.99 * COLUMN_WIDTH  # Figure 1, one column of the main text
PAPER_PRINT_WIDTH = 0.9 * TEXT_WIDTH  # the two-row bar figure, alone across the appendix page
APPENDIX_PRINT_WIDTH = 0.49 * TEXT_WIDTH  # the three-row bar figures, two side by side
LINE_PRINT_WIDTH = 0.95 * TEXT_WIDTH
# The type scale the bar figures' inch geometry was tuned at: 7 pt axis labels, i.e. Sizes(0.7).
# Whatever in a layout exists to hold text (a margin, a label offset, a badge's headroom) is that
# many inches times the ratio of the type actually drawn to this.
_TUNED_SCALE = 0.7
# Within-group dash/marker pairs for the line figure, cycled by an arm's position in its panel
# group. Kept short so the pattern is legible at a ~1.5in-wide axes: long dashes read as solid.
_LINE_STROKES = ((None, "o"), ((0, (2.2, 1.1)), "s"), ((0, (0.9, 0.9)), "^"))


def component_probs(cfg: SimpleNamespace, s_silent: float) -> npt.NDArray[np.float64]:
    """Deployment mixture weights over COMPONENTS for a given silent-sepsis share."""
    overt = 1.0 - s_silent - cfg.sepsis_b_share
    if overt < 0.0:
        raise ValueError(f"s_silent={s_silent} leaves a negative overt share.")
    return np.array(
        [
            cfg.p_healthy * (1.0 - cfg.healthy_b_share),
            cfg.p_healthy * cfg.healthy_b_share,
            cfg.p_flu,
            cfg.p_sepsis * overt,
            cfg.p_sepsis * s_silent,
            cfg.p_sepsis * cfg.sepsis_b_share,
        ]
    )


def sample_deployment(
    rng: np.random.Generator, cfg: SimpleNamespace, n: int, s_silent: float
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Draw n deployment instances: features (n, 2 + dim_nuisance), class labels, components.

    The first two coordinates are the signal plane the components live in; the remaining
    dim_nuisance coordinates are the same N(0, sigma_nuisance) noise for every component (see
    the config comment for why they exist).
    """
    comp = rng.choice(len(COMPONENTS), size=n, p=component_probs(cfg, s_silent))
    signal = COMPONENT_MEAN[comp] + rng.normal(0.0, 1.0, size=(n, 2)) * COMPONENT_STD[comp, None]
    nuisance = rng.normal(0.0, cfg.sigma_nuisance, size=(n, cfg.dim_nuisance))
    x = np.concatenate([signal, nuisance], axis=1)
    return x.astype(np.float32), COMPONENT_LABEL[comp], comp.astype(np.int64)


def make_data(
    rng: np.random.Generator, cfg: SimpleNamespace, f_b: float, s_silent: float
) -> dict[str, tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """Train / val / test splits as (x, y, component).

    Train is the BIASED ARCHIVE: a deployment draw with each Sepsis-B row kept only with
    probability f_b (dropped, not resampled -- the archive is simply missing those patients).
    Val and test are full deployment draws: calibration gets to see region-B catastrophes, so a
    point arm that still fails there fails because its prediction cannot express the risk, not
    because the experiment starved its calibration (the covid rare-class honesty argument).
    """
    x, y, comp = sample_deployment(rng, cfg, cfg.n_train, s_silent)
    keep = (comp != COMPONENTS.index("sepsis_b")) | (rng.uniform(size=len(comp)) < f_b)
    out = {"train": (x[keep], y[keep], comp[keep])}
    out["val"] = sample_deployment(rng, cfg, cfg.n_val, s_silent)
    out["test"] = sample_deployment(rng, cfg, cfg.n_test, s_silent)
    return out


class MLP(nn.Module):
    """Small (2 + dim_nuisance) -> hidden -> hidden -> 3 ReLU classifier, the MLE arm."""

    def __init__(self, dim_in: int, hidden: int) -> None:
        """Build the layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, dim_in) -> (B, 3) logits."""
        return self.net(x)


def _fit(
    cfg: SimpleNamespace,
    model: nn.Module,
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.int64],
    reducer: nn.Module | None = None,
) -> nn.Module:
    """Adam CE loop; reducer replaces the mean over per-instance losses (mirrors two_region_cvar)."""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=cfg.batch_size, shuffle=True
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            opt.zero_grad()
            per_instance = F.cross_entropy(model(xb), yb, reduction="none")
            loss = per_instance.mean() if reducer is None else reducer(per_instance)
            loss.backward()
            opt.step()
    model.eval()
    return model


def fit_mle(cfg: SimpleNamespace, x: npt.NDArray[np.float32], y: npt.NDArray[np.int64]) -> nn.Module:
    """The risk-neutral MLE arm: mean cross entropy."""
    return _fit(cfg, MLP(x.shape[1], cfg.hidden), x, y)


def fit_sqwash(cfg: SimpleNamespace, x: npt.NDArray[np.float32], y: npt.NDArray[np.int64], tail: float) -> nn.Module:
    """The training-time-CVaR baseline: SMOOTHED superquantile reduction at tail fraction `tail`.

    Trained ALIGNED, one model per evaluated tau with tail = tau (see the config comment). The
    smooth reducer is Laguel et al.'s recommended variant; at 10 seeds it was statistically
    indistinguishable from the plain one at tail 0.5, so the choice is a defensive default,
    not a tuned advantage. Smoothing coefficient 1.0 (the package default; 0.1 was unstable).
    """
    reducer = SuperquantileSmoothReducer(superquantile_tail_fraction=tail, smoothing_coefficient=cfg.sqwash_smoothing)
    return _fit(cfg, MLP(x.shape[1], cfg.hidden), x, y, reducer=reducer)


def fit_adacvar(
    cfg: SimpleNamespace,
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.int64],
    rng: np.random.Generator,
    tail: float,
) -> nn.Module:
    """The adaptive-sampling CVaR baseline: mean-CE steps on Exp3-sampled minibatches.

    Mirrors two_region_cvar._fit_adacvar / train_funcs.train_adacvar, trained ALIGNED at
    tail fraction `tail` = the evaluated tau.
    """
    model = MLP(x.shape[1], cfg.hidden)
    n = len(x)
    horizon = max(1, cfg.epochs * (n // cfg.batch_size))
    alpha = tail
    eta = math.sqrt((1.0 / alpha) * math.log(1.0 / alpha) / horizon) if alpha < 1.0 else 0.0
    sampler = Exp3Sampler(num_actions=n, batch_size=cfg.batch_size, alpha=alpha, eta=eta, rng=rng)
    loader = DataLoader(
        IndexedDataset(TensorDataset(torch.from_numpy(x), torch.from_numpy(y))),  # ty: ignore[invalid-argument-type]
        batch_sampler=sampler,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb, idx in loader:
            opt.zero_grad()
            per_instance = F.cross_entropy(model(xb), yb, reduction="none")
            per_instance.mean().backward()
            opt.step()
            sampler.update(per_instance.detach().numpy(), idx.numpy())
        sampler.normalize()
    model.eval()
    return model


# NOTE on the ABSENT "SQwash with costs" arm (tried twice, removed on purpose -- do not
# re-add without reading this). Training a policy on the superquantile of the decision cost
# is, by the Rockafellar-Uryasev decomposition, per-instance equivalent to
# argmin_a E[(c(y, a) - v*)+] at the optimal threshold: EXACTLY the cvar_minimax_point rule
# this experiment already evaluates (MLE + ours). Gradient training on soft per-instance
# costs only approximates that closed form, and badly: at loose tail fractions the
# superquantile weights both sides of every hedging trade equally and degenerates to
# mean-cost training (measured: no extra pocket hedging even when warm-started from the
# MLE's best-response scores with an entropy bonus), and at sharp tails it collapses to
# blanket hedging. The correct implementation of the objective is the decision rule itself.


def fit_crewra(cfg: SimpleNamespace, x: npt.NDArray[np.float32], y: npt.NDArray[np.int64]) -> nn.Module:
    """The ensemble credal baseline: credal_wrapper over n_members independently trained MLPs.

    predictor_type is explicit for the same reason as in two_region_cvar: the wrapper's
    permitted-type set has several entries, so probly's isinstance-based inference never
    matches a freshly built nn.Module. The ty ignores mirror that file's -- probly types the
    wrapper as a Predictor, not the nn.Module it is at runtime.
    """
    model = credal_wrapper(
        MLP(x.shape[1], cfg.hidden),
        num_members=cfg.n_members,
        predictor_type="logit_classifier",  # ty: ignore[unknown-argument]
    )
    for member in model:
        _fit(cfg, member, x, y)  # ty: ignore[invalid-argument-type]
    return model  # ty: ignore[invalid-return-type]


@torch.no_grad()
def crewra_sets(model: nn.Module, x: npt.NDArray[np.float32]) -> TorchProbabilityIntervalsCredalSet:
    """The per-class [min, max] probability box over the ensemble members."""
    return representer(model).predict(torch.from_numpy(x))


@torch.no_grad()
def mlp_probs(model: nn.Module, x: npt.NDArray[np.float32]) -> torch.Tensor:
    """(N, 3) softmax probabilities, float64 for the rules."""
    return model(torch.from_numpy(x)).softmax(-1).double()


class CrerlFit(SimpleNamespace):
    """The fitted evidence reference: whitener, reference, targets, bandwidth, chosen gamma."""


def lift(cfg: SimpleNamespace, x: npt.NDArray[np.float32]) -> torch.Tensor:
    """Append the constant lift coordinate.

    The method L2-normalises features internally; on raw 2-D points that would collapse the
    radius (all of region B onto region A's directions). The constant third coordinate makes
    the normalisation injective on the data range, preserving the plane's geometry.
    """
    t = torch.from_numpy(x)
    return torch.cat([t, torch.full((t.shape[0], 1), float(cfg.lift))], dim=1)


def fit_crerl(cfg: SimpleNamespace, x: npt.NDArray[np.float32], y: npt.NDArray[np.int64]) -> CrerlFit:
    """Fit the multinomial-RL evidence reference on the (standardised, lifted) training features.

    The whitening exponent comes from the config ("auto" selects via the method's health check;
    a number fixes it -- this experiment fixes 0.0, see the config comment for the measurement
    behind that choice).
    """
    feats = lift(cfg, x)
    targets = torch.from_numpy(y)
    gamma = cfg.whitening
    if gamma == "auto":
        gamma, _stats = select_whitening(
            feats, targets, cfg.ridge, 3, cfg.bandwidth_mult, cfg.num_neighbors, bandwidth_anchor=cfg.num_neighbors
        )
    whitener, reference, base = fit_evidence_reference(
        feats, targets, cfg.ridge, 3, whitening=float(gamma), bandwidth_anchor=cfg.num_neighbors
    )
    return CrerlFit(
        whitener=whitener,
        reference=reference,
        targets=targets,
        bandwidth=float(cfg.bandwidth_mult * base),
        gamma=float(gamma),
    )


def crerl_evidence(
    cfg: SimpleNamespace, fit: CrerlFit, x: npt.NDArray[np.float32]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel-weighted counts (N, 3) and evidence mass (N,) of x against the reference."""
    return compute_evidence(lift(cfg, x), fit.whitener, fit.reference, fit.targets, fit.bandwidth, cfg.num_neighbors, 3)


def crerl_sets(counts: torch.Tensor, mass: torch.Tensor, alpha: float) -> TorchProbabilityIntervalsCredalSet:
    """The alpha-cut probability box of the local multinomial relative likelihood."""
    lower, upper = compute_multinomial_bounds(counts, mass, alpha)
    return TorchProbabilityIntervalsCredalSet(lower.float().clamp(0, 1), upper.float().clamp(0, 1))


def standardiser(x: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], np.float32]:
    """Train-split per-dim mean and one GLOBAL std, the normalisation every consumer shares.

    The std is deliberately a single scalar, not per-dimension. Per-dimension standardisation
    is exactly compensated by everything downstream (the MLP is scale-equivariant under Adam,
    and the kernel's Mahalanobis whitening normalises each direction anyway), so it makes
    sigma_nuisance a silent no-op -- and worse, it INFLATES the nuisance dimensions to unit
    variance while compressing the signal dimensions, whose raw std is dominated by the cluster
    layout. That inversion of the true scales is what collapsed the kernel's contrast in the
    first 8-D run. A global scalar preserves the relative geometry and only fixes the overall
    scale (so the lift constant stays meaningful).
    """
    return x.mean(axis=0), x.std() + np.float32(1e-8)


def per_component_stats(values: npt.NDArray[np.floating], comp: npt.NDArray[np.int64]) -> dict[str, float]:
    """Mean of `values` within each DGP component (NaN for components absent from the sample)."""
    return {
        name: float(values[comp == index].mean()) if bool((comp == index).any()) else float("nan")
        for index, name in enumerate(COMPONENTS)
    }


def run_cell(seed: int, f_b: float, s_silent: float, cfg: SimpleNamespace) -> dict[str, Any]:
    """Train, fit, calibrate, decide and score every arm for one (seed, f_b, s_silent) cell."""
    rng = np.random.default_rng([17, seed, int(round(f_b * 1000)), int(round(s_silent * 1000))])
    torch.manual_seed(int(rng.integers(2**31)))
    data = make_data(rng, cfg, f_b, s_silent)
    mean, std = standardiser(data["train"][0])
    xs = {split: ((x - mean) / std).astype(np.float32) for split, (x, _, _) in data.items()}
    y = {split: y_ for split, (_, y_, _) in data.items()}
    comp = {split: c for split, (_, _, c) in data.items()}

    model = fit_mle(cfg, xs["train"], y["train"])
    # SQwash and AdaCVaR are trained ALIGNED: one model per evaluated tau, at training tail
    # fraction tau. Their best-response test actions are precomputed per tau here.
    aligned_actions: dict[tuple[str, float], npt.NDArray[np.int64]] = {}
    for tau in cfg.taus:
        sq = fit_sqwash(cfg, xs["train"], y["train"], tail=tau)
        ada = fit_adacvar(cfg, xs["train"], y["train"], rng, tail=tau)
        aligned_actions[("sqwash", tau)] = best_response_actions(mlp_probs(sq, xs["test"]), SPEC.cost_t).numpy()
        aligned_actions[("adacvar", tau)] = best_response_actions(mlp_probs(ada, xs["test"]), SPEC.cost_t).numpy()
    crewra = fit_crewra(cfg, xs["train"], y["train"])
    crerl = fit_crerl(cfg, xs["train"], y["train"])
    evidence = {split: crerl_evidence(cfg, crerl, xs[split]) for split in ("val", "test")}
    probs = {("mle", split): mlp_probs(model, xs[split]) for split in ("val", "test")}
    sets = {split: crerl_sets(*evidence[split], cfg.alpha) for split in ("val", "test")}
    cw_sets = {split: crewra_sets(crewra, xs[split]) for split in ("val", "test")}

    # Decision sets per arm: each credal arm's own box, the point arm's singleton. One dict so
    # calibration and test decisions cannot use different objects.
    val_sets = {
        "mle|cvar_minimax_point": singleton_credal_set(probs[("mle", "val")]),
        "crewra|cvar_minimax": cw_sets["val"],
        "crerl|cvar_minimax": sets["val"],
    }
    test_sets = {
        "mle|cvar_minimax_point": singleton_credal_set(probs[("mle", "test")]),
        "crewra|cvar_minimax": cw_sets["test"],
        "crerl|cvar_minimax": sets["test"],
    }
    val_targets = torch.from_numpy(y["val"])

    thresholds = {
        key: calibrate_action_var_thresholds([vs], [val_targets], cfg.taus, SPEC.cost_t, cfg.calib_grid)
        for key, vs in val_sets.items()
    }

    mass_test = evidence["test"][1].numpy()
    box = sets["test"]
    width_test = (box.upper_bounds - box.lower_bounds).sum(dim=1).numpy()
    crewra_width_test = (cw_sets["test"].upper_bounds - cw_sets["test"].lower_bounds).sum(dim=1).numpy()
    mlp_conf_test = probs[("mle", "test")].max(dim=1).values.numpy()

    records: list[dict[str, Any]] = []
    for key in ARM_ORDER:
        for tau_index, tau in enumerate(cfg.taus):
            method = key.split("|")[0]
            if (method, tau) in aligned_actions:
                actions = aligned_actions[(method, tau)]
                v: float | None = None
            elif key.endswith("best_response"):
                actions = best_response_actions(probs[(method, "test")], SPEC.cost_t).numpy()
                v = None
            else:
                v = thresholds[key][tau_index]
                actions = cvar_minimax_actions(test_sets[key], SPEC.cost_t, v).numpy()
            costs = realized_costs(actions, y["test"], SPEC)
            catastrophes = SPEC.is_critical_mistake(actions, y["test"])
            records.append(
                {
                    "key": key,
                    "label": ARM_LABELS[key],
                    "seed": seed,
                    "f_b": f_b,
                    "s_silent": s_silent,
                    "tau": tau,
                    "var_threshold": v,
                    "v_in_dominated_regime": v is not None and DOMINANCE_V is not None and v >= DOMINANCE_V,
                    "actions": actions.astype(np.int8),
                    "catastrophe_count": int(catastrophes.sum()),
                    "cost_by_component": per_component_stats(costs, comp["test"]),
                    "catastrophes_by_component": {
                        name: int(catastrophes[comp["test"] == index].sum()) for index, name in enumerate(COMPONENTS)
                    },
                    **metric_bundle(actions, y["test"], beta=tau, spec=SPEC),
                }
            )

    cell = {
        "seed": seed,
        "f_b": f_b,
        "s_silent": s_silent,
        "targets_test": y["test"].astype(np.int8),
        "component_test": comp["test"].astype(np.int8),
        "whitening_gamma": crerl.gamma,
        "bandwidth": crerl.bandwidth,
        "n_train_kept": int(len(y["train"])),
        "n_train_sepsis_b": int((comp["train"] == COMPONENTS.index("sepsis_b")).sum()),
        # Mechanism diagnostics on the test split, per component: the epistemic sweep memory's
        # required check (does the set actually widen where the data is thin?).
        "mass_by_component": per_component_stats(mass_test, comp["test"]),
        "width_by_component": per_component_stats(width_test, comp["test"]),
        "crewra_width_by_component": per_component_stats(crewra_width_test, comp["test"]),
        "mlp_confidence_by_component": per_component_stats(mlp_conf_test, comp["test"]),
        "records": records,
    }

    if seed == cfg.map_seed and f_b == cfg.headline_f_b and s_silent == cfg.headline_s_silent:
        cell["map"] = build_map(cfg, model, crerl, mean, std, thresholds, data["train"])
    return cell


def build_map(
    cfg: SimpleNamespace,
    model: nn.Module,
    crerl: CrerlFit,
    mean: npt.NDArray[np.float32],
    std: np.float32,
    thresholds: dict[str, list[float]],
    train: tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> dict[str, Any]:
    """Decision-region and mechanism grids for the headline cell's map figure."""
    g = cfg.map_grid
    x1, x2 = np.linspace(-13.0, 7.0, g), np.linspace(-4.0, 11.0, g)
    xx, yy = np.meshgrid(x1, x2)
    # The map is the nuisance = 0 slice of the feature space (the nuisance mode); the signal
    # plane is where every component lives, so this is the informative cross-section.
    signal = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid = np.concatenate([signal, np.zeros((signal.shape[0], cfg.dim_nuisance))], axis=1).astype(np.float32)
    grid_std = ((grid - mean) / std).astype(np.float32)
    tau_index = cfg.taus.index(cfg.headline_tau)

    p_mle = mlp_probs(model, grid_std)
    counts, mass = crerl_evidence(cfg, crerl, grid_std)
    box = crerl_sets(counts, mass, cfg.alpha)
    v = thresholds["crerl|cvar_minimax"][tau_index]
    return {
        "x1": x1,
        "x2": x2,
        "mle_actions": best_response_actions(p_mle, SPEC.cost_t).numpy().reshape(g, g).astype(np.int8),
        "crerl_actions": cvar_minimax_actions(box, SPEC.cost_t, v).numpy().reshape(g, g).astype(np.int8),
        "crerl_v": v,
        "mass": mass.numpy().reshape(g, g),
        "width": (box.upper_bounds - box.lower_bounds).sum(dim=1).numpy().reshape(g, g),
        "train_x": train[0],
        "train_y": train[1],
        "train_comp": train[2],
    }


# ---------------------------------------------------------------------------------------------
# Aggregation, verification and figures
# ---------------------------------------------------------------------------------------------


def collect(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-cell records, attaching each cell's test targets for decision-profile plots.

    Records store the chosen actions but not the targets (those are per-cell, shared by every
    record of the cell); the mistake-profile panel needs both, so the cell's target vector is
    stamped onto each record here (a shared reference, not a copy).
    """
    for cell in cells:
        for record in cell["records"]:
            record["_targets"] = cell["targets_test"]
    return [record for cell in cells for record in cell["records"]]


def agg(
    records: list[dict[str, Any]], key: str, f_b: float, s_silent: float, tau: float, metric: str
) -> tuple[float, float, list[float]]:
    """Seed mean, standard error and the per-seed values of one metric in one cell."""
    values = [
        float(r[metric])
        for r in records
        if r["key"] == key and r["f_b"] == f_b and r["s_silent"] == s_silent and r["tau"] == tau
    ]
    if not values:
        return float("nan"), float("nan"), []
    array = np.asarray(values)
    se = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return float(array.mean()), se, values


def verify(records: list[dict[str, Any]], cfg: SimpleNamespace) -> dict[str, bool]:
    """Check the goal conditions R1-R4 and print a verdict table.

    Each channel gate reads the CHANNEL'S OWN catastrophe counts (region B for epistemic,
    the silent pocket for aleatoric), not the pooled total: since the 8-D redesign the MLE's
    region-B failure never fully disappears (130 broad-subgroup samples do not cover it even
    at f_b = 1), so the pooled counts mix a flat aleatoric floor with a residual epistemic
    leak and their differences drown channel trends in cross-channel noise. Pooled numbers
    are still printed alongside.

    R1: CreRL + ours strictly better than the MLE on BOTH CVaR and POOLED catastrophe count at
        the headline cell (both uncertainty channels on), with non-overlapping +-1 se.
    R2: the epistemic channel, on region-B counts: the MLE's decrease monotonically in f_b
        (the reducible-by-data signature), CreRL's stay strictly below the MLE's at every f_b,
        and the starved-cell gap clearly exceeds the full-archive gap.
    R3: the aleatoric channel: (a) silent-pocket catastrophes strictly below the MLE's at
        f_b = 1, and (b) the CVaR premium bounded (within 10% of the MLE and strictly below
        the constant-ICU policy's v* = 8). A STRICT CVaR win is deliberately not required
        here: with abundant data the MLP's posterior is essentially calibrated, and against a
        KNOWN posterior a credal set can only add hedging beyond the CVaR optimum -- aleatoric
        risk alone does not need a set, which is exactly why the credal contribution
        concentrates in the epistemic channel (R2/R4).
    R4: the credal set's own contribution (not just the rule's), at the headline cell:
        (a) strictly better CVaR than the point ablation; (b) fewer region-B catastrophes than
        the point ablation; (c) per seed, the point ablation NEVER reaches the credal
        arm's operating point -- in every seed it either calibrates into the dominated regime
        (v >= D: the blanket retreat, mean cost ~3x) or commits more catastrophes than CreRL.
        CreRL vs the CreWra ensemble baseline (same rule, different set construction) is
        printed informationally, not gated.
        A seed-mean +-se test is deliberately not used for (c): the point arm is BIMODAL
        across seeds (v jumping between 4 and 8), and its huge variance is the pathology being
        demonstrated, not noise to average over.
    """
    tau = cfg.headline_tau
    ours, mle = "crerl|cvar_minimax", "mle|best_response"

    def cell(key: str, f_b: float, s_silent: float, metric: str) -> tuple[float, float]:
        mean_, se, _ = agg(records, key, f_b, s_silent, tau, metric)
        return mean_, se

    def strictly_below(key_lo: str, key_hi: str, f_b: float, s_silent: float, metric: str) -> tuple[bool, str]:
        lo, lo_se = cell(key_lo, f_b, s_silent, metric)
        hi, hi_se = cell(key_hi, f_b, s_silent, metric)
        ok = lo + lo_se < hi - hi_se
        return ok, f"{ARM_LABELS[key_hi]} {hi:.3f}+-{hi_se:.3f} vs {ARM_LABELS[key_lo]} {lo:.3f}+-{lo_se:.3f}"

    verdicts: dict[str, bool] = {}
    hf, hs = cfg.headline_f_b, cfg.headline_s_silent
    print(f"\n=== Goal verification (tau = {tau}) ===")

    ok = True
    for metric in ("cvar_cost", "catastrophe_count"):
        strict, text = strictly_below(ours, mle, hf, hs, metric)
        ok &= strict
        print(f"  R1 headline {metric}: {text} -> {'PASS' if strict else 'FAIL'}")
    verdicts["R1 headline"] = ok

    def component_cats(key: str, f_b: float, component: str) -> float:
        values = [
            r["catastrophes_by_component"][component]
            for r in records
            if r["key"] == key and r["f_b"] == f_b and r["s_silent"] == hs and r["tau"] == tau
        ]
        return float(np.mean(values))

    b_mle, b_ours = {}, {}
    for f_b in sorted(cfg.f_b_sweep):
        b_mle[f_b] = component_cats(mle, f_b, "sepsis_b")
        b_ours[f_b] = component_cats(ours, f_b, "sepsis_b")
        pooled_m, _ = cell(mle, f_b, hs, "catastrophe_count")
        pooled_o, _ = cell(ours, f_b, hs, "catastrophe_count")
        print(
            f"  R2 f_b={f_b}: region-B catastrophes MLE {b_mle[f_b]:.1f} vs CreRL+ours {b_ours[f_b]:.1f}"
            f"   (pooled {pooled_m:.1f} vs {pooled_o:.1f})"
        )
    order = sorted(b_mle)
    mle_decreases = all(b_mle[a] > b_mle[b] for a, b in zip(order, order[1:], strict=False))
    ours_below = all(b_ours[f] < b_mle[f] for f in order)
    starved_clear = (b_mle[order[0]] - b_ours[order[0]]) > 1.5 * (b_mle[order[-1]] - b_ours[order[-1]])
    verdicts["R2 epistemic-channel"] = mle_decreases and ours_below and starved_clear
    print(
        f"  R2 MLE region-B monotone-decreasing: {mle_decreases}; CreRL below at every f_b: {ours_below}; "
        f"starved gap clearly exceeds full-archive gap: {starved_clear} -> "
        f"{'PASS' if verdicts['R2 epistemic-channel'] else 'FAIL'}"
    )

    sil_mle = component_cats(mle, 1.0, "sepsis_silent")
    sil_ours = component_cats(ours, 1.0, "sepsis_silent")
    cat_ok = sil_ours < sil_mle
    pooled_text = strictly_below(ours, mle, 1.0, hs, "catastrophe_count")[1]
    print(
        f"  R3a aleatoric (silent-pocket) catastrophes at f_b=1: MLE {sil_mle:.1f} vs CreRL+ours "
        f"{sil_ours:.1f}   (pooled: {pooled_text}) -> {'PASS' if cat_ok else 'FAIL'}"
    )
    m_cvar, _ = cell(mle, 1.0, hs, "cvar_cost")
    o_cvar, _ = cell(ours, 1.0, hs, "cvar_cost")
    blanket = float(SPEC.cost.max(axis=0).min())  # the constant-action policy's CVaR at every tau
    premium_ok = o_cvar <= 1.10 * m_cvar and o_cvar < blanket
    print(
        f"  R3b aleatoric CVaR premium: CreRL+ours {o_cvar:.3f} vs MLE {m_cvar:.3f} "
        f"(<= +10% and < blanket {blanket}) -> {'PASS' if premium_ok else 'FAIL'}"
    )
    verdicts["R3 aleatoric"] = cat_ok and premium_ok

    point = "mle|cvar_minimax_point"
    cvar_ok, cvar_text = strictly_below(ours, point, hf, hs, "cvar_cost")
    print(f"  R4 set-vs-point cvar_cost: {cvar_text} -> {'PASS' if cvar_ok else 'FAIL'}")
    # Per-seed comparisons, pairing each seed's point-ablation record with the CreRL record of
    # the SAME seed: the point arm is bimodal (selective v < D in some seeds, blanket retreat
    # v >= D in others), so seed means average two different operating points and a mean +-se
    # test measures nothing. Region-B counts are compared only over the SELECTIVE seeds -- in a
    # blanket seed the arm reaches ~0 region-B catastrophes by ICU-ing every patient at ~3x
    # mean cost, which is a retreat, not region-B competence; those seeds are instead what the
    # operating-point check below counts.
    by_seed = {
        (r["key"], r["seed"]): r
        for r in records
        if r["f_b"] == hf and r["s_silent"] == hs and r["tau"] == tau and r["key"] in (ours, point)
    }
    seeds = sorted({seed for key, seed in by_seed if key == ours})
    selective = [seed for seed in seeds if not by_seed[(point, seed)]["v_in_dominated_regime"]]
    if selective:
        b_ours_sel = float(np.mean([by_seed[(ours, s)]["catastrophes_by_component"]["sepsis_b"] for s in selective]))
        b_point_sel = float(np.mean([by_seed[(point, s)]["catastrophes_by_component"]["sepsis_b"] for s in selective]))
        region_b_ok = b_ours_sel < b_point_sel
        print(
            f"  R4 region-B catastrophes over the point arm's {len(selective)} selective seed(s): "
            f"CreRL {b_ours_sel:.1f} vs point ablation {b_point_sel:.1f} -> "
            f"{'PASS' if region_b_ok else 'FAIL'}"
        )
    else:
        region_b_ok = True
        print(
            "  R4 region-B catastrophes: the point ablation was never selective (v >= D in every "
            "seed) -- nothing to compare; its blanket retreat is judged by the operating-point "
            "check -> PASS (vacuous)"
        )
    reaches = [
        seed
        for seed in selective
        if by_seed[(point, seed)]["catastrophe_count"] <= by_seed[(ours, seed)]["catastrophe_count"]
    ]
    never_reaches = not reaches
    print(
        f"  R4 point ablation reaches CreRL's operating point (v < D and catastrophes <= CreRL's) in "
        f"{len(reaches)}/{len(seeds)} seeds -> {'PASS' if never_reaches else 'FAIL'}"
    )
    verdicts["R4 credal-set-contribution"] = cvar_ok and region_b_ok and never_reaches

    # Informational (not gated): CreRL vs the ensemble credal baseline under the same rule.
    crewra = "crewra|cvar_minimax"
    for metric in ("cvar_cost", "catastrophe_count"):
        _, text = strictly_below(ours, crewra, hf, hs, metric)
        print(f"  INFO CreRL-vs-CreWra {metric}: {text}")
    print(
        f"  INFO CreRL-vs-CreWra region-B catastrophes: {component_cats(ours, hf, 'sepsis_b'):.1f} "
        f"vs {component_cats(crewra, hf, 'sepsis_b'):.1f}"
    )

    for name, is_ok in verdicts.items():
        print(f"  VERDICT {name}: {'PASS' if is_ok else 'FAIL'}")
    return verdicts


def _save(fig: plt.Figure, stem: str) -> None:
    """Write the figure as png and pdf under PLOTS_PATH."""
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(PLOTS_PATH / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_for_print(build: Callable[[Sizes], plt.Figure], stem: str, print_width: float) -> None:
    """Draw a paper figure for the width the paper includes it at, and write it under PLOTS_PATH.

    Args:
        build: Draws the figure with the given sizes and returns it.
        stem: File name without extension.
        print_width: Width in inches the paper includes the figure at.
    """
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    # One pass more than the default: these layouts hold more text beside their panels than a
    # plain grid of axes, so their width follows the type more closely and settles more slowly.
    fig, sizes = render_for_print(build, print_width, passes=3)
    save(fig, stem, PLOTS_PATH, print_width, sizes)
    plt.close(fig)


def _type_ratio(sizes: Sizes) -> float:
    """How large the type is drawn against the type the inch layouts were tuned for (_TUNED_SCALE)."""
    return sizes.scale / _TUNED_SCALE


def _text_width_in(text: str, face: FontProperties, size: float) -> float:
    """Width in inches of a text (its widest line) as matplotlib draws it in a face at a size."""
    fig = plt.figure()
    width = fig.text(0, 0, text, fontproperties=face, fontsize=size).get_window_extent().width / fig.dpi
    plt.close(fig)
    return width


def _fit_two_lines(text: str, face: FontProperties, size: float, room_in: float) -> str:
    """Break a one-line text that is wider than its room onto two lines of the most equal width.

    The same layout is drawn with type of very different sizes (see _TUNED_SCALE), so whether a
    label fits is decided where it is drawn rather than written into the label. A text that fits,
    or already carries a line break, is returned unchanged.
    """
    if "\n" in text or _text_width_in(text, face, size) <= room_in:
        return text
    words = text.split(" ")
    breaks = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    if not breaks:
        return text
    return "\n".join(min(breaks, key=lambda lines: max(_text_width_in(line, face, size) for line in lines)))


def fig_headline_bars(records: list[dict[str, Any]], cfg: SimpleNamespace) -> None:
    """Headline cell bars: CVaR, catastrophe count and mean cost, per arm with per-seed dots."""
    metrics = [
        ("cvar_cost", f"CVaR$_{{{cfg.headline_tau}}}$ cost"),
        ("catastrophe_count", "catastrophic decisions"),
        ("mean_cost", "mean cost"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for index, key in enumerate(ARM_ORDER):
            mean_, se, values = agg(records, key, cfg.headline_f_b, cfg.headline_s_silent, cfg.headline_tau, metric)
            ax.bar(index, mean_, yerr=se, color=ARM_COLORS[key], alpha=0.85, capsize=3)
            ax.scatter([index] * len(values), values, color="black", s=8, zorder=3, alpha=0.5)
        ax.set_xticks(range(len(ARM_ORDER)))
        ax.set_xticklabels([ARM_LABELS[k] for k in ARM_ORDER], rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Synthetic triage, headline cell (f_b={cfg.headline_f_b}, s_silent={cfg.headline_s_silent}), "
        f"tau={cfg.headline_tau}, {len(cfg.seeds)} seeds",
        fontsize=10,
    )
    _save(fig, "cost_sensitive_triage_bars")


def _sweep_panel(
    ax: plt.Axes,
    records: list[dict[str, Any]],
    cfg: SimpleNamespace,
    xs: list[float],
    cell_of: Callable[[float], tuple[float, float]],
    metric: str,
) -> None:
    """One sweep panel: metric vs knob, one line per arm, +-1 se band."""
    for key in ARM_ORDER:
        means, ses = [], []
        for knob in xs:
            f_b, s_silent = cell_of(knob)
            mean_, se, _ = agg(records, key, f_b, s_silent, cfg.headline_tau, metric)
            means.append(mean_)
            ses.append(se)
        means_a, ses_a = np.asarray(means), np.asarray(ses)
        ax.plot(xs, means_a, marker="o", color=ARM_COLORS[key], label=ARM_LABELS[key], markersize=4)
        ax.fill_between(xs, means_a - ses_a, means_a + ses_a, color=ARM_COLORS[key], alpha=0.15)
    ax.grid(alpha=0.3)


def fig_epistemic_sweep(records: list[dict[str, Any]], cfg: SimpleNamespace) -> None:
    """Metric vs f_b at the headline silent share. Read right-to-left: arms must coincide at 1.0."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    xs = sorted(cfg.f_b_sweep)
    titles = (f"CVaR$_{{{cfg.headline_tau}}}$ cost", "catastrophic decisions")
    for ax, metric, title in zip(axes, ("cvar_cost", "catastrophe_count"), titles, strict=True):
        _sweep_panel(ax, records, cfg, xs, lambda f_b: (f_b, cfg.headline_s_silent), metric)
        ax.set_xlabel("f_b (kept Sepsis-B fraction)")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Epistemic sweep (s_silent={cfg.headline_s_silent}); the gap is reducible by data", fontsize=10)
    _save(fig, "cost_sensitive_triage_epistemic")


def fig_aleatoric_sweep(records: list[dict[str, Any]], cfg: SimpleNamespace) -> None:
    """Metric vs s_silent at f_b = 1: the purely aleatoric channel, no epistemic gap at all."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    xs = sorted(cfg.silent_sweep)
    titles = (f"CVaR$_{{{cfg.headline_tau}}}$ cost", "catastrophic decisions")
    for ax, metric, title in zip(axes, ("cvar_cost", "catastrophe_count"), titles, strict=True):
        _sweep_panel(ax, records, cfg, xs, lambda s: (1.0, s), metric)
        ax.set_xlabel("s_silent (silent sepsis share)")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7)
    fig.suptitle("Aleatoric sweep (f_b = 1, no epistemic gap)", fontsize=10)
    _save(fig, "cost_sensitive_triage_aleatoric")


def _arm_line_style(key: str, sizes: Sizes) -> dict[str, Any]:
    """Line kwargs for one arm: the bar figures' group hue, plus a within-group dash and marker.

    A line figure cannot use the bar split as-is -- three baselines all drawn in the same blue
    would be three indistinguishable lines. So hue still carries the GROUP (blue baselines, tinted
    crimson for the rule on somebody else's uncertainty, full crimson for ours) and the dash
    pattern and marker separate arms within a group, by an arm's position in its PAPER_PANELS
    group (_LINE_STROKES). Ours additionally carries the heavier line and is drawn on top, so it
    stays findable at a glance.

    Widths, marker size and outline, cap styles, halo and drawing order are paper_style's
    (line_style), so these lines are built like every other line of the paper; only the colour,
    dash and marker are this figure's own.
    """
    index_in_group = next(group.index(key) for group in PAPER_PANELS if key in group)
    dashes, marker = _LINE_STROKES[index_in_group % len(_LINE_STROKES)]
    if key == FLAGSHIP_ARM:
        color = OURS_COLOR
    elif key in OURS_ARMS:
        color = matplotlib.colors.to_hex(_lighten(OURS_COLOR, _TEST_TINT))
    else:
        color = BASELINE_BLUE
    # Any method that is not ours has the baseline stroke; which one is asked for does not matter.
    style = line_style(OURS if key == FLAGSHIP_ARM else "base", sizes)
    return style | {"color": color, "linestyle": "solid" if dashes is None else dashes, "marker": marker}


def _mark_headline_level(ax: plt.Axes, position: float, sizes: Sizes) -> None:
    """Shade the sweep level the rest of the paper reports, and say so on the axes.

    Every other figure in this experiment is measured at one cell of this sweep, and without a mark
    the reader has no way to locate it -- the sweep looks like five equal settings when four of them
    exist only to place the fifth. Shading it turns the panel into context for the main result:
    left of the band the subgroup is absent entirely, right of it the archive holds progressively
    more of the cases the main figure does without.

    A band rather than a line, because the claim is about a whole column of points, and it sits at
    zorder 0 so it reads as background rather than as a sixth series. Neutral gray: a tinted band
    reads as a colour with a meaning, and every colour in these figures already has one.

    Drawn on every panel, at each one's own reported setting -- the headline archive cell in the
    sweep, tau = 0.1 in the two tau panels -- so the reader can find the main figure's operating
    point on whichever axis they are reading.
    """
    ax.axvspan(position - 0.42, position + 0.42, color="#e8e8e8", zorder=0, linewidth=0)
    ax.text(
        position,
        0.965,
        LINE_BAR_FIGURE_TAG,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontproperties=FP_SEMIBOLD,
        fontsize=sizes.tick,
        color="0.45",
        zorder=6,
    )


def fig_line_curves(records: list[dict[str, Any]], cfg: SimpleNamespace, tau: float | None = None) -> None:
    """The appendix line figure: the subgroup sweep and both tau curves, three panels, one legend.

    (a) CVaR_tau against the kept Sepsis-B fraction. Read RIGHT TO LEFT: at f_b = 1 the archive
        contains the subgroup, so the EPISTEMIC increment is gone and the curves flatten;
        everything that opens up as f_b falls is the reducible channel. The arms do NOT meet at
        f_b = 1 and are not supposed to -- the residual gap is the ALEATORIC channel (silent sepsis
        inside the healthy cluster), which no amount of subgroup data removes. The design's
        signature is the increment vanishing, not the arms coinciding; verify()'s R3 gates the
        aleatoric channel separately for exactly that reason.
    (b) CVaR_tau and (c) the catastrophe count against tau at the headline cell -- the quantity the
        rule optimises, and the quantity a clinician actually cares about.

    The three groups respond to tau through DIFFERENT channels, which is what (c) separates. MLE
    alone is genuinely flat there: its decisions never read tau, so its catastrophe count cannot
    move (its CVaR curve in (b) still falls, but that is the METRIC moving under a fixed policy,
    not the policy responding). SQwash and AdaCVaR are trained ALIGNED -- run_cell fits a fresh
    model per evaluated tau at training tail fraction tau -- so they respond through TRAINING. The
    CVaR-minimax arms keep one model and recalibrate the threshold, so they respond through the
    RULE, which is why they are the ones sweeping from ~0 up to the baselines' level.

    Both knobs are placed CATEGORICALLY: neither f_b (0, 0.05, 0.25, 0.5, 1) nor tau (0.01 ... 0.2)
    is evenly spaced, and a linear axis would crush the closely spaced levels into a fifth of the
    panel and hand the rest of the width to the gap above them.

    Colours are the bar figures' (_arm_line_style): the group's hue, with a dash and a marker
    telling the arms of a group apart. Everything else is the paper's shared style
    (plotting/paper_style.py): type, stroke widths, haloed lines, the reference panel chrome -- a
    hairline box, no grid -- and the legend below with ours set apart.
    """
    tau = cfg.headline_tau if tau is None else tau
    f_bs = sorted(cfg.f_b_sweep)
    # The sweep axis is labelled in ACTUAL SUBGROUP CASES, not in the keep fraction f_b. f_b is an
    # implementation knob -- "5% of the subgroup's rows survived the filter" -- and it reads as a
    # small perturbation when the point is the opposite: the deployment cohort is the same
    # throughout and only the TRAINING DATA changes, from six recorded cases to a hundred and
    # twenty. The count says that; a percentage of an unstated denominator does not. It is exact in
    # expectation, since make_data draws n_train rows and keeps each subgroup one with probability
    # f_b, giving n_train * p_sepsis * sepsis_b_share * f_b of them.
    #
    # CAVEAT, deliberate and to be carried by the caption: the counts are region-B cases ONLY. The
    # sweep moves just that subgroup, and the training data holds ~800 sepsis cases of the other two
    # components at every level of it -- so the leftmost tick reads 0 while the data still contains
    # hundreds of sepsis patients, none of them of this presentation. The axis is worded for a
    # reader who has met the subgroup in the text; it is not self-contained without them.
    cases_per_unit_f_b = cfg.n_train * cfg.p_sepsis * cfg.sepsis_b_share
    # (x levels, tick text, x label, metric, y label, the cell each x point names, marked level)
    specs = (
        (
            f_bs,
            [f"{round(cases_per_unit_f_b * f_b):g}" for f_b in f_bs],
            "recorded sepsis cases in the data",
            "cvar_cost",
            f"CVaR$_{{{tau}}}$ cost",
            lambda f_b: (f_b, tau),
            cfg.headline_f_b,
        ),
        (
            cfg.taus,
            [f"{t:g}" for t in cfg.taus],
            "τ",
            "cvar_cost",
            "CVaR$_τ$ cost",
            lambda t: (cfg.headline_f_b, t),
            cfg.headline_tau,
        ),
        (
            cfg.taus,
            [f"{t:g}" for t in cfg.taus],
            "τ",
            "catastrophe_count",
            "catastrophic decisions",
            lambda t: (cfg.headline_f_b, t),
            cfg.headline_tau,
        ),
    )
    arms = [key for group in PAPER_PANELS for key in group]
    # Per panel and arm: the seed mean and its standard error at every level. Aggregated once, since
    # the figure is drawn more than once (render_for_print).
    series: list[dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]] = []
    for levels, _ticks, _xlabel, metric, _ylabel, cell_of, _marked in specs:
        panel_series = {}
        for key in arms:
            cells = [cell_of(level) for level in levels]
            points = [agg(records, key, f_b, cfg.headline_s_silent, level_tau, metric) for f_b, level_tau in cells]
            panel_series[key] = (np.asarray([p[0] for p in points]), np.asarray([p[1] for p in points]))
        series.append(panel_series)

    def build(sizes: Sizes) -> plt.Figure:
        """Draw the figure with its type at the given sizes; everything above is data."""
        use_fira_mathtext()
        fig_w, fig_h = 7.0, 1.72
        fig = plt.figure(figsize=(fig_w, fig_h))
        panel_w, panel_h, y0 = 1.70 / fig_w, 1.13 / fig_h, 0.36 / fig_h
        # Panel pitch is 2.36in: 1.70 of axes plus 0.66 of gutter, which is what the next panel's y
        # label and its widest y number need. Uniform, so the three read as one row rather than as
        # a figure whose spacing drifts.
        axes = [fig.add_axes((x_in / fig_w, y0, panel_w, panel_h)) for x_in in (0.50, 2.86, 5.22)]
        # matplotlib's own distances between a tick, its label and the axis label are points tuned
        # for the reference's type; they shrink with it here, or the labels drift off their axes.
        tick_pad = plt.rcParams["xtick.major.pad"] * sizes.scale
        label_pad = plt.rcParams["axes.labelpad"] * sizes.scale

        entries = []
        for ax, (levels, ticks, xlabel, _metric, ylabel, _cell_of, marked), panel_series in zip(
            axes, specs, series, strict=True
        ):
            positions = np.arange(len(levels), dtype=float)
            for key in arms:
                style = _arm_line_style(key, sizes)
                means_a, ses_a = panel_series[key]
                (line,) = ax.plot(positions, means_a, **style)
                # +-1 se over seeds on EVERY panel. It costs almost nothing on the two CVaR panels
                # (se <= 0.5 against a 15-wide range, so the bands are hairlines), and on the
                # catastrophe panel it is arguably the most important thing in the figure: at
                # tau = 0.1 the point and ensemble ablations sit at 47.7 +- 25.0 and 32.4 +- 18.7
                # against ours at 36.7 +- 2.4. Their MEANS straddle ours, so a band-less panel
                # invites reading a difference that is not there -- while the bands show the real
                # separation, which is in the spread: those arms swing between operating points
                # from seed to seed (the bimodality R4 gates per seed for exactly this reason) and
                # ours does not. Hiding that would flatter the mean comparison and lose the
                # stability result at the same time.
                ax.fill_between(positions, means_a - ses_a, means_a + ses_a, **band_style(style["color"]))
                if ax is axes[0]:  # one legend for the figure, built from the first panel
                    entries.append((line, _arm_legend_label(key, cfg)))
            ax.set_xticks(positions)
            ax.set_xticklabels(ticks)
            ax.set_xlim(-0.25, len(levels) - 0.75)
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            # Top headroom for the figure tag. Autoscale leaves a series running within a few
            # percent of the top -- MLE's catastrophe count does exactly that in the third panel --
            # and the tag would then sit on the line it is meant to annotate. Applied to every panel
            # so the three tags sit at the same height and read as one mark rather than three.
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, lo + (hi - lo) * 1.14)
            set_xlabel(ax, xlabel, sizes)
            set_ylabel(ax, ylabel, sizes)
            ax.xaxis.labelpad = ax.yaxis.labelpad = label_pad
            ax.tick_params(pad=tick_pad)
            style_axes(ax, sizes)
            _mark_headline_level(ax, float(list(levels).index(marked)), sizes)
        # The shaded bands are +-1 SEM over seeds. That is stated in the CAPTION, not in the legend:
        # the legend row is for the arms. Worth knowing when reading the figure -- ours' band is 2-4
        # px tall, because its seeds land within +-0.07 CVaR, so on that arm the shading is
        # invisible and can be misread as absent rather than as too tight to see.
        #
        # One row in PAPER_PANELS order -- the argmax arms, then the two that put our rule on
        # somebody else's predictor -- with ours set apart on its right. It is set apart by an empty
        # cell of the SAME row rather than through legend_below's `ours`, which draws a second
        # legend centred on the first by bounding box: against a single row that leaves ours off
        # the row's baseline, by an amount that changes with the backend, because its subscript
        # makes its box the taller one. Cells of one row share their baseline exactly.
        # The legend's top edge sits a little below the canvas, clear of the x labels; the saved
        # figure is cropped to its content.
        spacer = (plt.Line2D([], [], linestyle="none"), "")
        legend_below(fig, [[*entries[:-1], spacer, entries[-1]]], ncol=len(entries) + 1, sizes=sizes, top=-0.03)
        # No suptitle: the house style carries the cell in the caption, not on the canvas.
        return fig

    _save_for_print(build, "cost_sensitive_triage_line_curves", LINE_PRINT_WIDTH)


FLU_MISTREATED_COLOR = "#d3c4ac"
MISTAKE_CELLS: tuple[tuple[str, int, tuple[int, ...], str, str], ...] = (
    ("sepsis $\\rightarrow$ sent home (20)", 2, (0,), "#67000d", "sepsis sent home"),
    ("sepsis $\\rightarrow$ flu treatment (14)", 2, (1,), "#d7301f", "sepsis as flu"),
    ("healthy $\\rightarrow$ ICU (8)", 0, (2,), "#fc8d59", "healthy in ICU"),
    ("healthy $\\rightarrow$ flu treatment (6)", 0, (1,), "#fdd49e", "healthy as flu"),
    ("flu mistreated (4)", 1, (0, 2), FLU_MISTREATED_COLOR, "flu mistreated"),
)


def mistake_profile(
    records: list[dict[str, Any]], key: str, f_b: float, s_silent: float, tau: float, cost_weighted: bool
) -> list[float]:
    """Seed-mean count (or count x cost) of each MISTAKE_CELLS entry for one arm in one cell.

    Cost-weighting is what keeps the severe cells visible: as raw counts the stack is swamped
    by cheap mistakes (near v = 4 the rule arms route the entire flu population to the ICU at
    cost 4 -- thousands of instances) and the ~25 catastrophes disappear at that scale, which
    inverts the panel's message. Weighted by cost, each bar decomposes the arm's total
    incurred mistake cost, so a cost-20 block is exactly 5x the height of a cost-4 block.
    """
    rs = [r for r in records if r["key"] == key and r["f_b"] == f_b and r["s_silent"] == s_silent and r["tau"] == tau]
    profile = []
    for _label, true_class, actions, _color, _short in MISTAKE_CELLS:
        counts = [int(((r["_targets"] == true_class) & np.isin(r["actions"], actions)).sum()) for r in rs]
        weight = float(SPEC.cost[true_class, actions[0]]) if cost_weighted else 1.0
        # Per patient, so the stack is a decomposition of panel (b)'s mean cost (up to the
        # small cost of CORRECT decisions, 1 for treated flu and 2 for sepsis in the ICU).
        n_test = len(rs[0]["_targets"]) if rs else 1
        profile.append(float(np.mean(counts)) * weight / n_test)
    return profile


def tail_cost_profile(records: list[dict[str, Any]], key: str, f_b: float, s_silent: float, tau: float) -> list[float]:
    """Seed-mean contribution of each MISTAKE_CELLS entry to CVaR_tau, for one arm in one cell.

    The exact decomposition of the tail-cost bar: metrics.cvar averages the worst
    k = int(tau * N) realized costs, so summing each cell's share of that same top-k sum,
    divided by the same k, reproduces CVaR_tau to the last digit. The stack is therefore an
    anatomy of the headline number itself, not a separate quantity that merely correlates with
    it -- which is what makes it worth putting directly under the CVaR bars.

    Two facts make this well defined without any tie-breaking convention. First, every non-zero
    entry of the cost table is attained by exactly ONE MISTAKE_CELLS entry (4 is flu -> No
    Action and flu -> ICU, which the table already merges into one segment), so instances tied
    at the tail boundary belong to the same segment however the sort orders them, and the
    per-cell sums are invariant. Second, the CORRECT-but-costly decisions (treated flu at 1,
    sepsis in the ICU at 2) never enter the top-k at the tau levels this figure uses -- measured
    at every arm of the headline cell, their contribution is exactly 0.0 -- so the five mistake
    cells tile the tail completely and the stack loses nothing. Both properties are asserted at
    render time in fig_paper rather than trusted.
    """
    rs = [r for r in records if r["key"] == key and r["f_b"] == f_b and r["s_silent"] == s_silent and r["tau"] == tau]
    if not rs:
        return [float("nan")] * len(MISTAKE_CELLS)
    per_seed = []
    for r in rs:
        targets, actions = r["_targets"], r["actions"]
        costs = SPEC.cost[targets, actions]
        k = int(tau * len(costs))
        tail = np.argsort(costs, kind="stable")[-k:]
        tail_costs, tail_targets, tail_actions = costs[tail], targets[tail], actions[tail]
        per_seed.append(
            [
                float(tail_costs[(tail_targets == true_class) & np.isin(tail_actions, acts)].sum()) / k
                for _label, true_class, acts, _color, _short in MISTAKE_CELLS
            ]
        )
    return list(np.asarray(per_seed).mean(axis=0))


def _rounded_stack(  # noqa: PLR0913, PLR0917
    ax,  # noqa: ANN001
    x0: float,
    x1: float,
    bounds: list[float],
    colors: list[str],
    radius_in: float,
    hairline: float,
) -> None:
    """A STACKED bar whose silhouette is rounded exactly like plot_runtime_bars._rounded_bar.

    Rounding each segment separately would put a corner arc on every internal boundary and the
    stack would read as a pile of lozenges. So the outline is built once, from the floor to the
    total, and the segments are plain rectangles CLIPPED to it: only the topmost segment picks up
    the rounding, and the internal boundaries stay flat, which is what makes the stack still read
    as one bar of the same family as the solid ones in the panel above.

    Geometry (axes coordinates, radius in inches, the underhang that keeps the bottom corners
    square) is the sibling function's, so a stacked bar and a solid bar of the same height are
    the same shape. `bounds` is the cumulative ladder, length len(colors) + 1, and `hairline` the
    width in points of the page-white rule on each internal boundary.
    """
    to_axes = ax.transData + ax.transAxes.inverted()
    (ax0, _), (ax1, _) = to_axes.transform([(x0, bounds[0]), (x1, bounds[0])])
    ys = [float(to_axes.transform((x0, b))[1]) for b in bounds]
    aspect = _axes_aspect(ax)
    radius = radius_in / (ax.get_position().width * ax.figure.get_figwidth())
    drop = radius * aspect + 0.02
    outline = matplotlib.patches.FancyBboxPatch(
        (ax0, ys[0] - drop),
        ax1 - ax0,
        ys[-1] - ys[0] + drop,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=ax.transAxes,
        mutation_aspect=aspect,
        linewidth=0,
        facecolor="none",
        edgecolor="none",
        zorder=2,
    )
    ax.add_patch(outline)
    for lo, hi, color in zip(ys[:-1], ys[1:], colors, strict=True):
        segment = matplotlib.patches.Rectangle(
            (ax0, lo), ax1 - ax0, hi - lo, transform=ax.transAxes, linewidth=0, facecolor=color, zorder=2
        )
        segment.set_clip_path(outline)
        ax.add_patch(segment)
    # Hairline of page-white on each internal boundary, the same trick the runtime figure's split
    # legend pill uses to stop two fills bleeding into one another at this size.
    for y in ys[1:-1]:
        line = plt.Line2D(
            [ax0, ax1], [y, y], transform=ax.transAxes, color="white", lw=hairline, solid_capstyle="butt", zorder=3
        )
        line.set_clip_path(outline)
        ax.add_line(line)


def _flagship_label(cfg: SimpleNamespace, rule: bool = False) -> str:
    """Our arm's name in the paper's house style: CreWare, alpha as a subscript, the ours tag.

    The label is meant to be set in the SEMIBOLD face as a whole -- name, subscript and tag -- which
    is how the paper marks its own method wherever it appears (paper_style.ours_label); the caller
    picks the face. With `rule` the decision rule is named as well ("CreWare + rule [ours]"), as the
    paper's bar and line figures do.

    The alpha is read from the CONFIG, never hardcoded. The runtime figure writes 0.95 because that
    is what its covid runs used; this experiment runs at cfg.alpha. Pinning the digits in the string
    would silently mislabel a hyperparameter the moment either number moved, and a wrong alpha in a
    paper figure is not a cosmetic error.

    The mathtext kerning is the runtime figure's, for the same reason: mathtext gives a bare "."
    TeX's punctuation spacing, which opens a visible gap on both sides. Only the RIGHT one is
    negated -- the period's ink sits low and left, so it needs the natural space before it to look
    centred.
    """
    subscript = f"{cfg.alpha:g}".replace(".", r".\!")
    return ours_label(rf"CreWare$_{{\,{subscript}}}$" + (" + rule" if rule else ""))


# A badge's padding in ems of its type, and the pitch of its lines.
_STICKER_PAD_EM = 0.45
_STICKER_LINESPACING = 1.35


def _sticker(ax: plt.Axes, text: str, sizes: Sizes, x: float = 0.5) -> None:
    """A small rounded badge in a panel's headroom, saying what the bars under it amount to.

    Neutral light gray, not one of the method or severity hues: the badge summarises a panel, so
    borrowing a colour already carrying meaning elsewhere in the figure would read as a reference
    to that specific thing (the crimson would claim it is about ours, the dark red that it is about
    one segment). Gray keeps it as annotation rather than data.

    Placed in the HEADROOM above the bars, never over them: these panels are read for the size of
    the dark block, and a label covering the thing it describes would be self-defeating. fig_paper
    widens the shared y limit to make that headroom -- widening only the bottom row would break the
    equal-height correspondence with the row above.

    Anchored va="top", so a one-line and a two-line badge still start at the same height and the
    three read as one band across the figure.

    `x` is in AXES coordinates and defaults to the panel centre. A badge that describes only some of
    its panel's bars is centred over those bars instead: horizontal position is how the reader
    infers scope, so it has to match what the sentence actually claims.
    """
    ax.text(
        x,
        0.97,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        multialignment="center",
        color="black",
        fontproperties=FP_REGULAR,
        fontsize=sizes.tick,
        linespacing=_STICKER_LINESPACING,
        zorder=6,
        bbox={"boxstyle": f"round,pad={_STICKER_PAD_EM}", "facecolor": "0.93", "edgecolor": "none"},
    )


def _swatch_legend(fig: plt.Figure, swatches: list[tuple[str, str]], sizes: Sizes, top: float = 0.0) -> None:
    """A frameless key of rounded colour pills below the figure, in as few rows as fit its width.

    The entries keep their reading order, left to right and then down (paper_style.legend_below).
    How many fit on a row depends on how large the type is drawn against the figure, which differs
    between the renders sharing a layout: the same five entries are one row in the two-row bar
    figure and two in the appendix pair, whose type is nearly twice the size against its bars.

    Args:
        fig: Figure to attach the legend to.
        swatches: (colour, label) per entry, in reading order.
        sizes: The figure's sizes.
        top: Figure-fraction y of the legend's top edge.
    """
    # The pill is a round-capped stroke, six tenths of the type's size thick.
    handles = [
        plt.Line2D([0], [0], color=color, lw=0.6 * sizes.legend, solid_capstyle="round") for color, _label in swatches
    ]
    entries = list(zip(handles, [label for _color, label in swatches], strict=True))
    for n_rows in range(1, len(entries) + 1):
        ncol = -(-len(entries) // n_rows)
        rows = [entries[i : i + ncol] for i in range(0, len(entries), ncol)]
        legend_below(fig, rows, ncol=ncol, sizes=sizes, handlelength=1.1, top=top)
        legend = fig.legends[-1]
        fig.canvas.draw()
        if ncol == 1 or legend.get_window_extent().width <= fig.bbox.width:
            return
        legend.remove()


def _legend_ladder(fig: plt.Figure, sizes: Sizes) -> None:
    """The cost-decomposition key, drawn as a SEVERITY LADDER rather than a legend box.

    The boxed legend this replaces had to sit inside the axes, which cost a third of the panel in
    headroom purely to clear the bars -- and under tail_decomposition that headroom is exactly what
    breaks the equal-height correspondence between the two rows. Moving it out of the axes buys
    that back.

    The rethink is in the reading order, not just the position. These five entries are not
    unordered categories: they are the cost table's non-zero cells, and the stack is drawn in that
    order, so the key is a scale, severe to mild, left to right (and on to a second row where one
    does not hold them, see _swatch_legend). Rounded pill swatches, in the paper's legend face.

    The cost sits in parentheses AFTER the name rather than leading it. Leading with the number
    read badly for a typographic reason: the entry then opens with mathtext, and mathtext's spacing
    around a `$...$` run does not collapse against the following space, so every label carried a
    wide gap between the digits and the words -- and with 20, 14, 8, 6, 4 being different widths,
    those gaps were ragged too. Trailing "(20)" puts the ordinary text first, where ordinary word
    spacing applies, and the parenthesis binds the number to the name it prices.
    """
    swatches = [
        (color, rf"{short} ($\mathbf{{{SPEC.cost[true_class, acts[0]]:g}}}$)")
        for _label, true_class, acts, color, short in MISTAKE_CELLS
    ]
    _swatch_legend(fig, swatches, sizes)


# Pitch of a row name's lines, and the gap between the last of them and the y numbers, in ems of the
# axis-label type.
_ROW_NAME_PITCH = 1.2
_ROW_NAME_GAP = 0.65


def _row_name(fig: plt.Figure, x_in: float, y_in: float, text: str, sizes: Sizes) -> None:
    """Name a row beside its axes: rotated text, the baseline of its first line at x_in.

    Every line is a text of its own, anchored ON ITS BASELINE. That is what puts the names of
    stacked rows on one vertical line in every backend, and it is why these are not y labels:
    matplotlib sets a y label by its bounding box, and so does any alignment of y labels after
    the fact. A box is measured in whole pixels of the canvas it was laid out on (0.7 pt each at
    the default 100 dpi), and mathtext boxes its ink where plain text boxes its font, so "CVaR_0.1
    cost" and "mean cost" pinned to one edge ended up most of a point apart in the pdf.

    Args:
        fig: Figure to draw on.
        x_in: Inches from the figure's left edge to the baseline of the first line; further
            lines follow towards the axes.
        y_in: Inches from the figure's bottom edge to the middle of the row.
        text: The name, one line or more.
        sizes: The figure's sizes.
    """
    width, height = fig.get_size_inches()
    for index, line in enumerate(text.split("\n")):
        fig.text(
            (x_in + index * _ROW_NAME_PITCH * sizes.label / 72) / width,
            y_in / height,
            line,
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="baseline",
            fontproperties=FP_SEMIBOLD,
            fontsize=sizes.label,
            color="black",
        )


def fig_paper(  # noqa: PLR0913, PLR0914, PLR0915
    records: list[dict[str, Any]],
    cfg: SimpleNamespace,
    tau: float | None = None,
    suffix: str = "",
    rows: tuple[str, ...] = ("cvar", "tail"),
    f_b: float | None = None,
    stickers: tuple[str, ...] = PAPER_STICKERS,
    limit_f_bs: tuple[float, ...] | None = None,
    print_width: float = PAPER_PRINT_WIDTH,
) -> None:
    """The one-column paper figure: two bar panels over the same arms, headline cell.

    Both rows carry the same arms in the same slots, split into the same three panels, so a
    segment always sits directly under the bar it explains. `rows` names what each row shows:

    - "cvar": CVaR_tau as a SOLID bar, with a +-1 se whisker and the per-seed values as dots.
      The uncertainty row.
    - "tail": the same CVaR_tau, STACKED by which costly (true class, action) decision produced
      it (tail_cost_profile), so the bar's total is unchanged and its anatomy is visible -- the
      risk-neutral arms carry the dark "sepsis sent home" block, the risk-averse arms trade it
      for the orange "healthy in ICU" premium block. Keeps the se whisker, since the total is
      still exactly the CVaR; drops the seed dots, which a five-colour stack cannot carry.
    - "mean_bar": mean cost as a SOLID bar, the same treatment "cvar" gives the tail -- se whisker
      and per-seed dots. The PRICE of the tail win, with its spread across seeds.
    - "mean": the per-patient mistake cost over the whole test population, stacked the same way.
      Its total is the arm's mean cost up to the small cost of correct decisions, so it is that
      same price broken into the decisions that produce it.

    Any number of rows is allowed and they are laid out top-down, so rows[0] is always the top one
    and the bottom row always carries the method names and the stickers. The paper render is
    ("cvar", "tail") -- the top row carries the uncertainty and the bottom the anatomy of that same
    number, the redundancy in total height being the point. The APPENDIX render is
    ("cvar", "mean_bar", "mean"): the tail, its average-cost price, and the decisions that price is
    made of, which is the original four-panel figure's three bar plots in one column.

    Rows measuring the same quantity share a y scale (see row_lims): with independent limits
    identical numbers render at different heights and the visual claim silently breaks.

    `tau` defaults to the headline; pass another level (and a filename suffix) for comparison
    renders -- note that below tau ~ 0.05 every rule-based arm collapses to the blanket policy
    and ties at CVaR 8.

    `print_width` is the width in inches the paper includes the render at. The bars are laid out in
    inches and do not depend on it; the TYPE does, since it has to print at the size of every other
    figure's (paper_style.render_for_print). Included across the appendix page the type is small
    against the bars; two side by side at half that width, it is nearly twice the size against
    them, and the layout gives it the room (see build).
    """
    if any(kind not in ROW_KINDS for kind in rows):
        raise ValueError(f"rows must be drawn from {sorted(ROW_KINDS)}, got {rows}")
    tau = cfg.headline_tau if tau is None else tau
    hs = cfg.headline_s_silent
    hf = cfg.headline_f_b if f_b is None else f_b
    # The stickers state MEASURED facts about one cell, so moving the cell without moving them
    # would leave the badges asserting the headline cell's numbers over another cell's bars.
    if hf != cfg.headline_f_b and stickers is PAPER_STICKERS:
        raise ValueError(
            f"f_b={hf} is not the headline cell {cfg.headline_f_b}, but the headline stickers were "
            f"left in place. Pass a `stickers` set measured at this cell (see PAPER_STICKERS_FB25)."
        )
    panels = [list(group) for group in PAPER_PANELS]
    plotted = [key for group in panels for key in group]

    def arm_color(key: str) -> str:
        if key == FLAGSHIP_ARM:
            return OURS_COLOR
        # to_hex, not the raw rgb triple: the rounded-bar helper is typed for a colour STRING, and
        # keeping one type through means the ours tint and the flat hues stay interchangeable.
        return matplotlib.colors.to_hex(_lighten(OURS_COLOR, _TEST_TINT)) if key in OURS_ARMS else BASELINE_BLUE

    # `level` is passed rather than closed over: `tau` is declared float | None on the signature and
    # narrowed above, but that narrowing does not follow into a nested function.
    def cell_series(cell_f_b: float, level: float) -> tuple[dict[str, Any], dict[str, dict[str, list[float]]]]:
        """Per-arm aggregates and stack heights for one archive cell."""
        cell_stats = {
            metric: {k: agg(records, k, cell_f_b, hs, level, metric) for k in plotted}
            for metric in {ROW_METRIC[kind] for kind in rows}
        }
        cell_stacks: dict[str, dict[str, list[float]]] = {}
        for kind in set(rows) & {"tail", "mean"}:
            cell_stacks[kind] = {}
            for key in plotted:
                if kind == "tail":
                    heights = tail_cost_profile(records, key, cell_f_b, hs, level)
                    # The stack must reproduce the CVaR it claims to decompose; a silent mismatch
                    # would make the row a different quantity wearing the CVaR axis. Float noise only.
                    if not np.isclose(sum(heights), cell_stats["cvar_cost"][key][0], rtol=1e-9, atol=1e-9):
                        raise AssertionError(
                            f"{key} at f_b={cell_f_b}: tail decomposition sums to {sum(heights):.6f}, but "
                            f"CVaR is {cell_stats['cvar_cost'][key][0]:.6f}. A correct-but-costly cell has "
                            f"entered the tau={level} tail, so MISTAKE_CELLS no longer tiles it -- extend the "
                            f'table or drop the "tail" row.'
                        )
                else:
                    heights = mistake_profile(records, key, cell_f_b, hs, level, cost_weighted=True)
                cell_stacks[kind][key] = heights
        return cell_stats, cell_stacks

    stats, stacks = cell_series(hf, tau)

    def row_extent(kind: str, from_stats: dict[str, Any], from_stacks: dict[str, dict[str, list[float]]]) -> float:
        """The tallest thing a row has to fit, whiskers and seed dots included."""
        if kind in SOLID_KINDS:
            by_arm = from_stats[ROW_METRIC[kind]]
            return max(max(by_arm[k][0] + by_arm[k][1], *by_arm[k][2]) for k in plotted)
        return max(sum(from_stacks[kind][k]) for k in plotted)

    # Extents are taken over EVERY cell in limit_f_bs, not just the one being drawn, so a set of
    # renders can share one y scale and be compared bar-for-bar across cells. Without it each figure
    # autoscales to its own data and the same bar height means a different number in each -- at 25%
    # the CVaR-minimax arms' mean cost is a third of its 5% value, so read against their own axes the
    # two figures look alike. Every render in the set passes the same tuple, so the limits do not
    # depend on which one is drawn first.
    cells_for_limits = (hf,) if limit_f_bs is None else limit_f_bs
    series_for_limits = [cell_series(cell, tau) if cell != hf else (stats, stacks) for cell in cells_for_limits]
    extents = [max(row_extent(kind, st, sk) for st, sk in series_for_limits) for kind in rows]

    # "by mistake" earns its two extra words: without them a label says "mean cost" while the axis
    # shows five stacked quantities, and the reader has to reach the legend to learn that the height
    # is a decomposition rather than five separate bars. "grouped by mistake type" says the same
    # thing at nearly twice the length, and the label has to fit beside a 1.02in row.
    labels = {
        "cvar": f"CVaR$_{{{tau}}}$ cost",
        "tail": f"CVaR$_{{{tau}}}$ by mistake",
        "mean_bar": "mean cost",
        "mean": "mean cost by mistake",
    }
    # An arm is named on one line under its bar, predictor and rule together, as in Figure 1.
    tick_labels = {key: _arm_legend_label(key, cfg) for group in PAPER_PANELS for key in group}
    segment_colors = [cell[3] for cell in MISTAKE_CELLS]

    def build(sizes: Sizes) -> plt.Figure:  # noqa: PLR0914, PLR0915
        """Draw the figure with its type at the given sizes; everything above is data."""
        use_fira_mathtext()
        # The type against the type this layout was tuned for. The bars keep their inches whatever
        # it is; the margins and offsets that exist to hold text are tuned inches times this.
        type_ratio = _type_ratio(sizes)

        # Geometry in INCHES off a fixed inches-per-data-unit, exactly as plot_runtime_bars._plot_broken
        # does it: the figure width FOLLOWS from the slot pitch, which is what keeps a bar the same
        # physical width in the one-slot ours panel as in the wide baseline one. unit_in, half and the
        # margins are that figure's values verbatim. The pitch is NOT: it carries a 0.7in slot here
        # because these method names are horizontal and much longer than "CreEns" -- "AdaCVaR" is the
        # binding label, and rotating them instead would break the runtime figure's flat baseline.
        unit_in, half, pitch = 0.7007, 0.16, 1.0
        left_in, right_in, gap_in = 0.44 * type_ratio, 0.10, 0.18
        pad = 0.415  # half the 0.83 shoulder the runtime figure leaves a panel, so bars never touch a spine

        # A panel's slots are `pitch` apart. The rule arms' one-line labels outgrow that slot where
        # the type is drawn large against the bars (the appendix pair), and side by side they would
        # touch. Such a panel is pitched by its labels instead: half of each neighbour plus a gap.
        def label_in_of(key: str) -> float:
            return _text_width_in(tick_labels[key], FP_SEMIBOLD if key == FLAGSHIP_ARM else FP_LIGHT, sizes.tick)

        def panel_pitch(group: list[str]) -> float:
            widths = [label_in_of(key) for key in group]
            neighbours = [(a + b) / 2 for a, b in zip(widths, widths[1:], strict=False)]
            return max([pitch, *((need + 0.8 * sizes.tick / 72) / unit_in for need in neighbours)])

        pitches = {tuple(group): panel_pitch(group) for group in panels}
        spans = [(len(group) - 1) * pitches[tuple(group)] + 2 * pad for group in panels]
        # The ours panel holds ONE slot but the widest label ("CreWare_0.95 [ours]"), which would
        # otherwise overhang into the gap and collide with "CreWra" next door. Widening the panel
        # rather than shrinking the label keeps every bar the same width, which the split layout is
        # built on -- the runtime figure buys the same room the same way (its ours_span is 0.95).
        # 0.42 is that room for the type the layout was tuned at; where the label is drawn larger
        # the panel follows it, with a tenth of an inch on either side. That margin also seats the
        # two-line badge above the bar, which is a little wider than the label -- and since the
        # width comes from the label alone, renders that differ only in their badges (the appendix
        # pair) keep one geometry and can be set side by side.
        label_in = label_in_of(FLAGSHIP_ARM)
        # The badge over the ours bar has to fit its panel as well. Measured over every badge set, not
        # just this render's, so the appendix pair keeps one geometry.
        badge_in = max(
            _text_width_in(line, FP_REGULAR, sizes.tick)
            for badges in (PAPER_STICKERS, PAPER_STICKERS_FB25)
            for line in badges[-1].split("\n")
        )
        badge_in += 2 * _STICKER_PAD_EM * sizes.tick / 72
        spans[-1] = max(spans[-1] + 0.42, (label_in + 2 * 0.10) / unit_in, (badge_in + 2 * 0.04) / unit_in)
        fig_w = left_in + sum(spans) * unit_in + (len(panels) - 1) * gap_in + right_in
        # Equal row heights, unlike the runtime figure's 1.5x: there the rows are different quantities
        # on different scales, here they are the SAME number twice, so any height difference would
        # misrepresent the identity the layout is built on.
        #
        # A row is 1.02in at the tuned type: the bars' own height plus the headroom a two-line badge
        # needs above the tallest of them, which is what the bottom row's y limit is widened by (a
        # factor of 1.42 there). The badge grows with the type, the bars do not, so larger type buys
        # its extra room by making the rows taller -- every row, to keep them equal -- instead of
        # squashing the bars under it. Smaller type leaves the tuned geometry alone.
        bars_in = 1.02 / 1.42
        h_row = bars_in + (1.02 - bars_in) * max(type_ratio, 1.0)
        sticker_headroom = h_row / bars_in
        bottom_in, gap_v, top_in = 0.30 * type_ratio, 0.13, 0.04 * type_ratio
        n_rows = len(rows)
        fig_h = bottom_in + n_rows * h_row + (n_rows - 1) * gap_v + top_in
        fig = plt.figure(figsize=(fig_w, fig_h))

        cols, x_in = [], left_in
        for group, span in zip(panels, spans, strict=True):
            cols.append((x_in / fig_w, span * unit_in / fig_w, group))
            x_in += span * unit_in + gap_in
        # Rows are laid out TOP-DOWN so rows[0] is the top one, whatever the count: the appendix render
        # stacks three and the paper two, and an index that means "top" in one and "second" in the other
        # would silently move every sticker and tick label.
        rows_geom = [((bottom_in + (n_rows - 1 - i) * (h_row + gap_v)) / fig_h, h_row / fig_h) for i in range(n_rows)]
        grid = [[fig.add_axes((x0, y0, w, h)) for x0, w, _ in cols] for y0, h in rows_geom]

        # Rows measuring the SAME quantity share a scale -- "cvar" and "tail" are the same number by
        # construction, so with independent limits they would render at different heights and the claim
        # the layout rests on (this stack IS that bar) silently breaks. A "mean" row against either is a
        # different quantity (~4 vs ~11), and a shared scale would squash it to a sliver, so there each
        # row is scaled on its own.
        #
        # The bottom row's headroom has to clear the sticker, not just the tallest bar. Under a shared
        # scale that widened limit is applied to BOTH rows -- giving the extra room to one alone would
        # make the same numbers render at different heights, which is what sharing exists to prevent.
        #
        # One limit per QUANTITY, taken over every row measuring it, so rows drawn two ways off the same
        # number keep identical bar heights however many rows the render has. The group containing the
        # BOTTOM row gets the sticker's headroom, and all its rows get it together -- handing the extra
        # room to one row of a shared group is exactly what sharing exists to prevent.
        # Headroom is a PER-ROW requirement and the group limit is the max of what its rows each need:
        # only the row that actually carries the stickers has to clear one, and charging every row of
        # its group the sticker's allowance leaves the others visibly empty (in the three-row appendix
        # the mean-cost bars sat at half the panel height). Taking the max still guarantees the sticker
        # row its room, because its own requirement is one of the terms.
        limits: dict[str, float] = {}
        for index, kind in enumerate(rows):
            quantity = ROW_QUANTITY[kind]
            headroom = sticker_headroom if index == len(rows) - 1 else 1.10
            limits[quantity] = max(limits.get(quantity, 0.0), extents[index] * headroom)
        row_lims = [limits[ROW_QUANTITY[kind]] for kind in rows]

        for row, ylim in zip(grid, row_lims, strict=True):
            for a, (_, _, keys) in zip(row, cols, strict=True):
                a.set_ylim(0, ylim)
                # The ours panel is widened for its label, so its xlim has to grow with it -- centring
                # the single slot in the WIDER box, or the bar drifts left of its own tick.
                own = pitches[tuple(keys)]
                extra = (spans[-1] - ((len(keys) - 1) * own + 2 * pad)) / 2 if keys is panels[-1] else 0.0
                a.set_xlim(-pad - extra, (len(keys) - 1) * own + pad + extra)
                a.set_xticks(np.arange(len(keys), dtype=float) * own)
                a.grid(True, axis="y", color="0.92", lw=0.5 * sizes.scale)
                a.set_axisbelow(True)
                for side in ("top", "right"):
                    a.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    a.spines[side].set_linewidth(sizes.spine)
                    a.spines[side].set_color("black")
                a.yaxis.set_major_locator(plt.MaxNLocator(4))
            for a in row[1:]:  # the scale is read off the leftmost panel only
                a.spines["left"].set_visible(False)
                a.tick_params(axis="y", which="both", length=0, labelleft=False)
                a.set_yticks(row[0].get_yticks())
                a.set_ylim(row[0].get_ylim())

        def whisker(a: plt.Axes, x: float, mu: float, se: float) -> None:
            """The +-1 se bracket. Drawn on any row whose total IS the aggregated metric, stacked or not."""
            stroke: dict[str, Any] = {"color": "0.25", "lw": 0.6 * sizes.scale, "zorder": 4}
            a.plot([x, x], [mu - se, mu + se], **stroke)
            a.plot([x - half * 0.4, x + half * 0.4], [mu + se] * 2, **stroke)
            a.plot([x - half * 0.4, x + half * 0.4], [mu - se] * 2, **stroke)

        fig.canvas.draw()  # bars are placed in axes coords, so the layout has to be final first
        for row, kind in zip(grid, rows, strict=True):
            for a, (_, _, keys) in zip(row, cols, strict=True):
                for slot, key in enumerate(keys):
                    x, (mu, se, values) = slot * pitches[tuple(keys)], stats[ROW_METRIC[kind]][key]
                    if kind in SOLID_KINDS:
                        _rounded_bar(a, x - half, x + half, mu, arm_color(key), radius_in=_BAR_RADIUS_IN)
                        whisker(a, x, mu, se)
                        # the per-seed values, kept: they are what shows the point ablation is BIMODAL
                        # across seeds rather than merely noisy, the pathology R4 is built to
                        # demonstrate. A stacked row cannot carry them -- five hues plus scattered dots
                        # is past what a 0.7in slot reads at -- so they are the solid row's alone.
                        # A dot is a mark like the whisker's stroke, so it scales with the strokes
                        # (its area with their square) and prints at one size in every render.
                        a.scatter(
                            [x] * len(values),
                            values,
                            color="0.15",
                            s=2.2 * sizes.scale**2,
                            zorder=5,
                            alpha=0.55,
                            linewidths=0,
                        )
                    else:
                        bounds = list(np.concatenate([[0.0], np.cumsum(stacks[kind][key])]))
                        _rounded_stack(
                            a, x - half, x + half, bounds, segment_colors, _BAR_RADIUS_IN, hairline=0.5 * sizes.scale
                        )
                        # A "tail" stack totals exactly the CVaR, so the whisker still describes its
                        # height -- but only worth drawing when the render has no SOLID row on the same
                        # quantity, since against one it repeats a bracket the reader already has on
                        # that number a row away. A "mean" stack gets none: it totals the cost of the
                        # MISTAKES, which is less than the mean cost (correct decisions cost too), so
                        # the mean's whisker would float above the bar.
                        solid_twin = any(ROW_QUANTITY[k] == ROW_QUANTITY[kind] for k in rows if k in SOLID_KINDS)
                        if kind == "tail" and not solid_twin:
                            whisker(a, x, mu, se)

        # Slot centres are converted through the axes' own transform rather than worked out from pad and
        # pitch by hand, so a badge stays over its bars if either is retuned.
        badge_pad_in = _STICKER_PAD_EM * sizes.tick / 72
        for a, (_, _, keys), text, slots in zip(grid[-1], cols, stickers, PAPER_STICKER_SLOTS, strict=True):
            covered = range(len(keys)) if slots is None else slots
            centre = (min(covered) + max(covered)) / 2 * pitches[tuple(keys)]
            # A one-line badge wider than the room around its centre -- twice the distance to the
            # nearer edge of its panel -- would run over the y axis or into the next panel, so it
            # takes two lines there. Its position, which is the scope of its claim, does not move.
            x_lo, x_hi = a.get_xlim()
            room_in = 2 * min(centre - x_lo, x_hi - centre) * unit_in - 2 * badge_pad_in
            text = _fit_two_lines(text, FP_REGULAR, sizes.tick, room_in)
            _sticker(a, text, sizes, float((a.transData + a.transAxes.inverted()).transform((centre, 0))[0]))

        for row in grid[:-1]:  # method names belong to the bottom row only; the rows above share them
            for a in row:
                a.tick_params(axis="x", length=0, labelbottom=False)
        for a, (_, _, keys) in zip(grid[-1], cols, strict=True):
            a.set_xticklabels([tick_labels[k] for k in keys], multialignment="center")
            a.tick_params(axis="x", width=sizes.spine, length=sizes.tick_length, color="black", pad=1.2 * type_ratio)
        for a in (a for row in grid for a in row):
            a.tick_params(axis="y", width=sizes.spine, length=sizes.tick_length, color="black", pad=3.5 * type_ratio)
            for label in (*a.get_xticklabels(), *a.get_yticklabels()):
                label.set_fontproperties(FP_LIGHT)
                label.set_fontsize(sizes.tick)
                label.set_color("black")
        for row in grid:
            for a in row[1:]:
                a.tick_params(axis="y", which="both", length=0)
        # Ours is marked the way the paper marks it everywhere: the whole label in the semibold face,
        # name, subscript and tag. In black like the names beside it, not the crimson of its bar: the
        # bar already carries the colour, and a saturated hue at tick size reads LIGHTER than black,
        # so repeating it there would invert the emphasis.
        for a, (_, _, keys) in zip(grid[-1], cols, strict=True):
            if FLAGSHIP_ARM in keys:
                ours_tick = a.get_xticklabels()[keys.index(FLAGSHIP_ARM)]
                ours_tick.set_fontproperties(FP_SEMIBOLD)
                ours_tick.set_fontsize(sizes.tick)

        nudge = ScaledTranslation(0, _TICK_NUDGE_PT * type_ratio / 72, fig.dpi_scale_trans)
        for a in (row[0] for row in grid):  # only the left column carries y numbers
            for label in a.get_yticklabels():
                label.set_transform(label.get_transform() + nudge)

        # Each row is named left of its y numbers. A name too long for its row at this type takes
        # two lines rather than running into the rows above and below it, and every name starts on
        # the same vertical line, a gap and the deepest name's further lines clear of the widest
        # number in ANY row -- matplotlib would set each label off its own row's numbers and
        # stagger them.
        names = [_fit_two_lines(labels[kind], FP_SEMIBOLD, sizes.label, h_row) for kind in rows]
        fig.canvas.draw()  # the y numbers have their final text and place only once drawn
        numbers = [label for row in grid for label in row[0].get_yticklabels() if label.get_text()]
        numbers_in = min(label.get_window_extent().x0 for label in numbers) / fig.dpi
        further_lines = max(name.count("\n") for name in names)
        name_in = numbers_in - (_ROW_NAME_GAP + further_lines * _ROW_NAME_PITCH) * sizes.label / 72
        for (y0, height), name in zip(rows_geom, names, strict=True):
            _row_name(fig, name_in, (y0 + height / 2) * fig_h, name, sizes)

        _legend_ladder(fig, sizes)
        return fig

    _save_for_print(build, f"cost_sensitive_triage_paper{suffix}", print_width)


# Figure 1 tells the headline cell as the introduction's argument in three steps, one card each:
# (title, verdict, arms). Risk-neutral prediction is cheap on average and catastrophic in the tail;
# risk aversion on a POINT prediction (at training time, SQwash, or at decision time, our rule on the
# MLE) buys the tail back by treating nearly everybody; adding the credal set keeps the tail AND the
# average. The verdicts are one sentence pattern with two slots, so the three cards read as the
# truth table the paper is about: only the last one has both slots right.
INTRO_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("risk-neutral", "many catastrophes,\nlow average cost", ("mle|best_response",)),
    ("risk-averse", "few catastrophes,\nhigh average cost", ("sqwash|best_response", "mle|cvar_minimax_point")),
    ("credal risk-averse", "few catastrophes,\nlow average cost", (FLAGSHIP_ARM,)),
)
# An arm is labelled as a pipeline on ONE line: the predictor and, for OURS_ARMS, "+ rule" after it,
# the same words under both arms that use it. No group header says "CVaR-minimax" here, so the label
# has to. SEMIBOLD marks exactly what the paper introduces, the rule and the CreWare predictor, and
# nothing else: on the MLE arm only the rule is bold, while ours is semibold as a whole, which is what
# "[ours]" used to say. CreWare stands without its alpha subscript here; the appendix figure and the
# text carry the value. The predictor and the rule are kept apart because fig_intro sets them as
# separate texts (see there), not as one mathtext label like paper_style.OUR_RULE.
INTRO_ARM_LABELS = {
    "mle|best_response": "MLE",
    "sqwash|best_response": "SQwash",
    "mle|cvar_minimax_point": "MLE",
    FLAGSHIP_ARM: "CreWare",
}
INTRO_RULE_LABEL = "+ rule"
# MISTAKE_CELLS folded to the three DIRECTIONS a decision can be wrong in, as (label, colour, member
# indices): too little care for sepsis, too much for the healthy, and the cost-4 flu cell that is
# neutral between the two. The two cells this drops as separate entries (sepsis as flu, healthy as
# flu) are slivers at the headline cell -- at most 0.09 of a 1.1-4.3 bar -- and the appendix figure
# keeps them. Colours are the dominant member's, so a block means the same thing in both figures
# (the flu cell's is FLU_MISTREATED_COLOR in both).
# The figure carries no cost numbers, so the first label has to say in words what 20 and 14 said:
# this block is the "catastrophes" the card verdicts count.
INTRO_MISTAKES: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("sepsis undertreated (catastrophic)", "#67000d", (0, 1)),
    ("healthy overtreated", "#fc8d59", (2, 3)),
    ("flu mistreated", FLU_MISTREATED_COLOR, (4,)),
)


# Height of Fira Sans' ascenders in ems of its type: how far under its top a line's baseline sits.
_ASCENDER_EM = 0.74


def fig_intro(records: list[dict[str, Any]], cfg: SimpleNamespace) -> None:  # noqa: PLR0914, PLR0915
    """Figure 1: fig_paper's ("cvar", "mean") render reduced to four arms on three cards.

    Same headline cell, same two quantities and the same bar styling as fig_paper, which stays the
    appendix figure. What changes is the layout: the arms sit on three lightly tinted cards, read
    left to right (INTRO_GROUPS), each headed by what the approach is and what it amounts to, and
    the cost decomposition is folded to three kinds of mistake (INTRO_MISTAKES). The per-seed dots
    are left to the appendix as well; the +-1 se whisker stays.

    Drawn for one column of the main text (INTRO_PRINT_WIDTH), with the type sized to print there
    like every other figure's.
    """
    tau, hs, hf = cfg.headline_tau, cfg.headline_s_silent, cfg.headline_f_b
    plotted = [key for _, _, keys in INTRO_GROUPS for key in keys]
    stats = {k: agg(records, k, hf, hs, tau, "cvar_cost") for k in plotted}
    stacks = {}
    for key in plotted:
        cells = mistake_profile(records, key, hf, hs, tau, cost_weighted=True)
        stacks[key] = [sum(cells[i] for i in members) for _label, _color, members in INTRO_MISTAKES]
    row_lims = (
        max(stats[k][0] + stats[k][1] for k in plotted) * 1.10,
        max(sum(stacks[k]) for k in plotted) * 1.10,
    )
    row_names = ((f"CVaR$_{{{tau}}}$ cost", None), ("mean cost", "by mistake"))
    segment_colors = [color for _label, color, _members in INTRO_MISTAKES]

    def build(sizes: Sizes) -> plt.Figure:  # noqa: PLR0914, PLR0915
        """Draw the figure with its type at the given sizes; everything above is data."""
        use_fira_mathtext()
        # The type against the type fig_paper's layout was tuned for, which this one inherits its
        # tick and offset distances from.
        type_ratio = _type_ratio(sizes)

        # Geometry in INCHES throughout, and every axes' x data unit IS an inch, so a bar is fig_paper's
        # physical width (2 * 0.16 slots of 0.7007in) on a one-bar card and on the two-bar card alike.
        # Card widths follow from their text, not their bar count: the header binds on the last card
        # and the verdict on the first.
        bar_w, pitch = 0.224, 0.58
        card_w = (0.92, 1.36, 1.04)
        left_in, right_in, gap_in, inset = 0.42, 0.03, 0.12, 0.07
        h_row, gap_v, top_in = 0.65, 0.12, 0.02
        # The bands that hold text are as tall as their text: a card's head is its title over the
        # two verdict lines, and under the bars sit a tick and the one line of arm label.
        tick_pad, line_pitch, verdict_pitch = 1.2 * type_ratio, 1.2, 1.25  # points, and line pitches in ems
        title_in = 0.07
        verdict_in = title_in + 1.3 * sizes.header / 72
        head_in = verdict_in + 2 * verdict_pitch * sizes.tick / 72 + 0.04
        label_in = (sizes.tick_length + tick_pad + line_pitch * sizes.tick) / 72 + 0.045
        legend_in = 0.19 * type_ratio
        fig_w = left_in + sum(card_w) + (len(card_w) - 1) * gap_in + right_in
        fig_h = legend_in + label_in + 2 * h_row + gap_v + head_in + top_in
        fig = plt.figure(figsize=(fig_w, fig_h))
        # One figure-sized axes in inch units carries everything that is not a bar: the cards, their
        # headers and the row labels.
        canvas = fig.add_axes((0, 0, 1, 1))
        canvas.set_xlim(0, fig_w)
        canvas.set_ylim(0, fig_h)
        canvas.axis("off")

        card_y0, card_y1 = legend_in, fig_h - top_in
        row_y0 = (card_y0 + label_in + h_row + gap_v, card_y0 + label_in)  # top row first
        card_x0 = [left_in + sum(card_w[:i]) + i * gap_in for i in range(len(card_w))]
        ours_tint = matplotlib.colors.to_hex(_lighten(OURS_COLOR, 0.93))
        grid: list[list[plt.Axes]] = [[], []]
        for (title, verdict, keys), x0, width in zip(INTRO_GROUPS, card_x0, card_w, strict=True):
            # Very light, so the cards group without competing with the bars. Ours takes a tint of its
            # own crimson instead of the gray: the sequence has a destination, and this marks it.
            canvas.add_patch(
                matplotlib.patches.FancyBboxPatch(
                    (x0, card_y0),
                    width,
                    card_y1 - card_y0,
                    boxstyle="round,pad=0,rounding_size=0.06",
                    linewidth=0,
                    facecolor=ours_tint if FLAGSHIP_ARM in keys else "0.955",
                )
            )
            canvas.text(
                x0 + width / 2,
                card_y1 - title_in,
                title,
                ha="center",
                va="top",
                fontproperties=FP_SEMIBOLD,
                fontsize=sizes.header,
                color="black",
            )
            canvas.text(
                x0 + width / 2,
                card_y1 - verdict_in,
                verdict,
                ha="center",
                va="top",
                multialignment="center",
                fontproperties=FP_REGULAR,
                fontsize=sizes.tick,
                color="0.35",
                linespacing=verdict_pitch,
            )
            for row, y0 in zip(grid, row_y0, strict=True):
                a = fig.add_axes(((x0 + inset) / fig_w, y0 / fig_h, (width - 2 * inset) / fig_w, h_row / fig_h))
                a.set_facecolor("none")
                a.set_xlim(0, width - 2 * inset)
                a.set_xticks(
                    [(width - 2 * inset) / 2 + (slot - (len(keys) - 1) / 2) * pitch for slot in range(len(keys))]
                )
                row.append(a)

        nudge = ScaledTranslation(0, _TICK_NUDGE_PT * type_ratio / 72, fig.dpi_scale_trans)
        for row, ylim in zip(grid, row_lims, strict=True):
            row[0].set_ylim(0, ylim)
            row[0].yaxis.set_major_locator(plt.MaxNLocator(3, steps=[1, 2, 5, 10]))
            for a in row:
                a.set_yticks(row[0].get_yticks())
                a.set_ylim(0, ylim)
                # White rules on the tinted card, not gray ones on white: the card is the surface here.
                a.grid(True, axis="y", color="white", lw=0.6 * sizes.scale)
                a.set_axisbelow(True)
                for side in ("top", "right", "left"):
                    a.spines[side].set_visible(False)
                a.spines["bottom"].set_linewidth(sizes.spine)
                a.spines["bottom"].set_color("black")
                a.tick_params(axis="y", which="both", length=0, labelleft=False)
            # The scale is read off the first card only, and its numbers sit OUTSIDE the card, in the
            # margin with the row label, so the cards hold nothing but bars and names.
            row[0].tick_params(axis="y", labelleft=True, pad=inset * 72 + 2.5 * type_ratio)
            for label in row[0].get_yticklabels():
                label.set_fontproperties(FP_LIGHT)
                label.set_fontsize(sizes.tick)
                label.set_color("black")
                label.set_transform(label.get_transform() + nudge)

        # Both row names are set on one baseline (_row_name), so they sit on one vertical line, and
        # so do the note and the arrow beside them: the note on a baseline of its own, the arrow
        # where the middle of the note's letters is.
        name_x, detail_x, arrow_in = 0.085 * type_ratio, 0.185 * type_ratio, 0.15
        arrow_x = detail_x - 0.3 * sizes.tick / 72
        arrow = {
            "arrowstyle": "-|>",
            "color": "0.4",
            "lw": 0.7 * sizes.scale,
            "mutation_scale": 5.5 * type_ratio,
            "shrinkA": 0,
            "shrinkB": 0,
        }
        for (name, detail), y0 in zip(row_names, row_y0, strict=True):
            y_mid = y0 + h_row / 2
            _row_name(fig, name_x, y_mid, name, sizes)
            if detail is None:
                canvas.annotate(
                    "", xy=(arrow_x, y_mid - arrow_in / 2), xytext=(arrow_x, y_mid + arrow_in / 2), arrowprops=arrow
                )
            else:
                canvas.text(
                    detail_x,
                    y_mid,
                    detail,
                    rotation=90,
                    rotation_mode="anchor",
                    ha="center",
                    va="baseline",
                    fontproperties=FP_REGULAR,
                    fontsize=sizes.tick,
                    color="0.35",
                )

        whisker: dict[str, Any] = {"color": "0.25", "lw": 0.6 * sizes.scale, "zorder": 4}

        def run_pt(text: str, face: FontProperties) -> float:
            """Width in points of a label run, measured at ten times its size to beat pixel rounding."""
            return _text_width_in(text, face, 10 * sizes.tick) * 72 / 10

        space_pt = run_pt("a b", FP_LIGHT) - run_pt("ab", FP_LIGHT)
        label_drop_pt = sizes.tick_length + tick_pad + _ASCENDER_EM * sizes.tick
        fig.canvas.draw()  # bars are placed in axes coords, so the layout has to be final first
        for (_, _, keys), top, bottom in zip(INTRO_GROUPS, *grid, strict=True):
            for x, key in zip(top.get_xticks(), keys, strict=True):
                mu, se, _values = stats[key]
                # The full pipeline in crimson and the baselines in the one colour they share
                # (INTRO_BASELINE_COLOR), so the eye lands on ours. The MLE under our rule is half
                # ours, and wears the light tint of our red that fig_paper gives the rule ablations.
                if key == FLAGSHIP_ARM:
                    color = OURS_COLOR
                elif key in OURS_ARMS:
                    color = matplotlib.colors.to_hex(_lighten(OURS_COLOR, _TEST_TINT))
                else:
                    color = INTRO_BASELINE_COLOR
                _rounded_bar(top, x - bar_w / 2, x + bar_w / 2, mu, color, radius_in=_BAR_RADIUS_IN)
                top.plot([x, x], [mu - se, mu + se], **whisker)
                for y in (mu - se, mu + se):
                    top.plot([x - bar_w * 0.2, x + bar_w * 0.2], [y, y], **whisker)
                bounds = list(np.concatenate([[0.0], np.cumsum(stacks[key])]))
                _rounded_stack(
                    bottom,
                    x - bar_w / 2,
                    x + bar_w / 2,
                    bounds,
                    segment_colors,
                    _BAR_RADIUS_IN,
                    hairline=0.5 * sizes.scale,
                )

            top.tick_params(axis="x", length=0, labelbottom=False)
            bottom.tick_params(axis="x", width=sizes.spine, length=sizes.tick_length, color="black", labelbottom=False)
            # The arm labels are plain texts on ONE baseline, a run per face, not tick labels. A label
            # that is half bold would otherwise be mathtext, and the vector backends set a mathtext
            # line up to a point above the plain "SQwash" beside it (its height is rounded up to
            # whole points before the glyphs are placed), whichever way the two are aligned.
            for x, key in zip(bottom.get_xticks(), keys, strict=True):
                if key == FLAGSHIP_ARM:
                    runs = [(f"{INTRO_ARM_LABELS[key]} {INTRO_RULE_LABEL}", FP_SEMIBOLD)]
                else:
                    runs = [(INTRO_ARM_LABELS[key], FP_LIGHT)]
                    if key in OURS_ARMS:
                        runs.append((INTRO_RULE_LABEL, FP_SEMIBOLD))
                widths = [run_pt(text, face) for text, face in runs]
                left_pt = -(sum(widths) + space_pt * (len(runs) - 1)) / 2
                for (text, face), width in zip(runs, widths, strict=True):
                    bottom.annotate(
                        text,
                        xy=(x, 0),
                        xycoords=("data", "axes fraction"),
                        xytext=(left_pt, -label_drop_pt),
                        textcoords="offset points",
                        ha="left",
                        va="baseline",
                        fontproperties=face,
                        fontsize=sizes.tick,
                        color="black",
                    )
                    left_pt += width + space_pt

        # The key sits in its own band under the cards, its top edge a little below them.
        swatches = [(color, label) for label, color, _members in INTRO_MISTAKES]
        _swatch_legend(fig, swatches, sizes, top=(legend_in - 0.08 * type_ratio) / fig_h)
        return fig

    _save_for_print(build, "cost_sensitive_triage_intro", INTRO_PRINT_WIDTH)


def _arm_legend_label(key: str, cfg: SimpleNamespace) -> str:
    """An arm's name in a paper figure, bar tick or legend entry alike; see PAPER_ARM_LABELS."""
    if key == FLAGSHIP_ARM:
        return _flagship_label(cfg, rule=True)
    return PAPER_ARM_LABELS.get(key, ARM_LABELS[key])


def fig_mechanism_map(map_data: dict[str, Any], cfg: SimpleNamespace) -> None:
    """Decision regions of the MLE and CreRL + ours, evidence mass and set width over the plane."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    x1, x2 = map_data["x1"], map_data["x2"]
    extent = (x1[0], x1[-1], x2[0], x2[-1])
    action_cmap = matplotlib.colors.ListedColormap(["#dddddd", "#ffd580", "#80b3ff"])

    for ax, grid, title in zip(
        axes[0],
        (map_data["mle_actions"], map_data["crerl_actions"]),
        ("MLE best response", f"CreRL + ours (v = {map_data['crerl_v']:.2f})"),
        strict=True,
    ):
        ax.imshow(grid, origin="lower", extent=extent, cmap=action_cmap, vmin=0, vmax=2, aspect="auto")
        scatter_train(ax, map_data)
        ax.set_title(title, fontsize=9)
    fig.text(0.5, 0.925, "decision regions: grey = No Action, orange = Treat Flu, blue = ICU", ha="center", fontsize=8)

    im = axes[1][0].imshow(
        np.log10(np.maximum(map_data["mass"], 1e-8)), origin="lower", extent=extent, cmap="viridis", aspect="auto"
    )
    fig.colorbar(im, ax=axes[1][0], label="log10 evidence mass")
    scatter_train(axes[1][0], map_data)
    axes[1][0].set_title("CreRL evidence mass", fontsize=9)

    im = axes[1][1].imshow(map_data["width"], origin="lower", extent=extent, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=axes[1][1], label="L1 set width")
    scatter_train(axes[1][1], map_data)
    axes[1][1].set_title(f"CreRL set width (alpha = {cfg.alpha})", fontsize=9)
    fig.suptitle(f"Mechanism map, headline cell, seed {cfg.map_seed}", fontsize=10)
    _save(fig, "cost_sensitive_triage_map")


def scatter_train(ax: plt.Axes, map_data: dict[str, Any]) -> None:
    """Overlay the training archive, class-coloured, sepsis components emphasised."""
    x, y_, comp = map_data["train_x"], map_data["train_y"], map_data["train_comp"]
    for label, color, name in ((0, "green", "Healthy"), (1, "olive", "Flu"), (2, "red", "Sepsis")):
        mask = y_ == label
        ax.scatter(x[mask, 0], x[mask, 1], s=3, color=color, alpha=0.25, label=name)
    b_mask = comp == COMPONENTS.index("sepsis_b")
    if b_mask.any():
        ax.scatter(x[b_mask, 0], x[b_mask, 1], s=14, color="darkred", marker="x", label="subgroup sepsis (train)")


# Tick labels for the paper figure, sitting horizontally under ~0.7in slots. The repo-internal
# ARM_LABELS above stay authoritative for the diagnostic figures; here the paper's own names apply,
# and the kernel-evidence credal predictor is called CreWare (its label is built from the config,
# see _flagship_label, so the alpha subscript can never go stale).
PAPER_ARM_LABELS = {
    "mle|best_response": "MLE",
    "sqwash|best_response": "SQwash",
    "adacvar|best_response": "AdaCVaR",
    # The three arms that use our decision rule say so in their name: no header above the panels
    # does it for them, and without it "MLE" would stand under two panels. The rule is OURS on all
    # three, the predictor only on the last, so on these two the bold falls on "+ rule" alone
    # (paper_style.OUR_RULE), while ours is semibold as a whole (_flagship_label).
    "mle|cvar_minimax_point": f"MLE {OUR_RULE}",
    "crewra|cvar_minimax": f"CreWra {OUR_RULE}",
    # crerl|cvar_minimax's label is built from the config, see _flagship_label.
}


def make_figures(results: dict[str, Any]) -> None:
    """Render every figure from a results dict."""
    cfg = SimpleNamespace(**results["config"])
    records = collect(results["cells"])
    fig_headline_bars(records, cfg)
    fig_epistemic_sweep(records, cfg)
    fig_aleatoric_sweep(records, cfg)
    fig_paper(records, cfg)
    # The "_POP" render is the same layout with the bottom row decomposing the whole POPulation
    # instead of the tail, i.e. the mean-cost price of the tail win. Both are kept: the tail
    # decomposition is the tighter claim, the population one is the version that shows the
    # baselines are cheap on average, which is what the panel stickers assert.
    fig_paper(records, cfg, suffix="_POP", rows=("cvar", "mean"))
    # Both rows decomposed: the only render that shows WHICH mistakes drive the tail and which
    # drive the mean at the same time, at the cost of the solid row's per-seed dots.
    fig_paper(records, cfg, suffix="_BOTH", rows=("tail", "mean"))
    fig_paper(records, cfg, tau=0.05, suffix="_tau005")
    fig_intro(records, cfg)
    # The APPENDIX bar figures: the original four-panel figure's three bar plots -- tail cost, its
    # mean-cost price, and the decisions that price is made of -- in one three-row column, styled
    # like the paper figure. One per archive cell: fb05 is the headline (the subgroup is all but
    # missing), fb25 the partially-archived one, which is where the average-cost premium the
    # headline pays has already gone away. Each carries stickers measured at its OWN cell.
    # Both carry the SAME y limits, taken over both cells, so the pair can be read side by side: a
    # bar that is half as tall really is half the cost. The 5% cell sets every limit in practice
    # (its CVaR-minimax arms pay ~4.6 mean cost against ~1.8 at 25%), but the limits are computed
    # over the pair rather than copied from one figure to the other, so neither render depends on
    # the other having been drawn first.
    # The paper sets the two side by side, each at just under half the text width, so their type is
    # drawn for that width rather than for the full one the two-row renders above are included at.
    appendix_rows = ("cvar", "mean_bar", "mean")
    appendix_cells = (cfg.headline_f_b, 0.25)
    fig_paper(
        records,
        cfg,
        suffix="_appendix_fb05",
        rows=appendix_rows,
        limit_f_bs=appendix_cells,
        print_width=APPENDIX_PRINT_WIDTH,
    )
    fig_paper(
        records,
        cfg,
        suffix="_appendix_fb25",
        rows=appendix_rows,
        f_b=0.25,
        stickers=PAPER_STICKERS_FB25,
        limit_f_bs=appendix_cells,
        print_width=APPENDIX_PRINT_WIDTH,
    )
    fig_line_curves(records, cfg)
    maps = [cell["map"] for cell in results["cells"] if "map" in cell]
    if maps:
        fig_mechanism_map(maps[0], cfg)


def print_diagnostics(cells: list[dict[str, Any]], cfg: SimpleNamespace) -> None:
    """Mechanism summary of the headline cell: mass, width and MLP confidence per component."""
    headline = [c for c in cells if c["f_b"] == cfg.headline_f_b and c["s_silent"] == cfg.headline_s_silent]
    if not headline:
        return
    print("\n=== Mechanism diagnostics, headline cell (seed means over test split) ===")
    print(f"  whitening gamma: {[c['whitening_gamma'] for c in headline]}")
    print(f"  sepsis-B train rows kept: {[c['n_train_sepsis_b'] for c in headline]}")
    for field, label in (
        ("mass_by_component", "evidence mass"),
        ("width_by_component", "CreRL set L1 width"),
        ("crewra_width_by_component", "CreWra set L1 width"),
        ("mlp_confidence_by_component", "MLP max prob"),
    ):
        means = {name: float(np.nanmean([c[field][name] for c in headline])) for name in COMPONENTS}
        print(f"  {label}: " + ", ".join(f"{name}={value:.3f}" for name, value in means.items()))


def main() -> None:
    """Run the sweep (or re-plot), save results, render figures, verify the goal conditions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny run: 2 seeds, fewer epochs, fewer cells")
    parser.add_argument("--plot-only", action="store_true", help="re-render figures from the saved pickle")
    args = parser.parse_args()
    # paper_style.save reports the size each paper figure's labels print at; show it.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results_path = RESULTS_PATH / RESULTS_FILE
    if args.plot_only:
        results = joblib.load(results_path)
        make_figures(results)
        verify(collect(results["cells"]), SimpleNamespace(**results["config"]))
        return

    if args.smoke:
        C.seeds = [1, 2]
        C.epochs = 40
        C.n_train, C.n_val, C.n_test = 2000, 1000, 4000
        C.f_b_sweep = [0.0, C.headline_f_b, 1.0]
        C.silent_sweep = [0.0, C.headline_s_silent]
        # Its own file: a smoke run must never clobber the full run's pickle (it did once --
        # the figures and verification silently ran on 2-seed smoke data afterwards).
        results_path = RESULTS_PATH / RESULTS_FILE.replace(".pkl", "_smoke.pkl")

    cells_spec = [(f_b, C.headline_s_silent) for f_b in C.f_b_sweep]
    cells_spec += [(1.0, s) for s in C.silent_sweep if (1.0, s) not in cells_spec]
    jobs = [(seed, f_b, s_silent) for seed in C.seeds for f_b, s_silent in cells_spec]
    print(f"Running {len(jobs)} cells ({len(C.seeds)} seeds x {len(cells_spec)} knob settings) ...")
    cells = Parallel(n_jobs=C.n_jobs, verbose=5)(
        delayed(run_cell)(seed, f_b, s_silent, C) for seed, f_b, s_silent in jobs
    )

    results = {
        "config": {**vars(C), "cost_spec": SPEC.name, "dominance_threshold": DOMINANCE_V},
        "cells": cells,
    }
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, results_path)
    print(f"Results written to {results_path}")

    records = collect(cells)
    print_diagnostics(cells, C)
    dominated = sorted({r["key"] for r in records if r["v_in_dominated_regime"]})
    if dominated:
        print(
            f"WARNING: calibrated v >= dominance threshold {DOMINANCE_V} for arms {dominated}; "
            f"their zero catastrophe counts (if any) measure the cost table, not the method."
        )
    make_figures(results)
    verify(records, C)


if __name__ == "__main__":
    main()
