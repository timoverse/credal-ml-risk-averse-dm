"""Coverage-efficiency trade-off plot: coverage (x) vs efficiency (y), colored by credal alpha.

One point per (method, relative-likelihood alpha): x is the mean coverage across seeds (fraction
of test instances whose first-order target distribution lies in the predicted credal set, see
experiments/coverage_efficiency.py), y is the mean set-size efficiency 1 - mean(upper - lower).
Error bars show plus/minus one std across seeds. Alpha-sweep methods (credal_rl_multinomial,
efficient_credal_prediction, ...) are drawn as one marker per alpha, colored by alpha on a shared
colorbar and connected in alpha order by a light guide line: a small alpha is a weak likelihood
cut, hence a large credal set (high coverage, low efficiency), while alpha near 1 shrinks the set
toward the MLE (low coverage, high efficiency). Methods without an alpha (credal_wrapper's
ensemble box, ...) appear as a single labeled baseline point. Markers distinguish methods.

Data comes from the W&B cache (see wandb_cache.py), where cov_eff/coverage and cov_eff/efficiency
appear as run-level scalar summary columns. Refresh the cache after new evaluations with:
    python -m plotting.wandb_cache --entity risk-averse --project paper
Pure DataFrame in, matplotlib figure out.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plotting.wandb_cache import load_runs

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

# Fixed marker per known method so figures stay comparable across plots; unknown methods cycle.
_METHOD_MARKERS = {
    "credal_rl_multinomial": "o",
    "efficient_credal_prediction": "s",
    "credal_wrapper": "*",
    "credal_ensembling": "P",
}
_FALLBACK_MARKERS = ("v", "^", "<", ">", "X", "d")


def _method_stats(df: pd.DataFrame, method: str, seeds: list[int] | None) -> pd.DataFrame:
    """Per-alpha mean/std of coverage and efficiency for one method (single row when it has no alpha).

    Rows are deduped to the latest run per (alpha, seed, num_train) identity. Methods without a
    method.train.alpha (e.g. credal_wrapper) fall into one group with alpha = NaN.

    Args:
        df: DataFrame from load_runs().
        method: method.name to select.
        seeds: Restrict to these seeds. None = all seeds present.

    Returns:
        DataFrame with columns alpha (NaN for no-alpha methods), coverage, coverage_std,
        efficiency, efficiency_std, and n_seeds (distinct seeds averaged per group), sorted by
        alpha. Empty if nothing matches.
    """
    sub = df[df["method.name"] == method]
    if seeds is not None:
        sub = sub[sub["seed"].isin(seeds)]
    sub = sub.dropna(subset=["cov_eff/coverage", "cov_eff/efficiency"])
    sub = sub.drop_duplicates(subset=["run_id"]).copy()
    if "method.train.alpha" not in sub.columns:
        sub["method.train.alpha"] = float("nan")
    # NaN never equals NaN, so a sentinel keeps no-alpha runs in one dedup/group bucket.
    sub["_alpha_key"] = sub["method.train.alpha"].fillna(-1.0)
    if "created_at" in sub.columns:
        sub = sub.sort_values("created_at", kind="mergesort")
    identity = [c for c in ("_alpha_key", "seed", "num_train") if c in sub.columns]
    sub = sub.drop_duplicates(subset=identity, keep="last")
    if sub.empty:
        return sub
    stats = (
        sub.groupby("_alpha_key")
        .agg(
            alpha=("method.train.alpha", "mean"),
            coverage=("cov_eff/coverage", "mean"),
            coverage_std=("cov_eff/coverage", "std"),
            efficiency=("cov_eff/efficiency", "mean"),
            efficiency_std=("cov_eff/efficiency", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index(drop=True)
        .sort_values("alpha")
    )
    stats[["coverage_std", "efficiency_std"]] = stats[["coverage_std", "efficiency_std"]].fillna(0.0)
    return stats


def plot_coverage_efficiency(
    df: pd.DataFrame,
    methods: str | list[str] = "credal_rl_multinomial",
    seeds: list[int] | None = None,
    cmap: str = "viridis",
    ax: Axes | None = None,
) -> Axes:
    """Plot the coverage-efficiency trade-off of one or more methods, colored by alpha where present.

    Alpha-sweep methods get one marker per alpha (shared colorbar, guide line in alpha order);
    methods without an alpha are single baseline points. Calls plt.show() at the end when ax is
    None (i.e. when this function created the figure); if ax is supplied, no show is called so the
    caller can compose subplots.

    Args:
        df: DataFrame from load_runs(). cov_eff/coverage and cov_eff/efficiency are run-level
            scalar columns duplicated onto every row of a run, so rows are deduped per run here.
        methods: method.name values to overlay; a single string plots one method.
        seeds: Restrict to these seeds. None = all seeds present in df.
        cmap: Matplotlib colormap name used for the alpha coloring.
        ax: Existing matplotlib Axes to draw onto. None creates a new figure and calls plt.show().

    Returns:
        The Axes the plot was drawn onto.

    Raises:
        ValueError: The cov_eff columns are missing from the cache, or no rows remain for any of
            the given methods and seeds.
    """
    created_own_fig = ax is None
    for col in ("cov_eff/coverage", "cov_eff/efficiency"):
        if col not in df.columns:
            raise ValueError(
                f"Column {col!r} not in the cache. Run experiments/coverage_efficiency.py first, then "
                "refresh the cache via python -m plotting.wandb_cache --entity ... --project ..."
            )
    method_list = [methods] if isinstance(methods, str) else list(methods)
    per_method = {m: _method_stats(df, m, seeds) for m in method_list}
    missing = [m for m, s in per_method.items() if s.empty]
    if missing:
        logger.warning("No cov_eff values for %s (seeds=%s); skipping.", missing, seeds)
    per_method = {m: s for m, s in per_method.items() if not s.empty}
    if not per_method:
        raise ValueError(f"No cov_eff values for any of methods={method_list!r} (seeds={seeds}).")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5))

    # One alpha color scale across all sweep methods so their curves are directly comparable.
    sweep_alphas = [a for s in per_method.values() for a in s["alpha"] if a == a]
    norm = Normalize(vmin=min(sweep_alphas), vmax=max(sweep_alphas)) if sweep_alphas else None
    fallback = iter(_FALLBACK_MARKERS)
    legend_handles = []
    scatter = None
    for method, stats in per_method.items():
        # A missing (alpha, seed) evaluation would silently average fewer seeds at that alpha;
        # surface it instead of letting one point quietly carry a different seed set.
        if stats["n_seeds"].nunique() > 1:
            logger.warning(
                "%s: unequal seed counts across alpha groups (n_seeds=%s); some points average "
                "fewer seeds than others.",
                method,
                stats["n_seeds"].tolist(),
            )
        marker = _METHOD_MARKERS.get(method) or next(fallback)
        has_alpha = stats["alpha"].notna().any()
        ax.plot(stats["coverage"], stats["efficiency"], color="0.75", linewidth=1, zorder=1)
        ax.errorbar(
            stats["coverage"],
            stats["efficiency"],
            xerr=stats["coverage_std"],
            yerr=stats["efficiency_std"],
            fmt="none",
            ecolor="0.6",
            capsize=3,
            zorder=2,
        )
        if has_alpha:
            scatter = ax.scatter(
                stats["coverage"],
                stats["efficiency"],
                c=stats["alpha"],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=70,
                zorder=3,
            )
        else:
            ax.scatter(stats["coverage"], stats["efficiency"], color="0.2", marker=marker, s=110, zorder=3)
        legend_handles.append(Line2D([], [], linestyle="none", marker=marker, color="0.2", markersize=8, label=method))
    if scatter is not None:
        ax.figure.colorbar(scatter, ax=ax, label=r"credal set $\alpha$")

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Efficiency")
    title = "Coverage-efficiency trade-off"
    if "dataset" in df.columns:
        datasets = sorted(df["dataset"].dropna().unique().tolist())
        if datasets:
            title = f"{title}\ndataset={','.join(map(str, datasets))}"
    ax.set_title(title)
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    if created_own_fig:
        plt.tight_layout()
        plt.show()
    return ax


if __name__ == "__main__":
    # Edit these and re-run from PyCharm. None on filter fields means "no constraint".
    DATASET: str | None = "cifar10"
    METHODS: list[str] = ["credal_rl_multinomial", "efficient_credal_prediction", "credal_wrapper"]
    SEEDS: list[int] | None = [1, 2, 3]

    filters: dict[str, object] = {"method.name": METHODS}
    if DATASET is not None:
        filters["dataset"] = DATASET
    df = load_runs(filters=filters)
    print(f"Loaded {len(df)} rows from {df['run_id'].nunique()} runs.")
    plot_coverage_efficiency(df, methods=METHODS, seeds=SEEDS)
