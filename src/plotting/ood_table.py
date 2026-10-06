r"""OOD detection table: one metric aggregated over seeds, formatted for a LaTeX table.

Rows are (method, alpha) and columns are OOD datasets, so one table covers a whole alpha sweep
against several OOD sets. Cells are the seed mean plus/minus the seed standard deviation, rendered
as $mean \scriptstyle \pm std$ so the result can be pasted into a LaTeX tabular. Numbers come from
the kind="ood" rows of the W&B cache, which experiments/ood_detection.py writes as
ood/<ood_dataset>/<decomposition>/<component>/<metric> summary keys.

Data comes from the W&B cache (see wandb_cache.py). Pure DataFrame in, DataFrame of strings out.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plotting.shift_pareto import _METHOD_STYLES
from plotting.wandb_cache import load_runs

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _collapse_to_latest_run(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per cell identity, from the latest run_id.

    Duplicates arise when an artifact is retrained or ood_detection.py is re-run against it. The
    latest row wins, matching how the shift plots deduplicate. No-op when created_at is missing.

    Args:
        df: Long-form OOD rows.

    Returns:
        The deduplicated frame.
    """
    if "created_at" not in df.columns:
        return df
    identity = ["method.name", "seed", "num_train", "method.train.alpha", "ood_dataset", "metric"]
    df = df.sort_values("created_at", kind="mergesort")
    return df.drop_duplicates(subset=identity, keep="last")


def ood_table(
    df: pd.DataFrame,
    dataset: str,
    ood_datasets: list[str] | None = None,
    methods: list[str] | None = None,
    alphas: list[float] | None = None,
    seeds: list[int] | None = None,
    metric: str = "auroc",
    component: str = "epistemic",
    decomposition: str = "entropy",
    decimals: int = 3,
) -> pd.DataFrame:
    """Aggregate one OOD metric over seeds into a table of mean plus/minus std strings.

    Each cell aggregates the seeds available for that (method, alpha, OOD dataset); cells whose
    seed count differs from the most common one are logged as a warning, so a partially finished
    sweep is visible rather than silently averaged over fewer runs. A method without an alpha knob
    (credal_wrapper) contributes a single row.

    Args:
        df: Long-form DataFrame from load_runs().
        dataset: In-distribution dataset the runs were trained on, for example cifar10.
        ood_datasets: OOD dataset names to use as columns, in this order. None uses every OOD
            dataset present, sorted.
        methods: Method names to use as rows, in this order. None uses every method present, sorted.
        alphas: Restrict alpha-swept methods to these method.train.alpha values. Methods without an
            alpha are always kept. None keeps all alphas present.
        seeds: Restrict to these seeds. None uses all seeds present.
        metric: Which OOD metric to tabulate, one of auroc, aupr, fpr.
        component: Uncertainty component the scores came from, for example epistemic.
        decomposition: Decomposition label the scores came from, for example entropy.
        decimals: Digits after the decimal point for both mean and std.

    Returns:
        DataFrame of formatted strings, indexed by method label (with alpha when the method has
        one) and with one column per OOD dataset. Empty cells are rendered as a LaTeX dash.

    Raises:
        ValueError: no rows match the filters.
    """
    import pandas as pd  # noqa: PLC0415

    sub = df[(df["kind"] == "ood") & (df["dataset"] == dataset)]
    sub = sub[(sub["metric"] == metric) & (sub["component"] == component)]
    if "decomposition" in sub.columns:
        sub = sub[sub["decomposition"] == decomposition]
    if seeds is not None:
        sub = sub[sub["seed"].isin(seeds)]
    if ood_datasets is not None:
        sub = sub[sub["ood_dataset"].isin(ood_datasets)]
    if methods is not None:
        sub = sub[sub["method.name"].isin(methods)]
    if sub.empty:
        raise ValueError(
            f"No OOD rows for dataset={dataset!r}, metric={metric!r}, decomposition={decomposition!r}, "
            f"component={component!r}. These four must match what experiments/ood_detection.py was run "
            "with; then re-sync the cache (python -m plotting.wandb_cache ...)."
        )
    for col in ("method.train.alpha", "num_train"):
        if col not in sub.columns:
            sub = sub.assign(**{col: float("nan")})
    if alphas is not None:
        sub = sub[sub["method.train.alpha"].isin(alphas) | sub["method.train.alpha"].isna()]
    sub = _collapse_to_latest_run(sub)

    stats = (
        sub.groupby(["method.name", "method.train.alpha", "ood_dataset"], dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    modal_count = int(stats["count"].mode().iloc[0])
    for _, row in stats[stats["count"] != modal_count].iterrows():
        logger.warning(
            "Only %d seed(s) for method=%s alpha=%s ood=%s (most cells have %d).",
            int(row["count"]),
            row["method.name"],
            row["method.train.alpha"],
            row["ood_dataset"],
            modal_count,
        )

    # std is NaN for a single seed; report it as zero rather than dropping the cell.
    stats["cell"] = [
        f"${mean:.{decimals}f} \\scriptstyle \\pm {0.0 if pd.isna(std) else std:.{decimals}f}$"
        for mean, std in zip(stats["mean"], stats["std"], strict=True)
    ]
    stats["row_label"] = [
        _METHOD_STYLES.get(name, ("", None, name))[2] + ("" if pd.isna(alpha) else f" ($\\alpha={alpha:g}$)")
        for name, alpha in zip(stats["method.name"], stats["method.train.alpha"], strict=True)
    ]

    table = stats.pivot(index="row_label", columns="ood_dataset", values="cell")
    # Order rows by the requested method order then alpha, and columns by the requested OOD order.
    order = stats[["row_label", "method.name", "method.train.alpha"]].drop_duplicates()
    method_rank = {m: i for i, m in enumerate(methods or sorted(stats["method.name"].unique()))}
    order["rank"] = order["method.name"].map(method_rank).fillna(len(method_rank))
    order = order.sort_values(["rank", "method.train.alpha"], na_position="first")
    table = table.reindex(index=order["row_label"].tolist())
    columns = ood_datasets or sorted(stats["ood_dataset"].unique().tolist())
    table = table.reindex(columns=[c for c in columns if c in table.columns])
    table.index.name = None
    table.columns.name = None
    return table.fillna("--")


# Column order of the paper's OOD table: near-OOD (CIFAR-100, TinyImageNet) then far-OOD
# (MNIST, SVHN, Textures, Places365), matching the tabular header in the paper.
_PAPER_COLUMN_ORDER = ("cifar100", "tin", "mnist", "svhn", "textures", "places365")


def render_paper_rows(table: pd.DataFrame) -> str:
    r"""Render only the tabular body rows, in the paper's fixed column order.

    The surrounding skeleton (table*, header, bottomrule) lives in the paper; this returns just
    the method rows to paste between the header row and the bottom of the tabular. Columns follow
    _PAPER_COLUMN_ORDER regardless of how the table's columns were requested; datasets outside
    that list are appended in their incoming order. Cells arrive from ood_table already formatted
    as $mean \scriptstyle \pm std$ strings; row labels get their underscores escaped so raw
    method names compile as-is.

    Args:
        table: Output of ood_table(), formatted strings indexed by method label.

    Returns:
        One LaTeX row per method, newline-joined, each ending with the row terminator.
    """
    cols = [c for c in _PAPER_COLUMN_ORDER if c in table.columns]
    cols += [c for c in table.columns if c not in _PAPER_COLUMN_ORDER]
    lines = []
    for label, row in table[cols].iterrows():
        lines.append(str(label).replace("_", "\\_") + " & " + " & ".join(str(v) for v in row.tolist()) + " \\\\")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Edit these and re-run. None on filter fields means "no constraint".
    DATASET = "bloodmnist"
    OOD_DATASETS: list[str] | None = [
        "pathmnist",
        "tissuemnist",
        "dermamnist",
        "retinamnist",
        "octmnist",
        "breastmnist",
    ]
    # OOD_DATASETS = ["cifar100"]
    METHODS = ["credal_ensembling", "credal_bnn", "credal_wrapper", "credal_dro"]
    # METHODS = ["credal_relative_likelihood", "efficient_credal_prediction", "credal_rl_multinomial"]
    # METHODS = ["credal_rl_multinomial", "credal_bnn"]
    ALPHAS: list[float] | None = None  # e.g. [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    # ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    ALPHAS = [0.95]
    SEEDS: list[int] | None = [1, 2, 3]
    METRIC = "auroc"  # auroc | aupr | fpr
    # DECOMPOSITION and COMPONENT are part of the summary key ood_detection.py writes, so they must
    # match the decomposition= / component= the runs were evaluated with (its config defaults are
    # entropy / epistemic). Setting them to something never run yields "No OOD rows", not a silent
    # empty table. COMPONENT is total | aleatoric | epistemic (entropy decomposition), size (mean
    # interval width), or leverage (credal_rl_multinomial only); size/leverage runs use decomposition="set".
    DECOMPOSITION = "entropy"
    COMPONENT = "epistemic"

    df = load_runs(filters={"dataset": DATASET})
    print(f"Loaded {len(df)} rows from {df['run_id'].nunique()} runs.\n")
    table = ood_table(
        df,
        dataset=DATASET,
        ood_datasets=OOD_DATASETS,
        methods=METHODS,
        alphas=ALPHAS,
        seeds=SEEDS,
        metric=METRIC,
        component=COMPONENT,
        decomposition=DECOMPOSITION,
    )
    # print(table.to_string(), "\n")
    print("---- paper rows (paste below the Method header row) ----")
    print(render_paper_rows(table))
