r"""Export coverage-efficiency results from the W&B cache to CSV.

Two files per invocation, both keyed by (method, alpha):
  <out>.csv       one row per (method, alpha, seed) with the raw cov_eff/coverage and
                  cov_eff/efficiency of that run (deduped to the latest run per identity).
  <out>_agg.csv   one row per (method, alpha) with the seed mean and std of both metrics,
                  via plotting.coverage_efficiency._method_stats -- the exact aggregation the
                  trade-off figure plots, so table and figure can never disagree.

Data comes from the W&B cache (see wandb_cache.py), where cov_eff/coverage and
cov_eff/efficiency appear as run-level scalar summary columns after
experiments/coverage_efficiency.py has run. Refresh the cache first:
    python -m plotting.wandb_cache --entity risk-averse --project paper

Run: python src/plotting/coverage_efficiency_csv.py --dataset cifar10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from paths import RESULTS_PATH
from plotting.coverage_efficiency import _method_stats
from plotting.wandb_cache import load_runs

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_COV = "cov_eff/coverage"
_EFF = "cov_eff/efficiency"


def per_seed_table(df: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    """One row per (method, alpha, seed) with that run's coverage and efficiency.

    Rows are deduped to the latest run per (method, alpha, seed, num_train) identity, mirroring
    _method_stats' collapse, so a re-evaluated artifact contributes its newest numbers only.

    Args:
        df: DataFrame from load_runs(), already filtered to one dataset.
        methods: method.name values to include, in the output's method order.

    Returns:
        DataFrame with columns method, alpha (NaN for no-alpha methods), seed, coverage,
        efficiency, sorted by (method, alpha, seed). Empty if nothing matches.
    """
    sub = df[df["method.name"].isin(list(methods))].dropna(subset=[_COV, _EFF])
    sub = sub.drop_duplicates(subset=["run_id"]).copy()
    if "method.train.alpha" not in sub.columns:
        sub["method.train.alpha"] = float("nan")
    # NaN never equals NaN, so a sentinel keeps no-alpha runs in one dedup bucket per seed.
    sub["_alpha_key"] = sub["method.train.alpha"].fillna(-1.0)
    if "created_at" in sub.columns:
        sub = sub.sort_values("created_at", kind="mergesort")
    identity = [c for c in ("method.name", "_alpha_key", "seed", "num_train") if c in sub.columns]
    sub = sub.drop_duplicates(subset=identity, keep="last")
    renames = {"method.name": "method", "method.train.alpha": "alpha", _COV: "coverage", _EFF: "efficiency"}
    out = sub.rename(columns=renames)
    out["method"] = pd.Categorical(out["method"], categories=list(methods), ordered=True)
    return out[["method", "alpha", "seed", "coverage", "efficiency"]].sort_values(["method", "alpha", "seed"])


def aggregate_table(df: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    """One row per (method, alpha) with seed mean/std of coverage and efficiency.

    Args:
        df: DataFrame from load_runs(), already filtered to one dataset.
        methods: method.name values to include, in the output's method order.

    Returns:
        DataFrame with columns method, alpha, n_seeds, coverage, coverage_std, efficiency,
        efficiency_std. Methods without any cov_eff rows are skipped with a warning.
    """
    parts = []
    for method in methods:
        stats = _method_stats(df, method, seeds=None)
        if stats.empty:
            logger.warning("No cov_eff values for %s; skipping.", method)
            continue
        stats.insert(0, "method", method)
        parts.append(stats)
    if not parts:
        return pd.DataFrame()
    agg = pd.concat(parts, ignore_index=True)
    return agg[["method", "alpha", "n_seeds", "coverage", "coverage_std", "efficiency", "efficiency_std"]]


def main() -> None:
    """Write the per-seed and aggregated coverage-efficiency CSVs from the cache."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cifar10", help="dataset column value to filter the cache on")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="method.name values to include (default: every method with cov_eff rows for the dataset)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="per-seed CSV path; the aggregate goes next to it with an _agg suffix "
        "(default: results/coverage_efficiency_<dataset>.csv)",
    )
    args = parser.parse_args()

    df = load_runs(filters={"dataset": args.dataset})
    if df.empty or _COV not in df.columns:
        raise SystemExit(
            f"No cov_eff rows cached for dataset={args.dataset!r}. Run experiments/coverage_efficiency.py, "
            "then refresh the cache via python -m plotting.wandb_cache --entity ... --project ..."
        )
    methods = args.methods or sorted(df.dropna(subset=[_COV])["method.name"].unique().tolist())
    logger.info("Methods: %s", methods)

    out = args.out or RESULTS_PATH / f"coverage_efficiency_{args.dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    per_seed = per_seed_table(df, methods)
    per_seed.to_csv(out, index=False)
    logger.info("Wrote %d per-seed rows -> %s", len(per_seed), out)

    agg = aggregate_table(df, methods)
    agg_out = out.with_name(f"{out.stem}_agg{out.suffix}")
    agg.to_csv(agg_out, index=False)
    logger.info("Wrote %d aggregated rows -> %s", len(agg), agg_out)


if __name__ == "__main__":
    main()
