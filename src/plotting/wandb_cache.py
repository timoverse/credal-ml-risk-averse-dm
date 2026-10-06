r"""On-disk cache for W&B run summaries + configs, used by all plotting code.

Two-function API:
    cache_runs(entity, project, full_refresh=False)
                                 -> hits W&B (server-side filtered to runs updated since the
                                    last sync by default), upserts into cache/wandb/runs.parquet
    load_runs(filters=None)      -> reads the parquet, returns long-form DataFrame; never hits W&B

cache_runs takes no slicing filters; all per-plot filtering happens at load time. The
incremental query is driven by the max heartbeat_at currently in the cache -- which means
deletions on the W&B side are not picked up (the incremental query never returns them as
"modified"). Pass full_refresh=True (or `--full-refresh` on the CLI) to re-fetch every run.

Schema (long form). Every row carries the same run-level metadata:
    run_id, run_name, state, created_at, heartbeat_at
plus the flattened run config (dotted columns, e.g. method.name, method.train.alpha, ...) and
all scalar summary fields (test_acc, test_nll, test_ece, ...).

Each row also carries a `kind` discriminator:
  - kind="risk": one row per risk/<decision_rule>/<aggregation>/<loss>[/beta=<cvar_beta>] summary
    key, with columns decision_rule/aggregation/loss/value (and cvar_beta, set only for cvar_minimax)
    populated.
  - kind="shift": one row per shift/<corruption>/<severity>/<decision_rule>/<aggregation>/<loss>
    [/beta=<cvar_beta>] summary key (experiments/shift_risk_metric.py; severity 0 = clean
    baseline), with columns corruption/severity/decision_rule/aggregation/loss/value populated.
    Set-size rows come from shift/<corruption>/<severity>/set/efficiency keys
    (experiments/shift_set_size.py) and carry the rule-independent sentinel decision_rule="set"
    with loss="efficiency" and no aggregation; the pre-split .../set/mean/efficiency form is
    ignored (re-run shift_set_size.py to repopulate).
  - kind="sp":   one row per sp/<decomposition>/<component>/<decision_rule>/<loss>/<aggregation>
    grouping, with columns decomposition/component/decision_rule/loss/aggregation populated
    and the per-grouping fields auc (float), n_bins (int), bin_losses (list[float]).
  - kind="ood":  one row per ood/<ood_dataset>/<decomposition>/<component>/<metric> summary key
    (experiments/ood_detection.py), with columns ood_dataset/decomposition/component/metric/value
    populated; metric is auroc | aupr | fpr. One run contributes one row per metric per OOD set.
  - kind=None:   placeholder row for runs that have neither risk/* nor sp/* keys, so
    config-only / scalar-summary plotters still see the run.

Non-scalar summary values that don't match either key family are skipped.

Usage from a shell:
    python -m plotting.wandb_cache --entity timo-paul --project testing
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests
import wandb

from paths import CACHE_PATH

logger = logging.getLogger(__name__)

RUNS_CACHE_FILE = CACHE_PATH / "wandb" / "runs.parquet"

# Columns parsed out of `risk/<decision_rule>/<aggregation>/<loss>` keys, with an optional trailing
# `/beta=<cvar_beta>` segment that cvar_minimax appends (its eval-time calibration level).
_RISK_KEY_RE = re.compile(
    r"^risk/(?P<decision_rule>[^/]+)/(?P<aggregation>[^/]+)/(?P<loss>[^/]+)(?:/beta=(?P<cvar_beta>[^/]+))?$"
)
# Columns parsed out of `shift/<corruption>/<severity>/<decision_rule>/<aggregation>/<loss>` keys
# (experiments/shift_risk_metric.py, one per CIFAR-10-C cell x aggregation; severity 0 = the clean
# baseline), with the same optional trailing `/beta=<cvar_beta>` segment as the risk family.
_SHIFT_KEY_RE = re.compile(
    r"^shift/(?P<corruption>[^/]+)/(?P<severity>\d+)/"
    r"(?P<decision_rule>[^/]+)/(?P<aggregation>[^/]+)/(?P<loss>[^/]+)(?:/beta=(?P<cvar_beta>[^/]+))?$"
)

# Set-size keys `shift/<corruption>/<severity>/set/efficiency` (experiments/shift_set_size.py).
# Five segments, so no overlap with the six-segment _SHIFT_KEY_RE above; rows get the sentinel
# decision_rule="set" and loss="efficiency" with no aggregation.
_SHIFT_SET_KEY_RE = re.compile(r"^shift/(?P<corruption>[^/]+)/(?P<severity>\d+)/set/(?P<metric>efficiency|accuracy)$")

# Columns parsed out of `sp/<decomposition>/<component>/<decision_rule>/<loss>/<aggregation>/<field>`
# keys, where field is one of auc | bin_losses | n_bins. We accumulate the three fields per
# (decomposition, component, decision_rule, loss, aggregation) grouping into a single SP row.
_SP_KEY_RE = re.compile(
    r"^sp/(?P<decomposition>[^/]+)/(?P<component>[^/]+)/"
    r"(?P<decision_rule>[^/]+)/(?P<loss>[^/]+)/(?P<aggregation>[^/]+)/"
    r"(?P<field>auc|bin_losses|n_bins)$"
)

# Columns parsed out of `ood/<ood_dataset>/<decomposition>/<component>/<metric>` keys
# (experiments/ood_detection.py), one row per metric of one ID-vs-OOD evaluation. metric is left
# open rather than enumerated so new probly metrics land in the cache without a change here.
_OOD_KEY_RE = re.compile(
    r"^ood/(?P<ood_dataset>[^/]+)/(?P<decomposition>[^/]+)/(?P<component>[^/]+)/(?P<metric>[^/]+)$"
)


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict into {dotted_key: scalar} pairs. Skips non-scalars."""
    out: dict[str, Any] = {}
    for key, value in config.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, full))
        elif value is None or isinstance(value, int | float | str | bool):
            out[full] = value
    return out


def _rows_from_run(run: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Build kind-labeled rows from a run: one per risk/* key, one per sp/* grouping, or a None placeholder.

    Non-risk, non-sp scalar summary fields (test_acc, test_nll, test_ece, ...) are denormalized
    onto every row as run-level metadata, alongside the flattened config. Runs with neither
    risk/* nor sp/* keys still appear in the cache with kind=None so config-only / scalar-summary
    plotters see them.
    """
    # wandb's GraphQL Run type exposes heartbeatAt, not updatedAt; the heartbeat is
    # bumped server-side on summary.update() so it's the right "last touched" timestamp.
    heartbeat_at = getattr(run, "heartbeat_at", None)
    summary = dict(run.summary)
    other_scalars = {
        k: v
        for k, v in summary.items()
        if not _RISK_KEY_RE.match(k)
        and not _SHIFT_KEY_RE.match(k)
        and not _SHIFT_SET_KEY_RE.match(k)
        and not _SP_KEY_RE.match(k)
        and not _OOD_KEY_RE.match(k)
        and isinstance(v, int | float | str | bool)
    }
    base: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "heartbeat_at": str(heartbeat_at) if heartbeat_at else None,
        **_flatten(dict(run.config)),
        **other_scalars,
    }
    rows: list[dict[str, Any]] = []

    # risk/* rows: one per matched summary key.
    for key, value in summary.items():
        m = _RISK_KEY_RE.match(key)
        if m is None or not isinstance(value, int | float):
            continue
        fields = m.groupdict()
        fields["cvar_beta"] = float(fields["cvar_beta"]) if fields["cvar_beta"] is not None else None
        rows.append({**base, "kind": "risk", **fields, "value": float(value)})

    # shift/* rows: one per matched summary key (one CIFAR-10-C cell x aggregation).
    for key, value in summary.items():
        m = _SHIFT_KEY_RE.match(key)
        if m is None or not isinstance(value, int | float):
            continue
        fields = m.groupdict()
        if fields["decision_rule"] == "set":
            # Legacy pre-split set-size keys (shift/<c>/<s>/set/mean/efficiency), superseded by
            # the aggregation-free _SHIFT_SET_KEY_RE form; skip so vintages cannot mix.
            continue
        fields["severity"] = int(fields["severity"])
        fields["cvar_beta"] = float(fields["cvar_beta"]) if fields["cvar_beta"] is not None else None
        rows.append({**base, "kind": "shift", **fields, "value": float(value)})

    # shift set-size rows: one per cell and metric (efficiency, or the regular point-prediction
    # accuracy), decision-rule- and loss-independent (no aggregation); the metric name lands in
    # the loss column.
    for key, value in summary.items():
        m = _SHIFT_SET_KEY_RE.match(key)
        if m is None or not isinstance(value, int | float):
            continue
        rows.append(
            {
                **base,
                "kind": "shift",
                "corruption": m["corruption"],
                "severity": int(m["severity"]),
                "decision_rule": "set",
                "loss": m["metric"],
                "value": float(value),
            }
        )

    # sp/* rows: accumulate the three fields (auc / n_bins / bin_losses) per grouping,
    # then emit one row per grouping. Skip groupings that didn't accumulate an auc -- they
    # are partial writes from an interrupted run.
    sp_groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for key, value in summary.items():
        m = _SP_KEY_RE.match(key)
        if m is None:
            continue
        groupdict = m.groupdict()
        group_key = (
            groupdict["decomposition"],
            groupdict["component"],
            groupdict["decision_rule"],
            groupdict["loss"],
            groupdict["aggregation"],
        )
        sp_groups.setdefault(group_key, {k: groupdict[k] for k in groupdict if k != "field"})
        field = groupdict["field"]
        if field == "auc" and isinstance(value, int | float):
            sp_groups[group_key]["auc"] = float(value)
        elif field == "n_bins" and isinstance(value, int | float):
            sp_groups[group_key]["n_bins"] = int(value)
        elif field == "bin_losses" and isinstance(value, list):
            sp_groups[group_key]["bin_losses"] = [float(x) for x in value]
    for grouping in sp_groups.values():
        if "auc" not in grouping:
            continue
        rows.append({**base, "kind": "sp", **grouping})

    # ood/* rows: one per metric of one (ood_dataset, decomposition, component) evaluation.
    for key, value in summary.items():
        m = _OOD_KEY_RE.match(key)
        if m is None or not isinstance(value, int | float):
            continue
        rows.append({**base, "kind": "ood", **m.groupdict(), "value": float(value)})

    if not rows:
        rows.append({**base, "kind": None})
    return rows


def _stringify_mixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Store the non-null values of object columns that mix strings with other types as strings.

    Parquet needs one type per column, and config values are not always typed consistently across
    runs: method.params.bandwidth_anchor is the int 1 in most runs and the string "neighborhood" in
    others, which made to_parquet raise ArrowTypeError and the whole sync fail. Purely numeric or
    purely string columns are left untouched.

    Args:
        df: The merged cache frame about to be written.

    Returns:
        The same frame, with every mixed-type object column converted to strings (nulls kept).
    """
    for col in df.select_dtypes(include="object").columns:
        values = df[col].dropna()
        kinds = {isinstance(v, str) for v in values}
        if len(kinds) > 1:
            df[col] = df[col].map(lambda v: v if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return df


def cache_runs(
    entity: str,
    project: str,
    timeout: int = 60,
    full_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch new or updated runs from a W&B project and upsert them into the local cache.

    Incremental by default: passes a heartbeatAt cutoff (taken from the max value already in
    the cache) as a server-side filter, so only runs modified since the last sync come back.
    Pass full_refresh=True to ignore the cutoff and re-fetch every run -- needed to pick up
    deletions on the W&B side, or after schema changes.

    Args:
        entity: W&B entity.
        project: W&B project.
        timeout: W&B API timeout in seconds.
        full_refresh: If True, fetch every run regardless of the cache cutoff.

    Returns:
        The fetched-this-call rows as a DataFrame (not the full cache).
    """
    existing_df = pd.read_parquet(RUNS_CACHE_FILE) if RUNS_CACHE_FILE.exists() else pd.DataFrame()

    api = wandb.Api(timeout=timeout)
    filters: dict[str, Any] | None = None
    if not full_refresh and not existing_df.empty and "heartbeat_at" in existing_df.columns:
        non_null = existing_df["heartbeat_at"].dropna()
        # Guard against an all-null column: .max() on an empty Series returns NaN, which
        # serialises as `NaN` in the filter and gets rejected by wandb's GraphQL endpoint
        # with a 400. Pass the wandb-formatted timestamp straight through; reformatting via
        # pd.to_datetime+.isoformat() drops the Z suffix and the heartbeatAt index then
        # silently matches nothing.
        cached_max = non_null.max() if not non_null.empty else None
        if isinstance(cached_max, str) and cached_max:
            filters = {"heartbeatAt": {"$gt": cached_max}}
            logger.info("Incremental: fetching runs updated after %s", cached_max)
    if filters is None:
        logger.info("Full refresh: fetching all runs from %s/%s", entity, project)

    runs = api.runs(f"{entity}/{project}", filters=filters)
    new_rows: list[dict[str, Any]] = []
    n_runs = 0
    n_with_risk = 0
    n_failed = 0
    for run in runs:
        n_runs += 1
        try:
            rows = _rows_from_run(run)
        except (requests.exceptions.HTTPError, wandb.errors.CommError) as e:
            # A single run whose summary fetch 500s (transient outage, or a summary the
            # backend chokes on) must not kill the whole sync; skip it loudly instead.
            n_failed += 1
            logger.warning("Failed to fetch run %s (%s): %s -- skipping.", run.id, getattr(run, "name", "?"), e)
            continue
        if any(r.get("value") is not None for r in rows):
            n_with_risk += 1
        new_rows.extend(rows)
    logger.info(
        "Fetched %d run(s) (%d with risk/* or shift/* summaries) from %s/%s",
        n_runs,
        n_with_risk,
        entity,
        project,
    )
    if n_failed:
        logger.warning(
            "%d run(s) failed to fetch and are missing from the cache. If other runs in this sync "
            "have later heartbeats, the incremental watermark has advanced past the failures: re-run "
            "the sync once the W&B hiccup passes, or use --full-refresh to backfill.",
            n_failed,
        )

    if not new_rows:
        if filters is not None:
            logger.warning(
                "Incremental query returned 0 runs. If you just ran new experiments, the "
                "heartbeatAt filter format may not be matching wandb's index. Re-run with "
                "--full-refresh to bypass the filter and verify.",
            )
        else:
            logger.info("No runs returned; cache unchanged.")
        return pd.DataFrame()
    n_with_sp = sum(1 for r in new_rows if r.get("kind") == "sp")
    if n_with_sp:
        logger.info("...of which %d sp/* row(s) across all fetched runs.", n_with_sp)

    new_df = pd.DataFrame(new_rows)
    if not existing_df.empty:
        kept = existing_df[~existing_df["run_id"].isin(new_df["run_id"].unique())]
        merged = pd.concat([kept, new_df], ignore_index=True)
    else:
        merged = new_df
    RUNS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _stringify_mixed_columns(merged).to_parquet(RUNS_CACHE_FILE, index=False)
    logger.info("Cache now at %s with %d rows from %d runs.", RUNS_CACHE_FILE, len(merged), merged["run_id"].nunique())
    return new_df


def load_runs(filters: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read the local cache and return rows matching simple column-equality filters.

    Filters use DataFrame column names (dotted for nested config, e.g. method.name,
    method.train.alpha, plus decision_rule, aggregation, loss, ...). Values may be a scalar
    (exact match) or a list (membership). Never hits W&B.

    Args:
        filters: Dict of {column: scalar-or-list} pairs. None returns all rows.

    Returns:
        DataFrame filtered to matching rows. Empty if the cache is empty or nothing matches.

    Raises:
        FileNotFoundError: cache file does not exist (run cache_runs first).
    """
    if not RUNS_CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Cache file {RUNS_CACHE_FILE} does not exist. Run cache_runs(...) first or "
            f"`python -m plotting.wandb_cache --entity ... --project ...`."
        )
    df = pd.read_parquet(RUNS_CACHE_FILE)
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, value in filters.items():
        if col not in df.columns:
            raise KeyError(f"Filter column {col!r} not in cache columns {sorted(df.columns)}.")
        mask &= df[col].isin(value) if isinstance(value, list | tuple | set) else (df[col] == value)
    return df[mask].reset_index(drop=True)


def main(args: argparse.Namespace) -> None:
    """Programmatic entry point: configure logging, then fetch the project into the cache."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cache_runs(entity=args.entity, project=args.project, full_refresh=args.full_refresh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache W&B runs to a local parquet for plotting.")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Re-fetch every run instead of the incremental updatedAt-filtered set.",
    )
    args = parser.parse_args()
    main(args)
