"""Rebuild an OOD table from the logged score arrays, scoring with sklearn instead of probly.

probly's evaluate_ood sorts scores with np.flip(np.argsort(kind="mergesort")), which reverses
the stable order inside blocks of tied scores. Because evaluate_ood concatenates [id, ood],
tied OOD samples always rank above tied ID samples, so its AUROC is P(ood>id) + 1.0*P(tie)
rather than the correct Mann-Whitney value P(ood>id) + 0.5*P(tie). With no ties the two agree
exactly; with ties probly is inflated, and a fully saturated cell reports exactly 1.000000.

The inflation is bounded by 0.5 * P(id saturated) * P(ood saturated), since only ID-OOD ties
matter. It is therefore negligible for methods whose scores are continuous and material for
methods that clamp: credal_rl_multinomial pins every point with no local evidence at the
vacuous entropy log(K), so its far-OOD scores collapse onto a single value.

This reads the per-instance id_scores / ood_scores artifacts that ood_detection.py logged,
recomputes AUROC and AUPR with sklearn, and prints both a readable table and LaTeX rows in the
format plotting/ood_table.py emits. It also reports the fraction of CLEAN in-distribution
points sitting at log(K), which is what makes a near-1.0 score interpretable: a high score with
a low clean-vacuous fraction is a working detector, the same score with a high one is a model
that has given up on its own training distribution.

Needs W&B credentials. Downloaded arrays are cached under cache/ood_scores/<dataset>/, so
repeat runs are offline.

Edit the constants in the __main__ block (the ood_table.py convention) and run from the
project root or src/:
    python src/experiments/recompute_ood_sklearn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import average_precision_score, roc_auc_score

from artifacts import resolve_artifact_name
from paths import CACHE_PATH
from plotting.ood_table import render_paper_rows
from plotting.shift_pareto import _METHOD_STYLES

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)

# Per-method fit parameters that resolve_artifact_name encodes into the artifact name. alpha is
# None for methods with no alpha knob; delta_g is set only for credal_dro, whose name carries a
# _deltag<d> segment. Keep in sync with configs/method/*.yaml -- a wrong value here is
# indistinguishable from a missing artifact.
METHOD_SPECS: dict[str, dict[str, float | None]] = {
    "credal_rl_multinomial": {"alpha": 0.95, "delta_g": None},
    "credal_relative_likelihood": {"alpha": 0.95, "delta_g": None},
    "efficient_credal_prediction": {"alpha": 0.95, "delta_g": None},
    "credal_wrapper": {"alpha": None, "delta_g": None},
    "credal_dro": {"alpha": None, "delta_g": 0.5},
    "credal_ensembling": {"alpha": None, "delta_g": None},
    "credal_bnn": {"alpha": None, "delta_g": None},
}


def _artifact_stem(method: str, dataset: str, base_model: str, seed: int, alpha: float | None) -> str:
    """Build the artifact stem for one fitted model via the repo's own naming function.

    Args:
        method: Method name, a key of METHOD_SPECS.
        dataset: In-distribution dataset the model was trained on.
        base_model: Backbone name, e.g. resnet18.
        seed: Training seed.
        alpha: Alpha override for alpha-carrying methods; None uses METHOD_SPECS.

    Returns:
        The artifact stem, e.g. credal_dro_deltag0.5_resnet18_bloodmnist_seed1.
    """
    spec = METHOD_SPECS[method]
    train: dict[str, float] = {}
    resolved_alpha = spec["alpha"] if alpha is None else alpha
    if spec["alpha"] is not None and resolved_alpha is not None:
        train["alpha"] = resolved_alpha
    if spec["delta_g"] is not None:
        train["delta_g"] = spec["delta_g"]
    cfg: dict[str, Any] = {
        "method": {"name": method, "train": train},
        "base_model": base_model,
        "dataset": dataset,
        "seed": seed,
        "num_train": None,
    }
    return resolve_artifact_name(cfg)


def _fetch(
    api: wandb.Api, entity: str, project: str, name: str, filename: str, cache: Path
) -> npt.NDArray[np.floating] | None:
    """Download one .npy artifact, cached on disk; None when it does not exist.

    Args:
        api: Live wandb Api handle.
        entity: W&B entity.
        project: W&B project.
        name: Artifact name.
        filename: File inside the artifact.
        cache: Directory to cache downloads under.

    Returns:
        The loaded array, or None if the artifact is absent or undownloadable.
    """
    target = cache / name / filename
    if not target.exists():
        try:
            api.artifact(f"{entity}/{project}/{name}:latest").download(root=str(cache / name))
        except Exception as e:  # noqa: BLE001  a missing artifact is a normal, reportable outcome
            logger.warning("missing artifact %s (%s)", name, type(e).__name__)
            return None
    return np.load(target)


def _is_finite(scores: npt.NDArray[np.floating], label: str) -> bool:
    """Report whether every score is finite, logging where the bad entries are if not.

    Artifacts logged before ood_detection gained its finite guard can contain NaN: probly's
    convex-credal-set entropy solver returns NaN rather than raising when its L-BFGS diverges,
    and credal_bnn averages 20 stochastic forward passes per member, so one overflowing draw
    poisons that vertex. Skipping such a cell leaves it as "--" and keeps the rest of the table
    usable, rather than letting sklearn abort the whole run on the first bad array.

    Args:
        scores: Per-instance score array.
        label: Which array this is, for the warning.

    Returns:
        True when every entry is finite.
    """
    n_bad = int((~np.isfinite(scores)).sum())
    if n_bad:
        logger.warning("SKIPPING %s: %d of %d entries are not finite (re-run that cell)", label, n_bad, scores.size)
        return False
    return True


def _cell(values: list[float], decimals: int) -> str:
    r"""Format per-seed values as the $mean \scriptstyle \pm std$ string ood_table.py emits.

    Args:
        values: One value per seed.
        decimals: Digits after the decimal point.

    Returns:
        The LaTeX cell, or a dash when no seed produced a value.
    """
    if not values:
        return "--"
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return f"${arr.mean():.{decimals}f} \\scriptstyle \\pm {std:.{decimals}f}$"


def recompute(
    dataset: str,
    ood_datasets: list[str],
    methods: list[str],
    seeds: list[int],
    num_classes: int,
    *,
    base_model: str = "resnet18",
    alpha: float | None = None,
    entity: str = "risk-averse",
    project: str = "paper",
    decomposition: str = "entropy",
    component: str = "epistemic",
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]], dict[str, list[float]]]:
    """Recompute AUROC and AUPR with sklearn for every (method, OOD set, seed).

    Args:
        dataset: In-distribution dataset name.
        ood_datasets: OOD dataset names, in table-column order.
        methods: Method names, in table-row order; each must be a key of METHOD_SPECS.
        seeds: Training seeds to aggregate over.
        num_classes: Class count of the ID dataset, for the log(K) vacuity check.
        base_model: Backbone name.
        alpha: Override the alpha of alpha-carrying methods; None uses METHOD_SPECS.
        entity: W&B entity.
        project: W&B project.
        decomposition: Decomposition label the scores were computed with.
        component: Uncertainty component the scores were computed with.

    Returns:
        (auroc, aupr, id_vacuous), where auroc and aupr map method -> ood dataset -> per-seed
        values, and id_vacuous maps method -> per-seed clean vacuous fractions.
    """
    api = wandb.Api(timeout=120)
    cache = CACHE_PATH / "ood_scores" / dataset
    cache.mkdir(parents=True, exist_ok=True)
    vacuous_value = math.log(num_classes)

    auroc: dict[str, dict[str, list[float]]] = {}
    aupr: dict[str, dict[str, list[float]]] = {}
    id_vacuous: dict[str, list[float]] = {}

    for method in methods:
        auroc[method] = {o: [] for o in ood_datasets}
        aupr[method] = {o: [] for o in ood_datasets}
        id_vacuous[method] = []
        for seed in seeds:
            stem = _artifact_stem(method, dataset, base_model, seed, alpha)
            id_scores = _fetch(
                api, entity, project, f"id_scores_{stem}_{decomposition}_{component}", "id_scores.npy", cache
            )
            if id_scores is None:
                continue
            if not _is_finite(id_scores, f"id_scores {method} seed{seed}"):
                continue  # a poisoned ID array invalidates every OOD pair for this seed
            id_vacuous[method].append(float(np.mean(np.isclose(id_scores, vacuous_value, atol=1e-5))))
            for ood in ood_datasets:
                ood_scores = _fetch(
                    api,
                    entity,
                    project,
                    f"ood_scores_{stem}_{ood}_{decomposition}_{component}",
                    "ood_scores.npy",
                    cache,
                )
                if ood_scores is None:
                    continue
                if not _is_finite(ood_scores, f"ood_scores {method} seed{seed} vs {ood}"):
                    continue
                labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
                preds = np.concatenate([id_scores, ood_scores])
                auroc[method][ood].append(float(roc_auc_score(labels, preds)))
                aupr[method][ood].append(float(average_precision_score(labels, preds)))
        logger.info("%s: %d/%d seeds found", method, len(id_vacuous[method]), len(seeds))
    return auroc, aupr, id_vacuous


def _row_label(method: str, alpha: float | None) -> str:
    r"""Build the row label ood_table.py uses: styled method name plus an alpha suffix.

    Args:
        method: Method name.
        alpha: Alpha override; None uses METHOD_SPECS.

    Returns:
        The label, e.g. "RL-Multinomial ($\alpha=0.95$)" or "CredalWrapper".
    """
    resolved = METHOD_SPECS[method]["alpha"] if alpha is None else alpha
    suffix = "" if METHOD_SPECS[method]["alpha"] is None or resolved is None else f" ($\\alpha={resolved:g}$)"
    return _METHOD_STYLES.get(method, ("", None, method))[2] + suffix


def build_table(
    values: dict[str, dict[str, list[float]]],
    ood_datasets: list[str],
    methods: list[str],
    alpha: float | None,
    decimals: int,
) -> pd.DataFrame:
    r"""Assemble the same frame ood_table() returns: $mean \scriptstyle \pm std$ strings.

    Same index (styled method label with alpha suffix), same column order, same "--" for empty
    cells, so the result is interchangeable with ood_table() output and can be handed straight
    to render_paper_rows.

    Args:
        values: method -> ood dataset -> per-seed values, from recompute().
        ood_datasets: Column order.
        methods: Row order.
        alpha: Alpha override used for the row labels; None uses METHOD_SPECS.
        decimals: Digits after the decimal point.

    Returns:
        DataFrame of formatted strings, indexed by method label, one column per OOD dataset.
    """
    frame = pd.DataFrame(
        [[_cell(values[m][o], decimals) for o in ood_datasets] for m in methods],
        index=[_row_label(m, alpha) for m in methods],
        columns=ood_datasets,
    )
    frame.index.name = None
    frame.columns.name = None
    return frame


def _print_tables(
    auroc: dict[str, dict[str, list[float]]],
    aupr: dict[str, dict[str, list[float]]],
    id_vacuous: dict[str, list[float]],
    ood_datasets: list[str],
    methods: list[str],
    num_classes: int,
    alpha: float | None,
    decimals: int,
) -> None:
    """Print each metric's table and paper rows, then the clean vacuity column.

    Args:
        auroc: Per-seed AUROC values from recompute().
        aupr: Per-seed AUPR values from recompute().
        id_vacuous: Per-seed clean vacuous fractions from recompute().
        ood_datasets: Column order.
        methods: Row order.
        num_classes: Class count, for the log(K) header note.
        alpha: Alpha override used for the row labels.
        decimals: Digits after the decimal point.
    """
    for title, values in (("auroc", auroc), ("aupr", aupr)):
        table = build_table(values, ood_datasets, methods, alpha, decimals)
        print(f"\n{table.to_string()}\n")
        print(f"---- paper rows, {title} (sklearn) (paste below the Method header row) ----")
        print(render_paper_rows(table))

    # Not part of the paper table, but a near-1.0 AUROC only means what it appears to mean when
    # the method is not also calling clean in-distribution points vacuous.
    print(f"\n---- clean ID fraction at the vacuous value log({num_classes}) ----")
    labels = {m: _row_label(m, alpha) for m in methods}
    pad = max(len(label) for label in labels.values()) + 2
    for method in methods:
        values_ = id_vacuous[method]
        shown = f"{np.mean(values_):.4f}" if values_ else "--"
        print(f"{labels[method]:<{pad}s}{shown}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Edit these and re-run.
    DATASET = "bloodmnist"
    NUM_CLASSES = 8  # bloodmnist 8, pathmnist 9, cifar10 10
    BASE_MODEL = "resnet18"
    OOD_DATASETS = ["pathmnist", "tissuemnist", "dermamnist", "retinamnist", "octmnist", "breastmnist"]
    # Row order. Every name must be a key of METHOD_SPECS, which is what encodes each method's
    # alpha / delta_g into its artifact name.
    METHODS = [
        # "credal_ensembling",
        "credal_bnn",
        # "credal_wrapper",
        # "credal_dro",
        # "efficient_credal_prediction",
        # "credal_relative_likelihood",
        # "credal_rl_multinomial",
    ]
    SEEDS = [1, 2, 3]
    # Overrides the alpha of the alpha-carrying methods; None keeps each method's METHOD_SPECS value.
    ALPHA: float | None = 0.95
    # Must match what ood_detection.py was run with, or every artifact lookup misses.
    DECOMPOSITION = "entropy"
    COMPONENT = "epistemic"
    DECIMALS = 3
    ENTITY = "risk-averse"
    PROJECT = "paper"

    auroc_values, aupr_values, id_vacuous_values = recompute(
        DATASET,
        OOD_DATASETS,
        METHODS,
        SEEDS,
        NUM_CLASSES,
        base_model=BASE_MODEL,
        alpha=ALPHA,
        entity=ENTITY,
        project=PROJECT,
        decomposition=DECOMPOSITION,
        component=COMPONENT,
    )
    _print_tables(auroc_values, aupr_values, id_vacuous_values, OOD_DATASETS, METHODS, NUM_CLASSES, ALPHA, DECIMALS)
