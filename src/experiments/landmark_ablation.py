"""Landmark-compression ablation for credal_rl_multinomial: OOD AUROC and test seconds versus reference size.

The trained artifact stores the full whitened training reference (45000 points with labels on
cifar10). This script compresses that stored reference at evaluation time into M k-means
landmarks, each carrying its cluster's class histogram -- the exact construction the
reference_landmarks train knob would apply at fit time -- and sweeps M. Encoder, whitener,
bandwidth and alpha stay fixed from the artifact; the landmark count is the only moving part,
and the untouched full reference is scored first as the exact-method baseline row.

Per (M, k-means seed) cell the csv records the compression seconds (kmeans_s), the seconds of
one full credal test pass over the test split (test_s, measured by timing.timed_test, the same
public routine inference_timing.py reuses, so the column is directly comparable to the test_s
columns of results/timing/*.csv when run on the same machine), and the OOD detection AUROC of
the negated evidence mass against the six OpenOOD cifar10 OOD sets, computed with scikit-learn.
Encoder features of the (id, ood) test pairs are collected once and cached; each sweep cell only
recomputes the evidence mass against its compressed reference. Nothing is trained, saved or
uploaded; wandb is touched only to download the artifact (artifact_source=local reads the
checkpoints directory instead).

Run from the project root (wandb credentials required unless artifact_source=local):
    python src/experiments/landmark_ablation.py
    python src/experiments/landmark_ablation.py --seed 1 --landmarks 4096 256 --kmeans-seeds 1 device=cuda:0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The experiments dir itself, so timed_test is imported from the timing script rather than
# copied; the test_s columns of the two scripts are only comparable while they share it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import csv
import gc
import logging
import statistics
import time
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from timing import timed_test

import data
import utils
from artifacts import describe_run_source, load_model_for_evaluation, resolve_artifact_name
from methods.credal_rl_multinomial import CredalRLMultinomialPredictor
from paths import RESULTS_PATH
from training.train_funcs import collect_encoder_features

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

CONFIGS_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs")

# The six OOD sets of OpenOOD's CIFAR-10 benchmark (see data.get_ood_data). The ablation is
# cifar10-only for now, so any other recipe fails loudly before touching data.
OOD_DATASETS: dict[str, tuple[str, ...]] = {
    "cifar10": ("cifar100", "tin", "mnist", "svhn", "textures", "places365"),
}


def _compose_train_cfg(recipe: str, seed: int, overrides: list[str]) -> DictConfig:
    """Compose the train config identifying the credal_rl_multinomial artifact to load.

    The train config (not an eval one) is composed on purpose: it is what
    resolve_artifact_name reads to name the artifact, and it carries the loader knobs and amp
    flag timed_test needs -- the identical values the timing experiment measured with.

    Args:
        recipe: Recipe config name (configs/recipe/<recipe>.yaml).
        seed: Seed of the trained artifact to load.
        overrides: Extra hydra overrides forwarded from the command line.

    Returns:
        The composed train config.
    """
    full = ["method=credal_rl_multinomial", f"recipe={recipe}", f"seed={seed}", *overrides]
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        return compose(config_name="train", overrides=full)


def _compress_reference(
    reference: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    num_landmarks: int,
    kmeans_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """K-means the stored whitened reference into landmarks carrying class histograms.

    Mirrors the reference_landmarks branch of train_credal_rl_multinomial verbatim (same
    MiniBatchKMeans settings), so a sweep cell stores what a fit with that knob would have;
    only the random_state is the cell's own. The histograms sum to the reference size, which
    is what keeps the evidence mass on the scale of the full method.

    Args:
        reference: full projected reference, shape (N, D), any float dtype.
        targets: integer reference labels, shape (N,).
        num_classes: number of classes K.
        num_landmarks: number of k-means clusters M.
        kmeans_seed: random_state of the k-means fit.

    Returns:
        Landmark positions of shape (M, D) in float32, class masses of shape (M, K) in
        float32 (both on cpu), and the elapsed compression seconds.
    """
    start = time.perf_counter()
    kmeans = MiniBatchKMeans(n_clusters=num_landmarks, batch_size=4096, n_init=3, max_iter=60, random_state=kmeans_seed)
    assignments = torch.from_numpy(kmeans.fit_predict(reference.float().cpu().numpy())).long()
    landmarks = torch.from_numpy(kmeans.cluster_centers_).float()
    masses = torch.zeros(num_landmarks, num_classes, dtype=torch.float32)
    masses.index_put_((assignments, targets.long().cpu()), torch.ones(len(assignments)), accumulate=True)
    return landmarks, masses, time.perf_counter() - start


@torch.no_grad()
def _cache_pair_features(
    model: CredalRLMultinomialPredictor,
    cfg: DictConfig,
    ood_names: tuple[str, ...],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Encoder features of the matched (id, ood) test pairs, computed once for the whole sweep.

    Loaders come from data.get_ood_data with the same knobs ood_detection.py uses, so each
    pair's deterministic truncation to the smaller test set matches the OOD experiment's
    protocol.

    Args:
        model: loaded predictor whose encoder embeds the images.
        cfg: composed train config (dataset, batch size, seed, amp).
        ood_names: OOD dataset names to pair the id test set with.
        device: inference device.

    Returns:
        Mapping from ood name to (id features, ood features), float32 on device.
    """
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in ood_names:
        id_loader, ood_loader = data.get_ood_data(
            cfg.dataset, name, cfg.seed, batch_size=cfg.batch_size, num_workers=2, pin_memory=True
        )
        id_features, _ = collect_encoder_features(model.encoder, id_loader, device, bool(cfg.amp))
        ood_features, _ = collect_encoder_features(model.encoder, ood_loader, device, bool(cfg.amp))
        pairs[name] = (id_features, ood_features)
        logger.info(
            "Cached features %s vs %s: id %s, ood %s",
            cfg.dataset,
            name,
            tuple(id_features.shape),
            tuple(ood_features.shape),
        )
    return pairs


@torch.no_grad()
def _pair_aurocs(
    model: CredalRLMultinomialPredictor,
    pair_features: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """AUROC of the negated evidence mass for every cached (id, ood) pair.

    Low mass means high novelty, so the score is the negative mass with ood labelled 1.
    Computed with sklearn.metrics.roc_auc_score, whose tie handling (0.5 credit) is the
    standard one.

    Args:
        model: predictor whose current reference buffers define the evidence.
        pair_features: mapping from ood name to (id features, ood features).

    Returns:
        Mapping from ood name to AUROC.
    """
    aurocs: dict[str, float] = {}
    for name, (id_features, ood_features) in pair_features.items():
        _, id_mass = model.evidence(id_features)
        _, ood_mass = model.evidence(ood_features)
        scores = np.concatenate([-id_mass.cpu().numpy(), -ood_mass.cpu().numpy()])
        labels = np.concatenate([np.zeros(id_mass.shape[0]), np.ones(ood_mass.shape[0])])
        aurocs[name] = float(roc_auc_score(labels, scores))
    return aurocs


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str | int | float]]) -> None:
    """Rewrite the csv with one row per finished sweep cell.

    Args:
        path: Output csv path.
        fieldnames: Column order.
        rows: Finished rows, baseline first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    """Parse script knobs plus hydra overrides forwarded to the composed train config."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", default="cifar10_resnet18", help="Recipe config name.")
    parser.add_argument("--seed", type=int, default=1, help="Seed of the trained artifact to load.")
    parser.add_argument(
        "--landmarks",
        nargs="+",
        type=int,
        default=[20000, 10000, 5000, 2000, 1000],
        help="Landmark counts M to sweep, besides the always-included full-reference baseline.",
    )
    parser.add_argument("--kmeans-seeds", nargs="+", type=int, default=[1, 2, 3], help="K-means seeds per M.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output csv; default landmark_ablation_<recipe>_seed<seed>.csv. Relative paths resolve inside results/.",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. device=cuda:0 artifact_source=local.")
    # Overrides may be interspersed with the --flags (e.g. device=... before --out); plain
    # parse_args only accepts one contiguous positional chunk.
    return parser.parse_intermixed_args()


def main() -> None:
    """Load the artifact, cache pair features, run the landmark sweep, write the csv.

    Raises:
        ValueError: the recipe's dataset has no OOD benchmark configured here, or a requested
            landmark count does not lie strictly between 1 and the stored reference size.
        RuntimeError: the artifact was trained with a different method, is not a
            CredalRLMultinomialPredictor, or stores a landmark-compressed reference (no
            per-point labels) and therefore cannot seed the sweep.
    """
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = _compose_train_cfg(args.recipe, args.seed, args.overrides)
    if cfg.dataset not in OOD_DATASETS:
        raise ValueError(f"No OOD benchmark configured for dataset {cfg.dataset!r}; cifar10 only for now.")
    ood_names = OOD_DATASETS[cfg.dataset]

    utils.set_seed(args.seed)
    device = utils.get_device(cfg.get("device"))
    logger.info("Running on device: %s", device)

    artifact_name = resolve_artifact_name(cfg)
    model, train_cfg, run_id = load_model_for_evaluation(cfg, device)
    if train_cfg["method"]["name"] != cfg.method.name:
        msg = (
            f"Artifact {artifact_name!r} was trained with method {train_cfg['method']['name']!r}, "
            f"not {cfg.method.name!r}; check method/recipe/seed."
        )
        raise RuntimeError(msg)
    if not isinstance(model, CredalRLMultinomialPredictor):
        raise RuntimeError(f"Artifact {artifact_name!r} rebuilt as {type(model).__name__}, cannot sweep it.")
    if model.reference_targets is None:
        msg = (
            f"Artifact {artifact_name!r} stores a landmark-compressed reference (no per-point "
            "labels); the sweep needs a full-reference fit."
        )
        raise RuntimeError(msg)
    logger.info("Loaded %s from %s", artifact_name, describe_run_source(run_id))

    full_reference = cast("torch.Tensor", model.reference).detach().clone()
    full_targets = cast("torch.Tensor", model.reference_targets).detach().clone()
    n_reference = int(full_reference.shape[0])
    bad = [m for m in args.landmarks if not 1 < m < n_reference]
    if bad:
        raise ValueError(f"Landmark counts must lie strictly between 1 and {n_reference}, got {bad}.")

    pair_features = _cache_pair_features(model, cfg, ood_names, device)

    if args.out is None:
        out = RESULTS_PATH / f"landmark_ablation_{args.recipe}_seed{args.seed}.csv"
    else:
        # Relative --out resolves inside results/ so jobs land there no matter the cwd.
        out = args.out if args.out.is_absolute() else RESULTS_PATH / args.out
    if out.exists():
        logger.warning("Overwriting existing %s", out)

    fieldnames = ["m_landmarks", "kmeans_seed", "kmeans_s", "test_s", *(f"auroc_{n}" for n in ood_names)]
    rows: list[dict[str, str | int | float]] = []

    # Baseline first: the untouched full reference is the exact original method (a one-hot mass
    # matrix reproduces it, so this row anchors the sweep at zero compression) and its aurocs
    # must line up with the known OOD-table numbers.
    aurocs = _pair_aurocs(model, pair_features)
    test_s = timed_test(model, cfg, device)
    rows.append(
        {"m_landmarks": n_reference, "kmeans_seed": "", "kmeans_s": "", "test_s": test_s}
        | {f"auroc_{n}": v for n, v in aurocs.items()}
    )
    _write_csv(out, fieldnames, rows)
    logger.info(
        "baseline (full %d): test %.3f s, mean auroc %.4f", n_reference, test_s, statistics.fmean(aurocs.values())
    )

    for num_landmarks in args.landmarks:
        for kmeans_seed in args.kmeans_seeds:
            landmarks, masses, kmeans_s = _compress_reference(
                full_reference, full_targets, model.num_classes, num_landmarks, kmeans_seed
            )
            # Same storage dtypes as the train path: float16 positions, float32 masses. The
            # per-point labels are cleared so the model is in the same state a compressed fit
            # would be (compute_evidence prefers the masses either way).
            model.reference = landmarks.half().to(device)
            model.reference_class_masses = masses.to(device)
            model.register_buffer("reference_targets", None)
            aurocs = _pair_aurocs(model, pair_features)
            test_s = timed_test(model, cfg, device)
            rows.append(
                {"m_landmarks": num_landmarks, "kmeans_seed": kmeans_seed, "kmeans_s": kmeans_s, "test_s": test_s}
                | {f"auroc_{n}": v for n, v in aurocs.items()}
            )
            _write_csv(out, fieldnames, rows)
            logger.info(
                "M=%d kmeans_seed=%d: kmeans %.1f s, test %.3f s, mean auroc %.4f",
                num_landmarks,
                kmeans_seed,
                kmeans_s,
                test_s,
                statistics.fmean(aurocs.values()),
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    logger.info("Done. Landmark ablation written to %s", out)


if __name__ == "__main__":
    main()
