"""Inference-only wall-clock timing of ALREADY TRAINED predictors: test seconds per method and seed.

Why this exists next to experiments/timing.py: that script times train + test in one go and therefore
has to train from scratch, which for credal_bnn is the ~5 h job that dominates the timing table. The
test-pass number, though, needs no fresh training -- the models are already in W&B from the OOD
experiment. This script downloads those artifacts (the same ones experiments/ood_detection.py
evaluates, resolved by artifacts.resolve_artifact_name from method/recipe/seed) and times only the
test pass, so the CreBNN row of the timing table can be filled in minutes instead of 15 GPU-hours.

The measurement is not a re-implementation: the timed window is timing.timed_test itself, called on
a model rebuilt from the checkpoint. Same test split, same batch size and amp setting from the
recipe, same one untimed warmup batch, same representer(model).predict per batch, same
channels_last inputs and cuda synchronisation around the timer. So the seconds produced here are
directly comparable to the test_s columns of results/timing/*.csv. Training is NOT timed and no
train_s column is written -- there is no from-scratch training in this script at all.

Only inference is measured, so the numbers say nothing about training cost; the model download
happens outside the timed window.

Run from the project root (W&B credentials required, unlike timing.py):
    python src/experiments/inference_timing.py
    python src/experiments/inference_timing.py --methods CreBNN CreWra --seeds 1 2 3 --repeats 3
    python src/experiments/inference_timing.py --methods CreBNN artifact_source=local device=cuda:0

Positional arguments are hydra overrides applied to every composed train config (device=...,
batch_size=..., wandb.project=..., artifact_source=local to read CHECKPOINTS_PATH instead of W&B).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The experiments dir itself, so `timing` (the train+test timing script) is importable: its
# timed_test is the measurement, shared rather than copied so the two scripts cannot drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import csv
import gc
import logging
import statistics
from typing import TYPE_CHECKING

import torch
from hydra import compose, initialize_config_dir
from timing import TABLE_METHODS, resolve_methods, timed_test

import utils
from artifacts import describe_run_source, load_model_for_evaluation, resolve_artifact_name
from paths import RESULTS_PATH

if TYPE_CHECKING:
    import torch.nn as nn
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

CONFIGS_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs")


def _compose_train_cfg(method: str, recipe: str, seed: int, overrides: list[str]) -> DictConfig:
    """Compose the train config for this method/recipe/seed, as timing.py and train.py do.

    The train config (not the eval ones) is composed on purpose: it is what resolve_artifact_name
    reads to name the artifact, and it carries the loader knobs (batch_size, num_workers,
    pin_memory) and amp flag that timed_test needs -- the identical values the timing experiment
    measured with, since both come from the same recipe.

    Args:
        method: Method config name (configs/method/<method>.yaml).
        recipe: Recipe config name (configs/recipe/<recipe>.yaml).
        seed: Seed of the trained artifact to load.
        overrides: Extra hydra overrides forwarded from the command line.

    Returns:
        The composed train config.
    """
    full = [f"method={method}", f"recipe={recipe}", f"seed={seed}", *overrides]
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        return compose(config_name="train", overrides=full)


def _load_trained_model(cfg: DictConfig, device: torch.device) -> nn.Module | list[nn.Module]:
    """Fetch the trained artifact for cfg and rebuild it on device, outside any timed window.

    Args:
        cfg: Composed train config identifying the artifact.
        device: Inference device.

    Returns:
        The rebuilt predictor, in eval mode.

    Raises:
        RuntimeError: The artifact was trained with a different method than cfg asks for, which
            would silently time the wrong predictor (same check as ood_detection.py).
    """
    artifact_name = resolve_artifact_name(cfg)
    model, train_cfg, run_id = load_model_for_evaluation(cfg, device)
    if train_cfg["method"]["name"] != cfg.method.name:
        msg = (
            f"Artifact {artifact_name!r} was trained with method {train_cfg['method']['name']!r}, "
            f"not {cfg.method.name!r}; check method/recipe/seed."
        )
        raise RuntimeError(msg)
    logger.info("Loaded %s from %s", artifact_name, describe_run_source(run_id))
    return model


def _write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    """Rewrite the csv with one row per finished method.

    Columns are method plus one test_s_seed<s> per seed, deliberately the same spelling the
    train+test timing csvs use, so a row from here drops straight into the merged timing table.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    """Parse script knobs plus hydra overrides forwarded to every composed train config."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["CreBNN"],
        help="Table names (e.g. CreBNN) or method config names (e.g. credal_bnn).",
    )
    parser.add_argument("--recipe", default="cifar10_resnet18", help="Recipe config name.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3], help="Seeds of the trained artifacts.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Timed test passes per seed. The csv holds their mean; every single pass goes to the log.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output csv; default inference_timing_<recipe>.csv. Relative paths resolve inside results/.",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. device=cuda:0 artifact_source=local.")
    # Overrides may be interspersed with the --flags (e.g. device=... before --out); plain
    # parse_args only accepts one contiguous positional chunk.
    return parser.parse_intermixed_args()


def main() -> None:
    """Time the test pass of every requested method at every seed and write the summary csv."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    methods = resolve_methods(args.methods)
    if args.out is None:
        out = RESULTS_PATH / f"inference_timing_{args.recipe}.csv"
    else:
        # Relative --out resolves inside results/ so parallel jobs land there no matter the cwd.
        out = args.out if args.out.is_absolute() else RESULTS_PATH / args.out
    if out.exists():
        logger.warning("Overwriting existing %s", out)

    # Resolve the device once so auto-picking cannot switch GPUs between seeds.
    first_cfg = _compose_train_cfg(methods[0][1], args.recipe, args.seeds[0], args.overrides)
    device = utils.get_device(first_cfg.get("device"))
    logger.info("Running on device: %s", device)
    logger.info(
        "Timing inference only: %s over seeds %s, %d pass(es) each.",
        ", ".join(display for display, _ in methods),
        args.seeds,
        args.repeats,
    )

    rows: list[dict[str, str | float]] = []
    for display, method in methods:
        per_seed: list[float] = []
        for seed in args.seeds:
            logger.info("=== %s (%s) seed %d ===", display, method, seed)
            cfg = _compose_train_cfg(method, args.recipe, seed, args.overrides)
            # Seeded like the eval scripts: a Bayesian predictor samples weights at inference, so
            # the pass is only reproducible with the seed set, and the timer must not be the first
            # thing to consume randomness.
            utils.set_seed(seed)
            model = _load_trained_model(cfg, device)
            passes = []
            for r in range(args.repeats):
                test_s = timed_test(model, cfg, device)
                passes.append(test_s)
                logger.info("%s seed %d pass %d/%d: test %.3f s", display, seed, r + 1, args.repeats, test_s)
            mean_s = statistics.fmean(passes)
            if args.repeats > 1:
                logger.info("%s seed %d: mean %.3f s over %d passes", display, seed, mean_s, args.repeats)
            per_seed.append(mean_s)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        row: dict[str, str | float] = {"method": display}
        row.update({f"test_s_seed{s}": t for s, t in zip(args.seeds, per_seed, strict=True)})
        rows.append(row)
        _write_csv(out, rows)
        logger.info("%s done: %s; csv updated: %s", display, [f"{t:.3f}s" for t in per_seed], out)
    logger.info("Done. Inference timings written to %s (methods known to the table: %s)", out, list(TABLE_METHODS))


if __name__ == "__main__":
    main()
