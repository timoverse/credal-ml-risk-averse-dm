"""Wall-clock timing per credal method for the paper's timing table: train and test seconds.

Per method and per run this times (1) the full from-scratch training pipeline and (2) one pass
over the full test split that generates the credal set for every batch via the method's
representer. Training goes through the exact train_funcs.train_model dispatch that
training/train.py uses, on the identically composed train config, but with wandb fully stubbed
out: no runs, no logging, no artifact lookup, download, or upload. Trainers that normally reuse
a wandb base or wrapper artifact therefore train from scratch, and trainers that require a base
predictor (credal_rl_multinomial and friends) get one trained locally inside the timed window,
so the reported train time is the full cost of producing the credal predictor from raw data.
credal_ensembling and credal_wrapper train the exact same ensemble (they differ only in the
inference-time representer), so the pair is trained once per seed and both rows report the same
train time; their test passes are timed separately.
Nothing is saved except one csv with a row per method holding the raw per-run train and test
seconds (means and stds are computed downstream); the csv is rewritten after each method
finishes, and per-run times also go to the log.

Run from the project root, for example:
    python src/experiments/timing.py
    python src/experiments/timing.py --runs 1 --methods EffCre CreWare epochs=1 num_train=512

Positional arguments are hydra overrides applied to every composed train config (epochs=...,
device=..., val_split=..., method.params.num_members=..., ...). Run r uses seed --seed + r,
matching the multi-seed convention of the other experiments. torch.compile is off by default
(its one-time compilation cost would pollute the timings, and the eval scripts run uncompiled);
pass compile_forward=true to time the compiled pipeline instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import gc
import logging
import time
from typing import TYPE_CHECKING, Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from probly.representer import representer

import data
import models
import utils
from paths import RESULTS_PATH
from training import train_funcs

if TYPE_CHECKING:
    import torch.nn as nn
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

CONFIGS_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs")

# Paper-table rows in order: display name -> method config name.
TABLE_METHODS: dict[str, str] = {
    "CreEns": "credal_ensembling",
    "CreBNN": "credal_bnn",
    "CreWra": "credal_wrapper",
    "CreDRO": "credal_dro",
    "CreRL": "credal_relative_likelihood",
    "EffCre": "efficient_credal_prediction",
    "CreWare": "credal_rl_multinomial",
}

# credal_ensembling and credal_wrapper share the exact same trained ensemble (they differ only
# in the representer used at inference), so the pair is trained once per seed: the first of the
# two to run stores its weights and train seconds, the second reuses both and only times its
# own test pass.
WRAPPER_PAIR = {"credal_ensembling", "credal_wrapper"}

# Trainers that raise without a base predictor (normally downloaded from wandb). For these a
# base is trained locally inside the timed train window and served through the patched
# load_base_predictor, so their train time is also the full from-scratch cost.
NEEDS_BASE = {"credal_rl_multinomial"}

# Base predictor served by _patched_load_base_predictor during a NEEDS_BASE training window.
_LOCAL_BASE: nn.Module | None = None


class _WandbStub:
    """Stands in for the wandb module name inside train_funcs so no wandb code ever runs."""

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        """Ignore the step-metric declarations of the per-member training loops."""


class _NoOpRun:
    """Minimal stand-in for a wandb run: absorbs log calls, keeps summary in a plain dict."""

    def __init__(self) -> None:
        """Create the run with an empty summary dict."""
        self.summary: dict[str, Any] = {}

    def log(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Discard per-epoch metrics."""


def _patched_load_base_predictor(cfg: DictConfig, device: torch.device) -> tuple[nn.Module, str] | None:
    """Serve the locally trained base when one exists, else None (train from scratch)."""
    if _LOCAL_BASE is None:
        return None
    return _LOCAL_BASE, "local-timing-base"


def _patched_load_credal_wrapper_ensemble(cfg: DictConfig, device: torch.device) -> None:
    """Never reuse a wrapper ensemble, so credal_ensembling trains its members from scratch."""
    return None


def _compose_train_cfg(method: str, recipe: str, seed: int, overrides: list[str]) -> DictConfig:
    """Compose the exact cfg training/train.py would see for this method, recipe, and seed.

    Args:
        method: Method config name (configs/method/<method>.yaml).
        recipe: Recipe config name (configs/recipe/<recipe>.yaml).
        seed: Seed for this run.
        overrides: Extra hydra overrides forwarded from the command line.

    Returns:
        The composed train config.
    """
    full = [f"method={method}", f"recipe={recipe}", f"seed={seed}", "wandb.enabled=false"]
    # Compilation is off unless explicitly requested: its one-time cost would pollute the
    # timings, and the eval scripts run uncompiled forwards.
    if not any(o.split("=", 1)[0] == "compile_forward" for o in overrides):
        full.append("compile_forward=false")
    full.extend(overrides)
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        return compose(config_name="train", overrides=full)


def _sync(device: torch.device) -> None:
    """Block until pending kernels finish so perf_counter brackets the real work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _move_to_device(model: nn.Module | list[nn.Module], device: torch.device) -> None:
    """Move the model (or each member of a list ensemble) onto device."""
    if isinstance(model, list):
        for member in model:
            member.to(device)  # ty: ignore[unresolved-attribute]
    else:
        model.to(device)


def _maybe_compile_forward(model: nn.Module | list[nn.Module], device: torch.device, enable: bool) -> None:
    """torch.compile(model.forward) when on CUDA and enabled, as train.py does."""
    if not enable or device.type != "cuda":
        return
    if isinstance(model, list):
        for member in model:
            member.forward = torch.compile(member.forward)  # ty: ignore[unresolved-attribute]
    else:
        model.forward = torch.compile(model.forward)


def _set_eval(model: nn.Module | list[nn.Module]) -> None:
    """Put the model (or each member of a list ensemble) into eval mode."""
    members = model if isinstance(model, list) else [model]
    for member in members:
        member.eval()  # ty: ignore[unresolved-attribute]


def _build_model(cfg: DictConfig, device: torch.device) -> nn.Module | list[nn.Module]:
    """models.build_model for cfg, moved to device and optionally compiled, as train.py does."""
    params_section = cfg.method.get("params", {})
    params = OmegaConf.to_container(params_section, resolve=True) if params_section else {}
    model = models.build_model(
        cfg.method.name,
        cfg.base_model,
        num_classes=data.DATASET_NUM_CLASSES[cfg.dataset],
        pretrained=cfg.pretrained,
        model_type=cfg.model_type,
        params=params,  # ty: ignore[invalid-argument-type]
    )
    _move_to_device(model, device)
    _maybe_compile_forward(model, device, cfg.compile_forward)
    return model


def _timed_train(
    cfg: DictConfig,
    base_cfg: DictConfig | None,
    device: torch.device,
    run: _NoOpRun,
) -> tuple[nn.Module | list[nn.Module], float]:
    """Build loaders and models, then time the full training pipeline for cfg.method.

    Loader and model construction stay outside the timed window; everything train.py hands to
    train_funcs.train_model is inside it. When base_cfg is given (NEEDS_BASE methods), the base
    predictor's training runs first, inside the same window, and is served to the method's
    trainer through the patched load_base_predictor.

    Args:
        cfg: Composed train config for the method.
        base_cfg: Composed train config for method=base, or None when no base is required.
        device: Device to train on.
        run: No-op run passed through to the training routines.

    Returns:
        Tuple of the trained model and the elapsed train seconds.

    Raises:
        TypeError: The base build unexpectedly produced a list ensemble.
    """
    global _LOCAL_BASE
    loader_kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
    }
    train_loader, val_loader, _ = data.get_train_data(
        cfg.dataset, val_split=cfg.val_split, num_train=cfg.num_train, seed=cfg.seed, **loader_kwargs
    )
    model = _build_model(cfg, device)
    base_model: nn.Module | None = None
    if base_cfg is not None:
        built = _build_model(base_cfg, device)
        if isinstance(built, list):
            raise TypeError("method=base unexpectedly built a list ensemble.")
        base_model = built

    _LOCAL_BASE = None
    _sync(device)
    start = time.perf_counter()
    if base_model is not None and base_cfg is not None:
        train_funcs.train_model(base_model, train_loader, val_loader, base_cfg, device, run)
        _set_eval(base_model)
        _LOCAL_BASE = base_model
    train_funcs.train_model(model, train_loader, val_loader, cfg, device, run)  # ty: ignore[invalid-argument-type]
    _sync(device)
    elapsed = time.perf_counter() - start
    _LOCAL_BASE = None

    # Shut down the training loaders' workers before the test loader spawns its own,
    # mirroring train.py's cleanup between the train and test phases.
    del train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, elapsed


@torch.no_grad()
def timed_test(model: nn.Module | list[nn.Module], cfg: DictConfig, device: torch.device) -> float:
    """Time one pass over the full test split generating the credal set for every batch.

    PUBLIC because experiments/inference_timing.py times pre-trained wandb artifacts with the exact
    same routine; the two scripts' test seconds are only comparable while they share this function.

    Mirrors the eval scripts: representer(model).predict per batch under autocast per cfg.amp,
    channels_last inputs on cuda. One untimed warmup batch runs first so one-time kernel and
    cudnn setup cost is not attributed to the timed pass.

    Args:
        model: Trained predictor (nn.Module, or a list of members for CreRL).
        cfg: Composed train config the model was trained with.
        device: Inference device.

    Returns:
        Elapsed seconds for the full test pass.
    """
    _set_eval(model)
    rep = representer(model)
    test_loader = data.get_test_data(
        cfg.dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    amp_enabled = bool(cfg.amp)

    warmup_inputs, _ = next(iter(test_loader))
    warmup_inputs = warmup_inputs.to(device)
    if device.type == "cuda" and warmup_inputs.ndim >= 4:
        warmup_inputs = warmup_inputs.contiguous(memory_format=torch.channels_last)
    with torch.amp.autocast(device.type, enabled=amp_enabled):
        rep.predict(warmup_inputs)

    _sync(device)
    start = time.perf_counter()
    for inputs_, _ in test_loader:
        inputs = inputs_.to(device, non_blocking=True)
        if device.type == "cuda" and inputs.ndim >= 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            rep.predict(inputs)
    _sync(device)
    elapsed = time.perf_counter() - start

    del test_loader
    gc.collect()
    return elapsed


def _write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    """Rewrite the csv with one row per finished method.

    Columns are taken from the first row in insertion order: method, then the raw per-seed
    train and test seconds (train_s_seed<s> and test_s_seed<s>, one column per run's seed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_methods(names: list[str]) -> list[tuple[str, str]]:
    """Map table display names or raw method config names to (display, config) pairs.

    Public for the same reason as timed_test: inference_timing.py must resolve --methods to the
    identical display names, or its rows would not line up with the ones in this script's csvs.
    """
    reverse = {v: k for k, v in TABLE_METHODS.items()}
    return [(n, TABLE_METHODS[n]) if n in TABLE_METHODS else (reverse.get(n, n), n) for n in names]


def _parse_args() -> argparse.Namespace:
    """Parse script knobs plus hydra overrides forwarded to every composed train config."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(TABLE_METHODS),
        help="Table names (e.g. CreWare) or method config names (e.g. credal_rl_multinomial).",
    )
    parser.add_argument("--recipe", default="cifar10_resnet18", help="Recipe config name.")
    parser.add_argument("--runs", type=int, default=3, help="Repetitions per method.")
    parser.add_argument("--seed", type=int, default=1, help="Base seed; run r uses seed + r.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output csv; default timing_<recipe>.csv. Relative paths resolve inside the results directory.",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. epochs=5 device=cpu num_train=512.")
    # Overrides may be interspersed with the --flags (e.g. device=... before --out); plain
    # parse_args only accepts one contiguous positional chunk.
    return parser.parse_intermixed_args()


def main() -> None:
    """Time every method over the requested runs and write the summary csv.

    Raises:
        TypeError: A wrapper-pair build unexpectedly produced a list ensemble.
    """
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    train_funcs.load_base_predictor = _patched_load_base_predictor  # ty: ignore[invalid-assignment]
    train_funcs.load_credal_wrapper_ensemble = _patched_load_credal_wrapper_ensemble  # ty: ignore[invalid-assignment]
    train_funcs.wandb = _WandbStub()  # ty: ignore[invalid-assignment]
    run = _NoOpRun()

    methods = resolve_methods(args.methods)
    if args.out is None:
        out = RESULTS_PATH / f"timing_{args.recipe}.csv"
    else:
        # Relative --out resolves inside results/ so parallel jobs land there no matter the cwd.
        out = args.out if args.out.is_absolute() else RESULTS_PATH / args.out
    if out.exists():
        logger.warning("Overwriting existing %s", out)

    # Resolve the device once so auto-picking cannot switch GPUs between runs.
    first_cfg = _compose_train_cfg(methods[0][1], args.recipe, args.seed, args.overrides)
    device = utils.get_device(first_cfg.get("device"))
    logger.info("Running on device: %s", device)

    rows: list[dict[str, str | float]] = []
    # Trained wrapper-pair ensembles by seed: (cpu state dict, train seconds). Stored by the
    # first of the pair to run, popped by the second so both rows share the same train number.
    shared_train: dict[int, tuple[dict[str, torch.Tensor], float]] = {}
    seeds = [args.seed + r for r in range(args.runs)]
    for display, method in methods:
        train_times: list[float] = []
        test_times: list[float] = []
        for r, seed in enumerate(seeds):
            logger.info("=== %s (%s) run %d/%d, seed %d ===", display, method, r + 1, args.runs, seed)
            cfg = _compose_train_cfg(method, args.recipe, seed, args.overrides)
            base_cfg = _compose_train_cfg("base", args.recipe, seed, args.overrides) if method in NEEDS_BASE else None
            utils.set_seed(seed)
            if method in WRAPPER_PAIR and seed in shared_train:
                state, train_s = shared_train.pop(seed)
                model = _build_model(cfg, device)
                if isinstance(model, list):
                    raise TypeError("wrapper-pair method unexpectedly built a list ensemble.")
                model.load_state_dict(state)
                logger.info("%s reuses the ensemble trained for its pair method (shared train time).", display)
            else:
                model, train_s = _timed_train(cfg, base_cfg, device, run)
                if method in WRAPPER_PAIR and not isinstance(model, list):
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    shared_train[seed] = (state, train_s)
            test_s = timed_test(model, cfg, device)
            train_times.append(train_s)
            test_times.append(test_s)
            logger.info("%s run %d: train %.2f s, test %.2f s", display, r + 1, train_s, test_s)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        row: dict[str, str | float] = {"method": display}
        row.update({f"train_s_seed{s}": t for s, t in zip(seeds, train_times, strict=True)})
        row.update({f"test_s_seed{s}": t for s, t in zip(seeds, test_times, strict=True)})
        rows.append(row)
        _write_csv(out, rows)
        logger.info("%s done (%d runs); csv updated: %s", display, args.runs, out)
    logger.info("Done. Timing table written to %s", out)


if __name__ == "__main__":
    main()
