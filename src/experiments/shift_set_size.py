"""Set-size-under-shift evaluation: credal set size per corrupted (corruption, severity) cell.

Sweeps the same (corruption, severity) grid as shift_risk_metric (CIFAR-10-C or MedMNIST-C,
per data.SHIFT_CORRUPTIONS) and records probly's set-size efficiency (1 - mean interval width)
plus the regular point-prediction accuracy per cell (evaluate()'s point semantics: the MLE
member, anchor, base, or kernel vote where one exists, the representation barycenter for the
ensemble methods) -- deliberately independent of the credal set, so the accuracy overlay is the
model-degradation signal the set growth is motivated against. The credal set, and hence its
size, is independent of any decision rule or per-instance loss, so no rule, loss, or CVaR
calibration is involved. Per-cell results go to the artifact's wandb run summary as
shift/<corruption>/<severity>/set/{efficiency,accuracy} keys ("set" = rule-independent
sentinel), parsed into the plotting cache by plotting/wandb_cache.py as kind="shift" rows with
loss="efficiency" or loss="accuracy".

Run: python src/experiments/shift_set_size.py method=credal_wrapper recipe=cifar10_resnet18
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import logging
from typing import TYPE_CHECKING

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from probly.decider import categorical_from_mean
from probly.metrics import efficiency
from probly.representation.credal_set.torch import TorchConvexCredalSet, TorchProbabilityIntervalsCredalSet
from probly.representer import representer
from tqdm import tqdm

import data
import utils
from artifacts import load_model_for_evaluation, resolve_artifact_name
from training.train_funcs import extract_point_predictor

if TYPE_CHECKING:
    from probly.representer import Representer
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


@torch.no_grad()
def _collect_set_metrics(
    rep: Representer,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    point: torch.nn.Module | None = None,
) -> tuple[float, float]:
    """Single inference pass returning probly's set-size efficiency and the point accuracy.

    The accuracy is that of the method's regular point prediction, matching evaluate()'s
    semantics and independent of the credal set: when a designated point predictor exists
    (mle_member, CreRL's MLE member, ECP's wrapped base, the multinomial vote) its argmax is
    scored with one extra forward per batch; when the point object is the full model
    (credal_wrapper, credal_ensembling, credal_bnn) the accuracy comes from the representation
    barycenter, categorical_from_mean of the already-computed set (the ensemble-mean prediction,
    no extra forward). This keeps the accuracy overlay a pure model-degradation signal, so set
    growth under shift is motivated against it rather than entangled with it. The tqdm bar shows
    the running mean set size (1 - efficiency) as batches complete.

    Args:
        rep: representer(model); must produce credal sets.
        loader: Loader for one (corruption, severity) cell (or the clean test set).
        device: Inference device.
        amp_enabled: Wrap forward in autocast.
        point: The designated point predictor, or None to score the representation barycenter.

    Returns:
        (efficiency, accuracy): probly's credal set-size efficiency 1 - mean(upper - lower) and
        the point-prediction accuracy, both batch-size-weighted over the loader.

    Raises:
        TypeError: the representer does not produce credal sets (e.g. base), so set size is
            undefined.
    """
    eff_weighted = 0.0
    n_correct = 0
    n_total = 0
    pbar = tqdm(loader, desc="Inference")
    for inputs_, targets_ in pbar:
        inputs = inputs_.to(device, non_blocking=True)
        if device.type == "cuda" and inputs.ndim >= 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            rep_out = rep.predict(inputs)
            if not isinstance(rep_out, TorchProbabilityIntervalsCredalSet | TorchConvexCredalSet):
                raise TypeError(
                    f"Set size needs a credal representer but rep.predict produced {type(rep_out).__name__}; "
                    "set-size evaluation is undefined for non-credal methods (e.g. base)."
                )
            if point is None:
                preds = categorical_from_mean(rep_out).probabilities.argmax(dim=-1)  # ty: ignore[unresolved-attribute]
            else:
                preds = point(inputs).argmax(dim=-1)
        targets = torch.as_tensor(targets_).long().flatten()
        n_correct += int((preds.cpu() == targets).sum())
        eff_weighted += float(efficiency(rep_out)) * inputs.shape[0]
        n_total += inputs.shape[0]
        pbar.set_postfix({"set_size": f"{1.0 - eff_weighted / n_total:.4f}", "acc": f"{n_correct / n_total:.3f}"})
    return eff_weighted / n_total, n_correct / n_total


@hydra.main(version_base=None, config_path="../../configs/", config_name="shift_set_size")
def main(cfg: DictConfig) -> None:
    """Load artifact, sweep the corruption benchmark's (corruption, severity) grid, report set size per cell."""
    logger.info("Shift set-size configuration:\n%s", OmegaConf.to_yaml(cfg))
    utils.set_seed(cfg.seed)
    device = utils.get_device(cfg.get("device"))
    logger.info("Running on device: %s", device)

    artifact_name = resolve_artifact_name(cfg)
    logger.info("Loading artifact %s from %s/%s", artifact_name, cfg.wandb.entity, cfg.wandb.project)
    model, train_cfg, run_id = load_model_for_evaluation(cfg, device)
    if train_cfg["method"]["name"] != cfg.method.name:
        raise RuntimeError(
            f"Artifact method ({train_cfg['method']['name']!r}) does not match cfg.method.name "
            f"({cfg.method.name!r}); check method/recipe/seed."
        )
    logger.info("Loaded model trained in wandb run %s", run_id)
    rep = representer(model)
    # Regular point predictor for the accuracy overlay: the same extraction evaluate() uses, so
    # the shift accuracy and test_acc share their semantics. None (wrapper, ensembling, bnn)
    # means the accuracy comes from the representation barycenter, reusing the per-batch set.
    point = extract_point_predictor(model)
    if point is not None:
        point.eval()

    # corruptions: "all" (or null/empty) sweeps the dataset's benchmark corruption set; a list of
    # names runs that subset. Validate upfront so a typo fails here, not an hour into the sweep.
    corruptions = data.resolve_shift_corruptions(cfg.dataset, cfg.get("corruptions"))
    severities = [int(s) for s in cfg.severities]
    logger.info(
        "Sweeping %d corruptions x %d severities (include_clean=%s)",
        len(corruptions),
        len(severities),
        cfg.include_clean,
    )

    # One flat (corruption, severity) work list; the clean baseline rides along as severity 0 so the
    # sweep loop and its per-cell memory cleanup stay a single code path.
    cells: list[tuple[str, int]] = []
    if cfg.include_clean:
        cells.append(("clean", 0))
    cells.extend((corruption, severity) for corruption in corruptions for severity in severities)

    records: list[dict[str, object]] = []
    for cell_idx, (corruption, severity) in enumerate(cells, start=1):
        logger.info("[%d/%d] %s severity %d", cell_idx, len(cells), corruption, severity)
        if corruption == "clean":
            # Workers come from the recipe. Unlike the risk sweep (small cells, solver-bound, where
            # worker startup dominates), this sweep is forward-pass bound on large cells, so
            # overlapping the CPU transform with GPU compute pays: a cifar10 all-corruptions job
            # is 760k images of transform, ~1 min that workers hide entirely.
            loader = data.get_test_data(
                cfg.dataset, batch_size=cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=True
            )
        else:
            loader = data.get_shift_data(
                cfg.dataset,
                corruption,
                severity,
                batch_size=cfg.eval_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
        eff, acc = _collect_set_metrics(rep, loader, device, bool(cfg.amp), point=point)
        records.append({"corruption": corruption, "severity": severity, "efficiency": eff, "accuracy": acc})
        logger.info(
            "[%d/%d] %s sev %d: set_size=%.4f (efficiency=%.4f) accuracy=%.4f",
            cell_idx,
            len(cells),
            corruption,
            severity,
            1.0 - eff,
            eff,
            acc,
        )

        # Free the loader (and its worker processes) before building the next cell's loader; 76 cells
        # in one process otherwise keep every dataset slice and CUDA block resident.
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if cfg.wandb.enabled:
        run = wandb.init(id=run_id, entity=cfg.wandb.entity, project=cfg.wandb.project, resume="must")
        # One scalar summary key per cell: the run summary is the results database (parsed by
        # plotting/wandb_cache.py as kind="shift" rows). The set size is decision-rule- and
        # loss-independent, so the key carries only the fixed sentinel "set" -- re-running the
        # same artifact overwrites the same key instead of duplicating.
        run.summary.update(
            {f"shift/{r['corruption']}/{r['severity']}/set/efficiency": r["efficiency"] for r in records}
            | {f"shift/{r['corruption']}/{r['severity']}/set/accuracy": r["accuracy"] for r in records}
        )
        run.finish()

    del model, rep
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
