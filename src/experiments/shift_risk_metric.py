"""Distribution-shift evaluation: mean and CVaR of the loss under benchmark corruptions.

Sweeps every (corruption, severity) cell of the dataset's corruption benchmark (CIFAR-10-C or MedMNIST-C,
per data.SHIFT_CORRUPTIONS) and reports each configured aggregation per cell.
The decision rule and (for cvar_minimax) its VaR threshold are fixed on the clean data and
then applied under shift. Per-cell results go to the artifact's wandb run summary as
shift/<corruption>/<severity>/<decision_rule>/<aggregation>/<loss>[/beta=<cvar_beta>] keys --
the wandb-as-database pattern, parsed into the plotting cache by plotting/wandb_cache.py as
kind="shift" rows. Set sizes under shift are
recorded by the separate experiments/shift_set_size.py: the credal set, and hence its size,
is independent of the decision rule and loss, so it does not belong in these summaries.

Run: python src/experiments/shift_risk_metric.py method=credal_wrapper recipe=cifar10_resnet18
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import logging
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf
from probly.representer import representer
from tqdm import tqdm

import data
import utils
from artifacts import load_model_for_evaluation, resolve_artifact_name
from decision_rules import (
    SUPPORTED_DECISION_RULES,
    apply_decision_rule,
    calibrate_var_thresholds,
    cvar_minimax_at,
    maxent_predictions,
)
from metrics import AGGREGATION_REGISTRY, PER_INSTANCE_LOSS_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy.typing as npt
    from probly.representer import Representer
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


@torch.no_grad()
def _collect_decisions_targets(
    rep: Representer,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    loss: str,
    decision_rule: str,
    var_threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single inference pass: apply the decision rule per batch and gather (probs, targets) on CPU.

    The tqdm bar shows the running mean loss as batches complete, so long sweeps give live
    terminal feedback.

    Args:
        rep: representer for the chosen (model, decision_rule) pair.
        loader: Test DataLoader.
        device: Inference device.
        amp_enabled: Wrap forward in autocast.
        loss: Per-instance loss the decision rule targets.
        decision_rule: One of SUPPORTED_DECISION_RULES.
        var_threshold: Calibrated VaR threshold passed through to apply_decision_rule for cvar_minimax.

    Returns:
        (probs, targets) on CPU, shapes (N, K) and (N,).
    """
    loss_fn = PER_INSTANCE_LOSS_REGISTRY[loss]
    probs_chunks: list[torch.Tensor] = []
    tgts_chunks: list[torch.Tensor] = []
    loss_sum = 0.0
    n_total = 0
    pbar = tqdm(loader, desc="Inference")
    for inputs_, targets_ in pbar:
        inputs = inputs_.to(device, non_blocking=True)
        if device.type == "cuda" and inputs.ndim >= 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            rep_out = rep.predict(inputs)
            batch_probs = apply_decision_rule(rep_out, loss, decision_rule, var_threshold=var_threshold)
        n_total += inputs.shape[0]
        batch_probs_cpu = batch_probs.detach().float().cpu()
        probs_chunks.append(batch_probs_cpu)
        tgts_chunks.append(targets_)
        # Live terminal feedback: running mean loss on the bar.
        loss_sum += float(loss_fn(batch_probs_cpu.numpy(), targets_.numpy().astype(np.int64)).sum())
        pbar.set_postfix({loss: f"{loss_sum / n_total:.4f}"})
    return torch.cat(probs_chunks), torch.cat(tgts_chunks)


def _evaluate_cell(
    rep: Representer,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
    loss_fn: Callable[[npt.NDArray, npt.NDArray], npt.NDArray[np.floating]],
    var_threshold: float | None,
) -> dict[str, float]:
    """Run inference on one loader and aggregate the per-instance loss every configured way.

    Args:
        rep: representer(model).
        loader: Loader for one (corruption, severity) cell (or the clean test set).
        device: Inference device.
        cfg: Run config (uses amp, loss, decision_rule, aggregations).
        loss_fn: Per-instance loss from PER_INSTANCE_LOSS_REGISTRY[cfg.loss].
        var_threshold: Calibrated cvar_minimax threshold, or None for other rules.

    Returns:
        {aggregation_name: value} over cfg.aggregations for this cell.
    """
    probs_t, targets_t = _collect_decisions_targets(
        rep, loader, device, bool(cfg.amp), cfg.loss, cfg.decision_rule, var_threshold=var_threshold
    )
    probs = probs_t.numpy()
    targets = targets_t.numpy().astype(np.int64)
    losses = loss_fn(probs, targets)
    return {agg: float(AGGREGATION_REGISTRY[agg](losses)) for agg in cfg.aggregations}


@torch.no_grad()
def _evaluate_cell_cvar_minimax(
    rep: Representer,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
    loss_fn: Callable[[npt.NDArray, npt.NDArray], npt.NDArray[np.floating]],
    v_by_beta: dict[float, float],
) -> dict[float, dict[str, float]]:
    """Evaluate one cell under cvar_minimax at every beta from a single inference pass.

    The credal sets do not depend on beta, so the forward pass is shared and the solver runs once
    with the calibrated thresholds stacked on a leading axis (cvar_minimax_at), instead of once
    per beta and batch. Per instance the solve is identical to the sequential per-batch path; the
    maxent warm start is per batch, exactly as that path computed it. The solver runs in full
    precision outside the collection autocast, as calibration always has.

    Args:
        rep: representer(model); must produce credal sets.
        loader: Loader for one (corruption, severity) cell (or the clean test set).
        device: Inference device.
        cfg: Run config (uses amp, loss, aggregations).
        loss_fn: Per-instance loss from PER_INSTANCE_LOSS_REGISTRY[cfg.loss].
        v_by_beta: Calibrated threshold per beta, from calibrate_var_thresholds.

    Returns:
        {beta: {aggregation_name: value}} over cfg.aggregations for this cell.
    """
    rep_outs, targets = utils.collect_credal_sets_targets(rep, loader, device, bool(cfg.amp))
    p_maxent = maxent_predictions(rep_outs)  # (N, K)
    betas = list(v_by_beta)
    probs = cvar_minimax_at(rep_outs, [v_by_beta[b] for b in betas], cfg.loss, p_maxent)  # (B, N, K)
    targets_np = torch.cat(targets).cpu().numpy().astype(np.int64)
    results: dict[float, dict[str, float]] = {}
    for beta, p in zip(betas, probs.detach().float().cpu().numpy(), strict=True):
        losses = loss_fn(p, targets_np)
        results[beta] = {agg: float(AGGREGATION_REGISTRY[agg](losses)) for agg in cfg.aggregations}
    return results


def _warn_nonfinite(cells: Iterable[dict[str, float]], corruption: str, severity: int) -> None:
    """Log a warning when a cell's aggregated loss is not finite.

    W&B silently drops NaN summary values, so without this a NaN cell just goes missing from the
    results while the job reports success. Seen in practice: under amp, one credal_wrapper member
    overflowed float16 on heavily corrupted PathMNIST images, giving NaN probability boxes; rerun
    such a cell with amp=false.

    Args:
        cells: Aggregation dicts of one (corruption, severity) cell (one per beta for cvar_minimax).
        corruption: Corruption name of the cell.
        severity: Severity of the cell.
    """
    bad = sorted({agg for cell in cells for agg, value in cell.items() if not np.isfinite(value)})
    if bad:
        logger.warning(
            "Non-finite %s for %s severity %d (NaN values are dropped from the W&B summary); "
            "under amp this can be a float16 overflow, rerun with amp=false.",
            bad,
            corruption,
            severity,
        )


@hydra.main(version_base=None, config_path="../../configs/", config_name="shift_risk_metric")
def main(cfg: DictConfig) -> None:
    """Load artifact, sweep the corruption benchmark's (corruption, severity) grid, report the loss per cell."""
    logger.info("Shift-evaluation configuration:\n%s", OmegaConf.to_yaml(cfg))
    utils.set_seed(cfg.seed)
    device = utils.get_device(cfg.get("device"))
    logger.info("Running on device: %s", device)

    if cfg.decision_rule not in SUPPORTED_DECISION_RULES:
        raise ValueError(f"Unknown decision_rule={cfg.decision_rule!r}. Choose from {list(SUPPORTED_DECISION_RULES)}.")
    if cfg.loss not in PER_INSTANCE_LOSS_REGISTRY:
        raise ValueError(f"Unknown loss={cfg.loss!r}. Choose from {sorted(PER_INSTANCE_LOSS_REGISTRY)}.")
    unknown_aggs = [a for a in cfg.aggregations if a not in AGGREGATION_REGISTRY]
    if unknown_aggs:
        raise ValueError(f"Unknown aggregations {unknown_aggs}. Choose from {sorted(AGGREGATION_REGISTRY)}.")
    loss_fn = PER_INSTANCE_LOSS_REGISTRY[cfg.loss]

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

    # cvar_minimax calibrates its VaR threshold v on the clean validation split once; it is then held
    # fixed and applied to every shifted cell (calibrate on clean data, deploy under shift). Every
    # other rule skips this and leaves v_by_beta empty. cvar_beta accepts a single level or a list:
    # the credal sets and the calibration grid solve do not depend on beta, so evaluating many
    # levels in one run shares the forward passes and calibration that per-beta runs would repeat.
    v_by_beta: dict[float, float] = {}
    if cfg.decision_rule == "cvar_minimax":
        raw_beta = cfg.get("cvar_beta")
        beta_list = list(raw_beta) if isinstance(raw_beta, list | ListConfig) else [raw_beta]
        if not beta_list or any(b is None or not 0.0 < float(b) <= 1.0 for b in beta_list):
            raise ValueError(f"decision_rule=cvar_minimax needs cvar_beta in (0, 1], got {raw_beta!r}.")
        betas = [float(b) for b in beta_list]
        val_split = train_cfg.get("val_split", 0.0)
        if val_split <= 0:
            raise ValueError(
                "decision_rule=cvar_minimax calibrates v on a validation split, but the artifact was "
                f"trained with val_split={val_split}; retrain with val_split > 0."
            )
        # num_workers=0 here and for the cells below: these datasets are in-memory arrays with
        # light transforms (a full dermamnist cell transforms in ~0.13 s single-process), so
        # worker processes cost more in per-loader startup than they can save.
        _, val_loader, _ = data.get_train_data(
            cfg.dataset,
            val_split=val_split,
            seed=train_cfg.get("seed"),
            batch_size=cfg.eval_batch_size,
            num_workers=0,
            pin_memory=True,
        )
        if val_loader is None:
            raise RuntimeError(f"get_train_data returned no validation loader for val_split={val_split}.")
        val_sets, val_targets = utils.collect_credal_sets_targets(rep, val_loader, device, bool(cfg.amp))
        v_by_beta = calibrate_var_thresholds(val_sets, val_targets, betas, cfg.loss)
        for beta, threshold in v_by_beta.items():
            logger.info("Calibrated cvar_minimax threshold v = %.4f at beta = %.3f", threshold, beta)
        del val_sets, val_targets, val_loader

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

    # Records carry a beta column: None for beta-free rules, one record per beta for cvar_minimax
    # (whose per-cell evaluation shares one inference pass across all betas).
    records: list[dict[str, object]] = []
    for cell_idx, (corruption, severity) in enumerate(cells, start=1):
        logger.info("[%d/%d] %s severity %d", cell_idx, len(cells), corruption, severity)
        if corruption == "clean":
            loader = data.get_test_data(cfg.dataset, batch_size=cfg.eval_batch_size, num_workers=0, pin_memory=True)
        else:
            loader = data.get_shift_data(
                cfg.dataset, corruption, severity, batch_size=cfg.eval_batch_size, num_workers=0, pin_memory=True
            )
        if cfg.decision_rule == "cvar_minimax" and cfg.loss != "zero_one":
            per_beta = _evaluate_cell_cvar_minimax(rep, loader, device, cfg, loss_fn, v_by_beta)
            _warn_nonfinite(per_beta.values(), corruption, severity)
            for beta, cell in per_beta.items():
                records.append({"corruption": corruption, "severity": severity, "beta": beta, **cell})
                values = {k: round(v, 4) for k, v in cell.items()}
                logger.info("[%d/%d] %s sev %d beta %g: %s", cell_idx, len(cells), corruption, severity, beta, values)
        else:
            # Beta-free rules; also cvar_minimax under zero_one, which reduces to plain minimax
            # and ignores the threshold, so every beta shares one per-batch evaluation.
            threshold = next(iter(v_by_beta.values()), None)
            cell = _evaluate_cell(rep, loader, device, cfg, loss_fn, threshold)
            _warn_nonfinite([cell], corruption, severity)
            for beta in list(v_by_beta) or [None]:
                records.append({"corruption": corruption, "severity": severity, "beta": beta, **cell})
            values = {k: round(v, 4) for k, v in cell.items()}
            logger.info("[%d/%d] %s sev %d: %s", cell_idx, len(cells), corruption, severity, values)

        # Free the loader (and its worker processes) before building the next cell's loader; 76 cells
        # in one process otherwise keep every dataset slice and CUDA block resident.
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if cfg.wandb.enabled:
        run = wandb.init(id=run_id, entity=cfg.wandb.entity, project=cfg.wandb.project, resume="must")
        # One scalar summary key per (cell, aggregation):
        # the run summary is the results database (parsed by plotting/wandb_cache.py as kind="shift").
        # beta rides in the key for cvar_minimax so different-beta passes do not collide.
        summary_updates: dict[str, float] = {}
        for r in records:
            beta_suffix = "" if r["beta"] is None else f"/beta={float(r['beta'])}"  # ty: ignore[invalid-argument-type]
            for agg in cfg.aggregations:
                key = f"shift/{r['corruption']}/{r['severity']}/{cfg.decision_rule}/{agg}/{cfg.loss}{beta_suffix}"
                summary_updates[key] = float(r[agg])  # ty: ignore[invalid-argument-type]
        run.summary.update(summary_updates)
        # v depends on the loss, so a non-log-loss pass puts the loss in the key rather than overwriting
        # the log-loss threshold; log loss keeps the original key, which earlier runs already hold.
        loss_segment = "" if cfg.loss == "log_loss" else f"{cfg.loss}/"
        for beta, threshold in v_by_beta.items():
            run.summary[f"cvar_calibration/v_star/{loss_segment}beta={float(beta)}"] = threshold
        run.finish()

    del model, rep
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
