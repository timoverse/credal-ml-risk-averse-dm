"""Coverage and efficiency of the predicted credal sets against first-order test targets.

The test loader comes from data.get_first_order_data, whose targets are ground-truth first-order
label distributions over the base dataset's test instances (cifar10 -> CIFAR-10H human soft
labels). Coverage is the fraction of instances whose target distribution lies in the predicted
credal set, with the membership test matching each method's declared representation:
probability-interval sets (credal_wrapper, credal_rl_multinomial, ...) use probly's containment
coverage lower <= p <= upper, exact because the box is the credal set; convex vertex sets
(credal_ensembling, ...) use probly's convex_hull_coverage, the exact LP membership test, because
containment in the box envelope of a hull would over-cover. Efficiency is probly's set-size
efficiency 1 - mean(upper - lower) of the envelope in both cases, so higher means a tighter
credal set. Both are rule-independent properties of the credal set itself, hence no decision rule
or loss is involved. Results go to the artifact's wandb run summary as cov_eff/coverage and
cov_eff/efficiency -- the same wandb-as-database pattern as shift_risk_metric's shift/* keys.

Run: python src/experiments/coverage_efficiency.py method=credal_wrapper recipe=cifar10_resnet18
Sweep the coverage/efficiency trade-off across methods and seeds via --multirun, e.g.:
  python src/experiments/coverage_efficiency.py method=credal_wrapper,credal_ensembling \
      recipe=cifar10_resnet18 seed=1,2,3 --multirun
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
from probly.metrics import convex_hull_coverage, coverage, efficiency
from probly.representation.credal_set.torch import TorchConvexCredalSet, TorchProbabilityIntervalsCredalSet
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution
from probly.representer import representer
from tqdm import tqdm

import data
import utils
from artifacts import load_model_for_evaluation, resolve_artifact_name

if TYPE_CHECKING:
    from probly.representer import Representer
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


@torch.no_grad()
def _evaluate_coverage_efficiency(
    rep: Representer,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    """Single inference pass: batch-size-weighted coverage and efficiency of the predicted credal sets.

    Args:
        rep: representer for the model; predict must return a credal set per batch.
        loader: Test DataLoader.
        device: Inference device.
        amp_enabled: Wrap forward in autocast.

    Returns:
        (coverage, efficiency). Coverage of the loader's first-order target distributions: exact
        box containment for probability-interval sets, exact LP hull membership for convex vertex
        sets. Efficiency is probly's set-size efficiency 1 - mean(upper - lower) of the envelope.
        Both are means over instances, so the batch-size-weighted aggregation over the loader is
        exact.

    Raises:
        TypeError: The representer does not produce credal sets (e.g. method=base).
    """
    cov_weighted = 0.0
    eff_weighted = 0.0
    n_total = 0
    for inputs_, targets_ in tqdm(loader, desc="Inference"):
        inputs = inputs_.to(device, non_blocking=True)
        if device.type == "cuda" and inputs.ndim >= 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            rep_out = rep.predict(inputs)
        # A vertex set's box envelope over-covers (a target can lie inside the box yet outside the
        # hull), so convex sets get probly's exact LP hull-membership test instead of the generic
        # envelope-containment coverage.
        if isinstance(rep_out, TorchConvexCredalSet):
            cov_batch = float(convex_hull_coverage(rep_out, TorchProbabilityCategoricalDistribution(targets_)))
        elif isinstance(rep_out, TorchProbabilityIntervalsCredalSet):
            cov_batch = float(coverage(rep_out, targets_))
        else:
            raise TypeError(
                f"coverage/efficiency need a credal representation, got {type(rep_out).__name__}; "
                "run this on a credal method."
            )
        cov_weighted += cov_batch * inputs.shape[0]
        eff_weighted += float(efficiency(rep_out)) * inputs.shape[0]
        n_total += inputs.shape[0]
    return cov_weighted / n_total, eff_weighted / n_total


@hydra.main(version_base=None, config_path="../../configs/", config_name="coverage_efficiency")
def main(cfg: DictConfig) -> None:
    """Load artifact, compute test-set coverage and efficiency of its credal sets, append to the run's wandb summary."""
    logger.info("Coverage/efficiency configuration:\n%s", OmegaConf.to_yaml(cfg))
    utils.set_seed(cfg.seed)
    device = utils.get_device(cfg.get("device"))
    logger.info("Running on device: %s", device)

    # Built before the artifact download so a dataset without a first-order counterpart fails fast.
    test_loader = data.get_first_order_data(cfg.dataset, batch_size=cfg.batch_size, num_workers=2, pin_memory=True)

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

    cov, eff = _evaluate_coverage_efficiency(rep, test_loader, device, bool(cfg.amp))
    logger.info("cov_eff/coverage = %.6f, cov_eff/efficiency = %.6f", cov, eff)

    if cfg.wandb.enabled:
        run = wandb.init(id=run_id, entity=cfg.wandb.entity, project=cfg.wandb.project, resume="must")
        run.summary["cov_eff/coverage"] = cov
        run.summary["cov_eff/efficiency"] = eff
        run.finish()

    # Release GPU memory before the next --multirun iteration: Hydra's basic launcher runs all
    # iterations in one Python process and PyTorch's caching allocator keeps CUDA blocks in its
    # pool indefinitely otherwise.
    del model, rep, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
