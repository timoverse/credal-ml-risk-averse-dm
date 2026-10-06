"""Out-of-distribution detection: load trained artifact, score id vs ood test sets, log metrics + raw scores to wandb.

Run: python src/experiments/ood_detection.py method=credal_wrapper recipe=cifar10_resnet18
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import tempfile
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import numpy.typing as npt
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from probly.evaluation.ood import evaluate_ood
from probly.quantification import quantify
from probly.representation.credal_set.torch import TorchConvexCredalSet, TorchProbabilityIntervalsCredalSet
from probly.representer import representer
from tqdm import tqdm

import data
import utils
from artifacts import load_model_for_evaluation, resolve_artifact_name
from methods.credal_rl_multinomial import CredalRLMultinomialPredictor

if TYPE_CHECKING:
    from probly.representer import Representer
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def _id_scores_artifact_name(cfg: DictConfig) -> str:
    """Build the wandb artifact name for the cached id_scores .npy.

    Excludes ood_dataset so the same id_scores can be reused across OOD sweeps.
    """
    return f"id_scores_{resolve_artifact_name(cfg)}_{cfg.decomposition}_{cfg.component}"


def _ood_scores_artifact_name(cfg: DictConfig) -> str:
    """Build the wandb artifact name for the ood_scores .npy (always logged)."""
    return f"ood_scores_{resolve_artifact_name(cfg)}_{cfg.ood_dataset}_{cfg.decomposition}_{cfg.component}"


def _per_instance_set_size(rep_out: Any) -> torch.Tensor:  # noqa: ANN401
    """Mean credal interval width per instance, the set-extent (nonspecificity) score.

    Probability-interval sets use their bounds directly; convex (vertex) sets use their per-class
    interval hull. Like the entropy components, the score is increasing in uncertainty.

    Args:
        rep_out: Output of rep.predict(...).

    Returns:
        Per-instance mean interval width, shape (B,).

    Raises:
        TypeError: rep_out is not a credal set, so set size is undefined (e.g. base).
    """
    if isinstance(rep_out, TorchProbabilityIntervalsCredalSet):
        return (rep_out.upper_bounds - rep_out.lower_bounds).mean(dim=-1)
    if isinstance(rep_out, TorchConvexCredalSet):
        return (rep_out.upper() - rep_out.lower()).mean(dim=-1)
    msg = f"component=size needs a credal representer but rep.predict produced {type(rep_out).__name__}."
    raise TypeError(msg)


# Probability floor applied to convex-set vertices before their entropy is optimised. Well below
# any probability that carries information (1e-12 shifts the entropy by ~1e-11 nats) and far above
# the float32 denormal range where the entropy gradient explodes.
_CONVEX_VERTEX_FLOOR = 1e-12


def _stabilize_convex_credal_set(rep_out: Any) -> Any:  # noqa: ANN401
    """Clamp a convex credal set's vertices off zero and cast to float64; pass others through.

    probly maximises the entropy of a convex credal set with L-BFGS over an unconstrained softmax
    parameterisation of the mixture weights (torch_convex_upper_entropy). The entropy gradient
    carries log(p) terms, so a vertex probability at or near zero makes the gradient enormous and
    the strong-Wolfe bracketing phase extrapolates without bound: max_step = t * 10 per iteration
    with up to 1.25 * _LBFGS_ITERS = 160 evaluations. The trial step reaches ~1e40 and
    p.add_(update, alpha=step_size) raises "value cannot be converted to type float without
    overflow" against float32's 3.4e38 ceiling.

    A float32 softmax over confident logits produces exactly that: measured in
    scratch/diagnose_float64_nan.py, the far-OOD-like case had 20 exact zeros per 10240 entries
    and a smallest nonzero probability of 8.4e-42, deep in the denormal range. This is why the
    failure tracks confident, mutually agreeing ensemble members (far-OOD) rather than degenerate
    hulls, which scratch/repro_convex_entropy_overflow.py showed are handled fine.

    The clamp is the fix and is sufficient on its own in either dtype. float64 is kept as
    headroom: on the same case, promoting to float64 WITHOUT clamping stopped the exception but
    left 2 of 128 instances at NaN, which would have been a silent corruption rather than a
    crash. Clamped, both dtypes agree on the finite range to four decimals.

    Only credal_ensembling and credal_bnn produce convex sets; every other method here yields a
    ProbabilityIntervalsCredalSet whose upper entropy is closed-form water-filling with no
    optimiser, so this returns those unchanged and their scores are bitwise unaffected.

    Args:
        rep_out: Output of rep.predict(...).

    Returns:
        A stabilised copy for convex sets, or rep_out itself when it is not convex.
    """
    if isinstance(rep_out, TorchConvexCredalSet):
        probabilities = rep_out.tensor.probabilities.double().clamp_min(_CONVEX_VERTEX_FLOOR)
        return TorchConvexCredalSet(probabilities / probabilities.sum(dim=-1, keepdim=True))
    return rep_out


@torch.no_grad()
def _collect_uncertainties(
    rep: Representer,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    component: str,
    predictor: Any = None,  # noqa: ANN401
) -> npt.NDArray[np.floating]:
    """Single inference pass; return per-instance uncertainty scores.

    Args:
        rep: representer(model).
        loader: id_loader or ood_loader.
        device: inference device.
        amp_enabled: wrap forward in autocast.
        component: 'total' / 'aleatoric' / 'epistemic' score via the credal entropy decomposition;
            'size' via the mean credal interval width (any credal method); 'leverage' via
            credal_rl_multinomial's alpha-free epistemic statistic, the negated evidence mass.
        predictor: the model, required for component='leverage' (its encoder and fitted buffers
            are used directly, bypassing the representer).

    Returns:
        1-D float32 numpy array of length len(loader.dataset).

    Raises:
        KeyError: component not in the representation's decomposition (e.g. asking
            for 'epistemic' on a non-credal base predictor).
    """
    chunks: list[torch.Tensor] = []
    for inputs_, _ in tqdm(loader, desc="Inference"):
        inputs = inputs_.to(device, non_blocking=True)
        if device.type == "cuda" and inputs.ndim >= 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            if component == "leverage":
                features = predictor.encoder(inputs).detach()
                # The multinomial method's alpha-free epistemic statistic is the evidence
                # mass; low mass means high novelty, so the score is its negative.
                _, mass = predictor.evidence(features)
                unc = -mass
            elif component == "size":
                unc = _per_instance_set_size(rep.predict(inputs))
            else:
                # float64 only for convex sets, whose entropy goes through an L-BFGS that
                # overflows float32 on confident vertices; a no-op for every other method.
                unc = quantify(_stabilize_convex_credal_set(rep.predict(inputs)))[component]  # ty: ignore[not-subscriptable]
        chunks.append(unc.detach().float().cpu())
    return torch.cat(chunks).numpy()


def _require_finite(scores: npt.NDArray[np.floating], label: str) -> None:
    """Raise if any score is NaN or infinite, naming where it came from.

    probly's convex-credal-set entropy solver returns NaN rather than raising when its L-BFGS
    wanders far enough, and credal_bnn's representer additionally averages 20 stochastic forward
    passes per member, so a single overflowing sample poisons that vertex. Either way a NaN score
    propagates silently: roc_auc_score returns NaN, the summary key stores NaN, and the table
    shows NaN with nothing pointing at the cause. Checked for cached id_scores too, since those
    bypass _collect_uncertainties entirely.

    Args:
        scores: Per-instance uncertainty scores.
        label: Which array this is, for the error message.

    Raises:
        RuntimeError: any entry is not finite.
    """
    n_bad = int((~np.isfinite(scores)).sum())
    if n_bad:
        raise RuntimeError(
            f"{n_bad} of {scores.size} {label} are not finite. Nothing was logged. This is the "
            "uncertainty computation diverging, not a data problem -- for a convex-credal-set "
            "method (credal_ensembling, credal_bnn) try amp=false, and inspect any already-cached "
            "arrays with scratch/inspect_cached_scores.py."
        )


def _log_array_artifact(
    run: wandb.sdk.wandb_run.Run,
    *,
    name: str,
    artifact_type: str,
    metadata: dict[str, Any],
    filename: str,
    array: npt.NDArray[np.floating],
) -> None:
    """Save array to a .npy file in a tempdir and log it as a wandb artifact.

    Args:
        run: Active wandb run to attach the artifact to.
        name: Artifact name.
        artifact_type: Artifact type label (e.g. 'id_scores', 'ood_scores').
        metadata: Metadata dict attached to the artifact.
        filename: Filename inside the artifact (e.g. 'id_scores.npy').
        array: Numpy array to persist.
    """
    art = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / filename
        np.save(path, array)
        art.add_file(str(path))
        run.log_artifact(art)


def _try_load_cached_id_scores(cfg: DictConfig, artifact_name: str) -> tuple[npt.NDArray[np.floating] | None, bool]:
    """Look up id_scores artifact in wandb; return (scores, loaded_from_cache).

    On any exception during artifact lookup or download (missing artifact, network
    error, ...) returns (None, False) so the caller falls through to recompute. A
    cache miss is never an error condition.

    Args:
        cfg: live eval cfg (uses cfg.wandb.entity/project).
        artifact_name: artifact name returned by _id_scores_artifact_name(cfg).

    Returns:
        (scores, loaded_from_cache). If loaded, scores is a 1-D float array.
    """
    if not cfg.wandb.enabled:
        return None, False
    qualname = f"{cfg.wandb.entity}/{cfg.wandb.project}/{artifact_name}:latest"
    try:
        api = wandb.Api(timeout=60)
        art = api.artifact(qualname)
        with tempfile.TemporaryDirectory() as td:
            art.download(root=td)
            scores = np.load(Path(td) / "id_scores.npy")
    except Exception as e:  # noqa: BLE001
        logger.info("No cached id_scores at %s (%s); will recompute.", qualname, e)
        return None, False
    logger.info("Loaded cached id_scores from %s (shape=%s).", qualname, scores.shape)
    return scores, True


@hydra.main(version_base=None, config_path="../../configs/", config_name="ood_detection")
def main(cfg: DictConfig) -> None:
    """Load artifact, score id vs ood, log metrics + raw arrays to the training run's wandb summary."""
    logger.info("OOD detection configuration:\n%s", OmegaConf.to_yaml(cfg))
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
    if cfg.component == "leverage" and not isinstance(model, CredalRLMultinomialPredictor):
        raise ValueError(
            f"component=leverage is defined only for credal_rl_multinomial "
            f"(got {cfg.method.name!r}); use component=size for a cross-method set-extent score."
        )
    logger.info("Loaded model trained in wandb run %s", run_id)

    id_loader, ood_loader = data.get_ood_data(
        cfg.dataset,
        cfg.ood_dataset,
        cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory=True,
    )
    rep = representer(model)

    id_art_name = _id_scores_artifact_name(cfg)
    id_scores, loaded_from_cache = _try_load_cached_id_scores(cfg, id_art_name)
    if id_scores is None:
        logger.info("Computing id_scores ...")
        id_scores = _collect_uncertainties(rep, id_loader, device, bool(cfg.amp), cfg.component, predictor=model)

    logger.info("Computing ood_scores ...")
    ood_scores = _collect_uncertainties(rep, ood_loader, device, bool(cfg.amp), cfg.component, predictor=model)

    # Before anything is logged: a NaN here would otherwise reach the summary keys and the table.
    _require_finite(id_scores, f"id_scores for {cfg.method.name}")
    _require_finite(ood_scores, f"ood_scores for {cfg.method.name} vs {cfg.ood_dataset}")

    metric_spec = cfg.get("metrics", "all")
    if isinstance(metric_spec, str):
        ood_metrics = evaluate_ood(id_scores, ood_scores, metrics=metric_spec)
    else:
        ood_metrics = evaluate_ood(id_scores, ood_scores, metrics=list(metric_spec))

    auroc = ood_metrics.get("auroc")
    logger.info(
        "OOD %s vs %s | %s/%s | metrics=%s | auroc=%s",
        cfg.dataset,
        cfg.ood_dataset,
        cfg.decomposition,
        cfg.component,
        ood_metrics,
        f"{auroc:.4f}" if auroc is not None else "n/a",
    )

    if cfg.wandb.enabled:
        run = wandb.init(id=run_id, entity=cfg.wandb.entity, project=cfg.wandb.project, resume="must")
        prefix = f"ood/{cfg.ood_dataset}/{cfg.decomposition}/{cfg.component}"
        for metric_name, value in ood_metrics.items():
            run.summary[f"{prefix}/{metric_name}"] = value
        common_meta = {
            "run_id": run_id,
            "method": cfg.method.name,
            "dataset": cfg.dataset,
            "decomposition": cfg.decomposition,
            "component": cfg.component,
            "seed": cfg.seed,
        }
        if not loaded_from_cache:
            _log_array_artifact(
                run,
                name=id_art_name,
                artifact_type="id_scores",
                metadata=common_meta,
                filename="id_scores.npy",
                array=id_scores,
            )
        _log_array_artifact(
            run,
            name=_ood_scores_artifact_name(cfg),
            artifact_type="ood_scores",
            metadata={**common_meta, "ood_dataset": cfg.ood_dataset},
            filename="ood_scores.npy",
            array=ood_scores,
        )
        run.finish()


if __name__ == "__main__":
    main()
