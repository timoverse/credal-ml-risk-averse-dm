"""Training routines for each method, plus the shared CE training loop and metric helpers."""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from probly.decider import categorical_from_mean
from probly.layers.torch import BayesConv2d, BayesLinear
from probly.method.efficient_credal_prediction import compute_efficient_credal_prediction_bounds
from probly.method.efficient_credal_prediction.torch import TorchEfficientCredalPredictor
from probly.metrics import expected_calibration_error
from probly.representer import representer
from probly.transformation.bayesian import collect_kl_divergence
from sqwash import SuperquantileReducer
from torch import optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from artifacts import describe_run_source, load_base_predictor, load_credal_wrapper_ensemble
from data import DATASET_NUM_CLASSES, get_train_data
from methods import Exp3Sampler, IndexedDataset
from methods.credal_rl_multinomial import fit_evidence_reference, select_whitening
from utils import BestModelTracker, EarlyStopping

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
) -> None:
    """Dispatch on cfg.method.name to the per-method routine. Trains in place."""
    train_section = cfg.method.get("train", {})
    train_kwargs = OmegaConf.to_container(train_section, resolve=True) if train_section else {}
    match cfg.method.name:
        case "base":
            train_base(model, train_loader, val_loader, cfg, device, run)
        case "credal_wrapper":
            train_credal_wrapper(model, train_loader, val_loader, cfg, device, run)
        case "credal_dro":
            train_credal_dro(model, train_loader, val_loader, cfg, device, run, train_kwargs)  # ty: ignore[invalid-argument-type]
        case "credal_bnn":
            train_credal_bnn(model, train_loader, val_loader, cfg, device, run, train_kwargs)  # ty: ignore[invalid-argument-type]
        case "credal_ensembling":
            train_credal_ensembling(model, train_loader, val_loader, cfg, device, run)
        case "credal_relative_likelihood":
            train_credal_relative_likelihood(model, train_loader, val_loader, cfg, device, run, train_kwargs)  # ty: ignore[invalid-argument-type]
        case "efficient_credal_prediction":
            train_efficient_credal_prediction(model, train_loader, val_loader, cfg, device, run, train_kwargs)  # ty: ignore[invalid-argument-type]
        case "credal_rl_multinomial":
            train_credal_rl_multinomial(model, train_loader, val_loader, cfg, device, run, train_kwargs)  # ty: ignore[invalid-argument-type]
        case "sqwash":
            train_sqwash(model, train_loader, val_loader, cfg, device, run, train_kwargs)  # ty: ignore[invalid-argument-type]
        case "adacvar":
            train_adacvar(model, train_loader, val_loader, cfg, device, run, train_kwargs)  # ty: ignore[invalid-argument-type]
        case _:
            raise ValueError(f"Unknown method: {cfg.method.name}")


def train_base(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
) -> None:
    """Train a single classifier with CE."""
    _training_loop(model, train_loader, val_loader, cfg, device, run)


def train_credal_wrapper(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
) -> None:
    """Train each ensemble member independently with CE."""
    # model is an EnsemblePredictor; iterating yields probly Predictor objects that are nn.Modules at runtime.
    for i, member in enumerate(model):  # ty: ignore[invalid-argument-type]
        _training_loop(member, train_loader, val_loader, cfg, device, run, log_prefix=f"member_{i}/")  # ty:ignore[invalid-argument-type]


def credal_dro_deltas(delta_g: float, num_members: int) -> list[float]:
    """Per-member DRO levels for CreDRO: uniform interpolation over [delta_g, 1].

    Implements Eq. 8 of Wang et al. (ICML 2026, arXiv:2602.08470); for M=5, delta_g=0.5
    this yields 0.5, 0.625, 0.75, 0.875, 1.0 (their Table 12). The released reference code
    instead interpolates over (delta_g, 1], excluding delta_g itself; we follow the paper.

    Args:
        delta_g: Global worst-case level delta_G in (0, 1]. The paper recommends [0.5, 1).
        num_members: Ensemble size M.

    Returns:
        List of M levels, delta_g first, 1.0 last (a single member gets delta_g).

    Raises:
        ValueError: delta_g outside (0, 1] or num_members < 1.
    """
    if not 0.0 < delta_g <= 1.0:
        raise ValueError(f"delta_g must be in (0, 1], got {delta_g}.")
    if num_members < 1:
        raise ValueError(f"num_members must be >= 1, got {num_members}.")
    if num_members == 1:
        return [delta_g]
    return [delta_g + (1.0 - delta_g) * i / (num_members - 1) for i in range(num_members)]


class TopFractionReducer(nn.Module):
    """Mean of the top floor(delta * B) per-instance losses (hard batch CVaR at level delta).

    The batch-wise top-delta approximation of the CVaR uncertainty set of Levy et al. (2020)
    used by CreDRO (Wang et al., ICML 2026): only the worst delta fraction of samples in each
    batch receives gradient. delta=1.0 recovers the plain mean (ERM). Plugged into
    _training_loop via loss_reducer, so validation reports the same top-delta objective on
    the val population (the sqwash convention).
    """

    def __init__(self, delta: float) -> None:
        """Store the selection fraction.

        Args:
            delta: Fraction of highest-loss samples to keep, in (0, 1].

        Raises:
            ValueError: delta outside (0, 1].
        """
        super().__init__()
        if not 0.0 < delta <= 1.0:
            raise ValueError(f"delta must be in (0, 1], got {delta}.")
        self.delta = float(delta)

    def forward(self, per_instance: torch.Tensor) -> torch.Tensor:
        """Reduce per-instance losses to the mean of the top floor(delta * B) values.

        Args:
            per_instance: Per-instance losses, shape (B,).

        Returns:
            Scalar loss tensor.
        """
        if self.delta >= 1.0:
            return per_instance.mean()
        # floor(delta * B), clamped to 1 so degenerate tiny batches still train.
        k = max(1, int(self.delta * per_instance.shape[0]))
        return per_instance.topk(k).values.mean()


def train_credal_dro(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train each ensemble member with the CreDRO top-delta CE objective.

    CreDRO (Wang et al., ICML 2026, arXiv:2602.08470): member i minimizes the mean CE of the
    top floor(delta_i * batch_size) highest-loss samples per batch, with delta_i uniformly
    interpolated over [delta_g, 1] via credal_dro_deltas. The last member (delta=1) is a
    plain ERM model. Inference-side the model is a stock credal_wrapper ensemble, so the box
    credal set of Eq. 9-10 comes from the existing ProbabilityIntervalsRepresenter.
    """
    delta_g = float(train_kwargs.get("delta_g", 0.5))
    members = list(model)  # ty: ignore[invalid-argument-type]
    deltas = credal_dro_deltas(delta_g, len(members))
    for i, (member, delta_i) in enumerate(zip(members, deltas, strict=True)):
        run.summary[f"member_{i}/delta"] = delta_i
        _training_loop(
            member,  # ty: ignore[invalid-argument-type]
            train_loader,
            val_loader,
            cfg,
            device,
            run,
            log_prefix=f"member_{i}/",
            loss_reducer=TopFractionReducer(delta_i),
        )


def train_credal_bnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train each Bayesian ensemble member with probly's ELBO (CE plus kl_penalty times layer KL).

    Members are Bayes-by-backprop networks (probly's bayesian transformation): every forward
    samples weights from the variational posterior, so the standard CE loop with the kl_penalty
    hook is exactly the ELBO of Blundell et al. that probly's ELBOLoss implements. Following the
    probly_benchmark convention, the penalty is kl_scale divided by the training-set size:
    kl_scale = 1 is the exact ELBO, kl_scale < 1 a tempered posterior.
    """
    dataset = getattr(train_loader, "dataset", None)
    dataset_size = len(dataset) if dataset is not None else len(train_loader) * cfg.batch_size
    kl_penalty = float(train_kwargs.get("kl_scale", 1.0)) / dataset_size
    run.summary["kl_penalty"] = kl_penalty
    # The benchmark's cifar10 override zeroes weight_decay for Bayesian members: the ELBO's KL
    # already regularizes toward the prior, so optimizer weight decay would double-count. Applied
    # on a config copy so nothing outside this trainer sees the change.
    if float(cfg.optimizer.get("params", {}).get("weight_decay", 0.0)) != 0.0:
        logger.info("credal_bnn: zeroing optimizer weight_decay (KL already regularizes toward the prior).")
        cfg = OmegaConf.merge(cfg, {"optimizer": {"params": {"weight_decay": 0.0}}})  # ty: ignore[invalid-assignment]
    for i, member in enumerate(model):  # ty: ignore[invalid-argument-type]
        _training_loop(
            member,  # ty: ignore[invalid-argument-type]
            train_loader,
            val_loader,
            cfg,
            device,
            run,
            log_prefix=f"member_{i}/",
            kl_penalty=kl_penalty,
        )


def train_credal_ensembling(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
) -> None:
    """Train credal_ensembling: reuse a matching credal_wrapper ensemble if it exists, else train from scratch.

    credal_ensembling and credal_wrapper share the exact same trained nn.ModuleList; they
    differ only in the representer used at inference. Reusing weights when available saves
    the full per-member CE training pass.
    """
    loaded = load_credal_wrapper_ensemble(cfg, device)
    if loaded is not None:
        wrapper_ensemble, source_run_id = loaded
        model.load_state_dict(wrapper_ensemble.state_dict())
        model.to(device)
        run.summary["source_run_id"] = source_run_id
        logger.info(
            "Loaded credal_wrapper ensemble from %s; skipping member training.", describe_run_source(source_run_id)
        )
        return
    logger.info("No credal_wrapper artifact found for this config; training credal_ensembling members from scratch.")
    for i, member in enumerate(model):  # ty: ignore[invalid-argument-type]
        _training_loop(member, train_loader, val_loader, cfg, device, run, log_prefix=f"member_{i}/")  # ty:ignore[invalid-argument-type]


def train_credal_relative_likelihood(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train CreRL: MLE first (member 0), then remaining members under RL thresholds.

    When alpha <= 0.0 only the MLE is trained -- bias-init members already span the
    simplex at initialisation (any model satisfies RL >= 0.0), so further training is
    unnecessary.
    """
    # iterating CRL yields probly Predictor objects; each is an nn.Module at runtime.
    members = list(model)  # ty: ignore[invalid-argument-type]
    alpha = float(train_kwargs.get("alpha", 0.5))
    batch_check = bool(train_kwargs.get("batch_check", False))
    n_remaining = len(members) - 1

    # Member 0 is the unbiased MLE. Reuse a matching base artifact when available so this
    # MLE is bit-identical to the standalone base / EffCre base predictor; otherwise train
    # it from scratch.
    loaded = load_base_predictor(cfg, device)
    if loaded is not None:
        base_model, source_run_id = loaded
        members[0].load_state_dict(base_model.state_dict())  # ty:ignore[unresolved-attribute]
        members[0].to(device)  # ty:ignore[unresolved-attribute]
        run.summary["base_run_id"] = source_run_id
        amp_enabled = bool(cfg.get("amp", False))
        train_metrics = validate(members[0], train_loader, device, amp_enabled)  # ty:ignore[invalid-argument-type]
        log_data: dict[str, Any] = {
            "member_0/train_loss": train_metrics["loss"],
            "member_0/relative_likelihood": 1.0,
        }
        if val_loader is not None:
            val_metrics = validate(members[0], val_loader, device, amp_enabled)  # ty:ignore[invalid-argument-type]
            log_data.update({f"member_0/val_{k}": v for k, v in val_metrics.items()})
        run.log(log_data)
        logger.info("Loaded member 0 MLE from base %s; skipping member-0 training.", describe_run_source(source_run_id))
    else:
        logger.info("No base artifact found for this config; training CreRL member 0 from scratch.")
        _training_loop(
            members[0],  # ty:ignore[invalid-argument-type]
            train_loader,
            val_loader,
            cfg,
            device,
            run,
            log_prefix="member_0/",
            extra_metrics={"member_0/relative_likelihood": 1.0},
        )

    amp_enabled = bool(cfg.get("amp", False))
    members[0].eval()  # ty:ignore[unresolved-attribute]
    max_ll = compute_log_likelihood(members[0], train_loader, device, amp_enabled)  # ty:ignore[invalid-argument-type]
    run.summary["max_ll"] = max_ll

    if alpha <= 0.0:
        logger.info("CreRL alpha=%g: bias-init members span the simplex; skipping their training.", alpha)
        return

    thresholds = torch.linspace(alpha, 1.0, n_remaining + 1)[:-1].tolist()
    logger.info("CreRL thresholds: %s", thresholds)

    for i, (member, target_rl) in enumerate(zip(members[1:], thresholds, strict=True), start=1):
        _training_loop(
            member,  # ty:ignore[invalid-argument-type]
            train_loader,
            val_loader,
            cfg,
            device,
            run,
            log_prefix=f"member_{i}/",
            rl_stop=(target_rl, max_ll, batch_check),
        )


def train_credal_rl_multinomial(
    model: nn.Module,
    train_loader: DataLoader,  # noqa: ARG001
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Reuse base artifact encoder, collect clean features, fit the whitened evidence reference, fill buffers (no SGD).

    The stored reference is the fitted model: whitener, float16 projected training features with
    their labels, the working kernel bandwidth (bandwidth_mult times the median self-excluded
    nearest neighbour distance) and alpha. The augmented train_loader is deliberately unused; the
    reference comes from a clean-transform rebuild of the same subset.

    Ends with a KERNEL-VOTE SANITY GATE on the validation split. The vote (counts / mass, the
    center every alpha-ball is drawn around) can silently collapse while the underlying encoder
    stays healthy -- on covid_xray it fit to 23% vote accuracy on seed 2 (voting one class on 93%
    of instances) with the very same encoder classifying at 92%, and every downstream credal
    figure averaged over that broken decision maker for days. The failure is invisible to any
    later stage: alpha only scales the ball radius around the vote, so no evaluation-side check
    can distinguish "conservative set" from "wrong center". Hence checked HERE, the only stage
    that can, and loudly: an accuracy this low is a broken fit, not a risk-averse one.
    """
    loaded = load_base_predictor(cfg, device)
    if loaded is None:
        raise RuntimeError(
            "credal_rl_multinomial requires a matching base artifact; "
            "train one first via src/training/train.py method=base recipe=<recipe>."
        )
    base, base_run_id = loaded
    run.summary["base_run_id"] = base_run_id
    logger.info("Loaded base encoder from wandb run %s", base_run_id)

    # strict=False drops base's linear.* keys (encoder.linear is nn.Identity).
    model.encoder.load_state_dict(base.state_dict(), strict=False)  # ty: ignore[unresolved-attribute]
    model.encoder.to(device).eval()  # ty: ignore[unresolved-attribute]

    # The reference describes the deployment-time feature density, so it is fitted on the same
    # training subset without augmentation (train_transforms=False reproduces the identical split
    # indices); rewrapped unshuffled so the stored reference order is run-reproducible.
    shuffled_clean, _, _ = get_train_data(
        cfg.dataset,
        val_split=cfg.val_split,
        num_train=cfg.num_train,
        seed=cfg.seed,
        train_transforms=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    clean_loader = DataLoader(
        shuffled_clean.dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False
    )
    train_features, train_targets = collect_encoder_features(
        model.encoder,  # ty: ignore[invalid-argument-type]
        clean_loader,
        device,
        bool(cfg.amp),
    )

    method_params = cfg.method.get("params", {})
    ridge = float(method_params.get("ridge", 1e-2))
    bandwidth_mult = float(method_params.get("bandwidth_mult", 0.55))
    num_neighbors = int(method_params.get("num_neighbors", 200))
    whitening = method_params.get("whitening", "auto")
    # Which self-excluded neighbour rank sets the bandwidth scale. The model resolved the string
    # form ("neighborhood" -> num_neighbors) in its constructor, so read it back from there rather
    # than re-parsing the config and risking the two disagreeing.
    bandwidth_anchor = int(getattr(model, "bandwidth_anchor", 1))
    num_classes = DATASET_NUM_CLASSES[cfg.dataset]
    if whitening == "auto":
        gamma, grid_stats = select_whitening(
            train_features, train_targets, ridge, num_classes, bandwidth_mult, num_neighbors, bandwidth_anchor
        )
        for grid_gamma, (grid_acc, grid_share) in grid_stats.items():
            run.summary[f"whitening_grid/acc_gamma{grid_gamma:g}"] = grid_acc
            run.summary[f"whitening_grid/share_gamma{grid_gamma:g}"] = grid_share
        logger.info(
            "Auto whitening chose gamma=%g; grid %s",
            gamma,
            {f"{g:g}": f"acc {a:.3f} share {s:.2f}" for g, (a, s) in grid_stats.items()},
        )
    else:
        gamma = float(whitening)
    whitener, reference, bandwidth_base = fit_evidence_reference(
        train_features, train_targets, ridge, num_classes, whitening=gamma, bandwidth_anchor=bandwidth_anchor
    )
    model.whitener = whitener.to(device)
    model.bandwidth = (bandwidth_mult * bandwidth_base).to(device)
    model.alpha = torch.tensor(float(train_kwargs["alpha"]), device=device)
    model.whitening_gamma = torch.tensor(gamma, device=device)
    run.summary["whitening_gamma"] = gamma

    num_landmarks = train_kwargs.get("reference_landmarks")
    if num_landmarks is not None:
        # Compressed reference: k-means landmarks in the whitened space, each carrying the class
        # histogram of its cluster. Removes the raw training features from the artifact at a
        # measured cost of about 0.003 size-AUROC versus the full reference.
        from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415  only the compressed path needs it

        kmeans = MiniBatchKMeans(
            n_clusters=int(num_landmarks), batch_size=4096, n_init=3, max_iter=60, random_state=int(cfg.seed)
        )
        assignments = torch.from_numpy(kmeans.fit_predict(reference.cpu().numpy())).long()
        landmarks = torch.from_numpy(kmeans.cluster_centers_).float()
        masses = torch.zeros(int(num_landmarks), num_classes, dtype=torch.float32)
        masses.index_put_((assignments, train_targets.long().cpu()), torch.ones(len(assignments)), accumulate=True)
        model.reference = landmarks.half().to(device)
        model.reference_class_masses = masses.to(device)
        n_reference = int(num_landmarks)
    else:
        model.reference = reference.half().to(device)
        model.reference_targets = train_targets.long().to(device)
        n_reference = int(reference.shape[0])

    logger.info(
        "Fitted evidence reference (%d points) on %s clean features; bandwidth=%.4f "
        "(base %.4f at neighbour rank %d x %.2f), alpha=%.4f",
        n_reference,
        tuple(train_features.shape),
        float(model.bandwidth),
        float(bandwidth_base),
        bandwidth_anchor,
        bandwidth_mult,
        float(train_kwargs["alpha"]),
    )
    run.summary["n_reference"] = n_reference
    run.summary["bandwidth"] = float(model.bandwidth)
    run.summary["bandwidth_base"] = float(bandwidth_base)
    # Logged so a fit's scale can be read back without re-deriving it: with the anchor at 1 this is
    # the nearest-neighbour spacing, with the anchor at num_neighbors it is the radius the kernel
    # actually integrates over, and the ratio between them is the diagnostic for a starved reference.
    run.summary["bandwidth_anchor"] = bandwidth_anchor
    run.summary["neighbors_per_reference"] = num_neighbors / max(n_reference, 1)

    if val_loader is not None:
        val_features, val_targets = collect_encoder_features(
            model.encoder,  # ty: ignore[invalid-argument-type]
            val_loader,
            device,
            bool(cfg.amp),
        )
        counts, mass = model.evidence(val_features.to(device))  # ty: ignore[call-non-callable]
        # Both onto the CPU before comparing: counts is on the inference device while
        # collect_encoder_features may return targets on either, so pin the comparison to one side.
        votes = counts.argmax(dim=1).cpu()
        val_targets = val_targets.long().cpu()
        vote_acc = float((votes == val_targets).double().mean())
        # The vote's class-frequency profile, because the observed collapse mode is a single class
        # absorbing the vote: max_share near 1 identifies it even when accuracy alone is ambiguous
        # (a majority-class dataset can have a mediocre-but-honest vote at the same accuracy).
        max_share = float(votes.bincount(minlength=int(counts.shape[1])).max()) / max(int(votes.numel()), 1)
        run.summary["vote_val_acc"] = vote_acc
        run.summary["vote_val_max_class_share"] = max_share
        logger.info("Kernel-vote validation accuracy %.4f; largest single-class vote share %.4f", vote_acc, max_share)
        if vote_acc < 0.5 or max_share > 0.9:
            logger.warning(
                "KERNEL VOTE LOOKS COLLAPSED: validation vote accuracy %.4f, largest single-class "
                "vote share %.4f. The credal set's CENTER is broken for this seed -- alpha cannot "
                "repair it, and every evaluation built on this artifact will average in a wrong "
                "decision maker. Investigate the whitened-feature geometry (ridge, bandwidth, "
                "fp16 reference storage) before using this artifact.",
                vote_acc,
                max_share,
            )


def train_sqwash(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a plain logit classifier with sqwash's SuperquantileReducer on per-instance CE.

    Per-batch losses are reduced by the empirical CVaR (mean of the worst alpha-fraction of
    losses in the minibatch) instead of the standard mean. alpha=1.0 recovers ERM; smaller
    alpha targets the tail more aggressively.

    See Laguel, Pillutla, Malick, Harchaoui, "Superquantiles at Work" (2021) for the
    underlying superquantile / CVaR formulation, and github.com/krishnap25/sqwash for the
    reference implementation.
    """
    alpha = float(train_kwargs.get("alpha", 0.5))
    reducer = SuperquantileReducer(superquantile_tail_fraction=alpha)
    run.summary["sqwash_alpha"] = alpha
    _training_loop(model, train_loader, val_loader, cfg, device, run, loss_reducer=reducer)


def train_adacvar(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train a logit classifier with mean CE on minibatches from an Exp3 sampler.

    Implements the paper-named adacvar variant from Curi, Levy, Jegelka, Krause,
    Adaptive Sampling for Stochastic Risk-Averse Learning (NeurIPS 2020). Per training
    step the sampler draws k = ceil(alpha * N) indices without replacement weighted by
    k-DPP marginals, then batch_size indices uniformly with replacement from that
    subset. The model is updated by mean CE on the minibatch and the sampler is updated
    by exponentiated gradient on the observed losses. See github.com/sebascuri/adacvar
    for the reference implementation.

    Args:
        model: Logit classifier to train.
        train_loader: Standard DataLoader yielding (x, y). Its dataset attribute is
            wrapped in an IndexedDataset to expose positions to the sampler.
        val_loader: Optional validation DataLoader, used unmodified.
        cfg: Hydra DictConfig with epochs, batch_size, optimizer, scheduler, AMP, etc.
        device: Inference and training device.
        run: wandb run-like object exposing log and summary.
        train_kwargs: Parsed method.train section. Required key alpha (float).
    """
    alpha = float(train_kwargs["alpha"])
    params_section = cfg.method.get("params") or {}
    params = OmegaConf.to_container(params_section, resolve=True) if params_section else {}
    eta_override = params.get("eta") if isinstance(params, dict) else None
    gamma = float(params.get("gamma", 0.0)) if isinstance(params, dict) else 0.0

    indexed = IndexedDataset(train_loader.dataset)
    n = len(indexed)
    horizon = max(1, cfg.epochs * (n // cfg.batch_size))
    if eta_override is None:
        eta = math.sqrt((1.0 / alpha) * math.log(1.0 / alpha) / horizon) if alpha < 1.0 else 0.0
    else:
        eta = float(eta_override)
    sampler = Exp3Sampler(
        num_actions=n,
        batch_size=cfg.batch_size,
        alpha=alpha,
        eta=eta,
        gamma=gamma,
    )
    # Validate on empirical CVaR_alpha of val per-instance losses, matching the
    # training objective. BestModelTracker / early stopping then select on the same
    # metric the training loop optimises. Mirrors train_sqwash. alpha=1 collapses
    # the reducer to the mean (SuperquantileReducer(1.0) = mean), which is the
    # correct degenerate behaviour when the sampler is uniform.
    val_reducer = SuperquantileReducer(superquantile_tail_fraction=alpha)

    loader_kwargs: dict[str, Any] = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
    }
    indexed_loader = DataLoader(indexed, batch_sampler=sampler, **loader_kwargs)

    optimizer = _get_optimizer(cfg.optimizer.name, model.parameters(), **cfg.optimizer.get("params", {}))
    scheduler = _get_scheduler(
        cfg.scheduler.get("name") if cfg.scheduler else None,
        optimizer,
        cfg.epochs,
        **cfg.scheduler.get("params", {}) if cfg.scheduler else {},
    )

    grad_clip_norm = cfg.get("grad_clip_norm")
    amp_enabled = bool(cfg.get("amp", False))
    scaler = GradScaler(device.type) if amp_enabled else None

    early_stop_cfg = cfg.get("early_stopping", {})
    patience = early_stop_cfg.get("patience", 0)
    min_delta = early_stop_cfg.get("min_delta", 0.0)
    early_stopper = EarlyStopping(patience, min_delta) if patience else None
    best_tracker = BestModelTracker(min_delta) if val_loader is not None else None

    run.summary["adacvar_eta"] = eta
    run.summary["adacvar_k"] = sampler.k

    for epoch in tqdm(range(cfg.epochs), desc="Epoch"):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for inputs_, targets_, idx_ in indexed_loader:
            inputs = inputs_.to(device, non_blocking=True)
            targets = targets_.to(device, non_blocking=True)
            if device.type == "cuda" and inputs.ndim >= 4:
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                if scaler is None:
                    raise ValueError("scaler must be provided when amp_enabled=True")
                with torch.amp.autocast(device.type):
                    logits = model(inputs)  # (B, C)
                    per_inst = F.cross_entropy(logits, targets, reduction="none")
                    loss = per_inst.mean()
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)  # (B, C)
                per_inst = F.cross_entropy(logits, targets, reduction="none")
                loss = per_inst.mean()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            sampler.update(
                per_inst.detach().float().cpu().numpy(),
                idx_.detach().cpu().numpy(),
            )
            running_loss += float(loss.item())
            n_batches += 1
        running_loss /= max(n_batches, 1)
        sampler.normalize()

        log_data: dict[str, Any] = {"train_loss": running_loss}
        val_loss: float | None = None
        if val_loader is not None:
            metrics = validate(model, val_loader, device, amp_enabled, loss_reducer=val_reducer)
            log_data.update({f"val_{k}": v for k, v in metrics.items()})
            val_loss = metrics["loss"]
        run.log(data=log_data)

        if best_tracker is not None and val_loss is not None:
            best_tracker.update(val_loss, model)
        if scheduler is not None:
            scheduler.step()
        if early_stopper is not None and val_loss is not None and early_stopper.should_stop(val_loss):
            run.summary["early_stopped"] = True
            logger.info("[adacvar] early stopping at epoch %d", epoch)
            break

    if best_tracker is not None:
        best_tracker.restore(model)
        if best_tracker.best_state_dict is not None:
            run.summary["best_val_loss"] = best_tracker.best_loss


def train_efficient_credal_prediction(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    train_kwargs: dict[str, Any],
) -> None:
    """Train EffCre: reuse a matching base artifact if it exists, else train from scratch. Then compute logit bounds."""
    # model.predictor is typed as probly's Predictor protocol; it's an nn.Module at runtime.
    predictor = model.predictor
    loaded = load_base_predictor(cfg, device)
    if loaded is not None:
        base_model, source_run_id = loaded
        predictor.load_state_dict(base_model.state_dict())  # ty: ignore[unresolved-attribute]
        predictor.to(device)
        run.summary["base_run_id"] = source_run_id
        logger.info("Loaded base predictor from %s; skipping base training.", describe_run_source(source_run_id))
    else:
        logger.info("No base artifact found for this config; training base predictor from scratch.")
        _training_loop(predictor, train_loader, val_loader, cfg, device, run)  # ty: ignore[invalid-argument-type]

    amp_enabled = bool(cfg.get("amp", False))
    alpha = float(train_kwargs.get("alpha", 0.5))
    num_classes = DATASET_NUM_CLASSES[cfg.dataset]

    logits_train, targets_train = collect_logits_targets(predictor, train_loader, device, amp_enabled)  # ty: ignore[invalid-argument-type]
    # probly's bounds func is generic over ArrayLike; torch.Tensor satisfies it at runtime.
    lower, upper = compute_efficient_credal_prediction_bounds(
        logits_train,  # ty: ignore[invalid-argument-type]
        targets_train,  # ty: ignore[invalid-argument-type]
        num_classes=num_classes,
        alpha=alpha,
    )
    lower_t = lower.to(device)
    upper_t = upper.to(device)
    if model.lower is None:
        model.lower = lower_t
        model.upper = upper_t
    else:
        # model.lower/upper are typed as ArrayLike per probly; here they're torch.Tensor.
        model.lower.copy_(lower_t)  # ty: ignore[call-non-callable]
        model.upper.copy_(upper_t)  # ty: ignore[call-non-callable]
    run.summary["efficient_credal_alpha"] = alpha


def _training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    device: torch.device,
    run: Any,  # noqa: ANN401
    *,
    log_prefix: str = "",
    extra_metrics: dict[str, float] | None = None,
    rl_stop: tuple[float, float, bool] | None = None,
    loss_reducer: nn.Module | None = None,
    kl_penalty: float | None = None,
) -> float | None:
    """Standard CE loop with optional relative-likelihood early-stop.

    rl_stop=(target_rl, max_ll, batch_check) breaks training as soon as the model's RL
    against max_ll crosses target_rl (checked per batch if batch_check, else per epoch).
    Returns the final RL when rl_stop is set, else None.

    loss_reducer, if provided, replaces the default mean reduction over the per-instance CE
    losses in each minibatch (used by sqwash to swap in CVaR aggregation). kl_penalty, if
    provided, turns the CE step into probly's ELBO step (used by credal_bnn members).
    """
    optimizer = _get_optimizer(cfg.optimizer.name, model.parameters(), **cfg.optimizer.get("params", {}))
    scheduler = _get_scheduler(
        cfg.scheduler.get("name") if cfg.scheduler else None,
        optimizer,
        cfg.epochs,
        **cfg.scheduler.get("params", {}) if cfg.scheduler else {},
    )

    grad_clip_norm = cfg.get("grad_clip_norm")
    amp_enabled = bool(cfg.get("amp", False))
    scaler = GradScaler(device.type) if amp_enabled else None

    early_stop_cfg = cfg.get("early_stopping", {})
    patience = early_stop_cfg.get("patience", 0)
    min_delta = early_stop_cfg.get("min_delta", 0.0)
    early_stopper = EarlyStopping(patience, min_delta) if patience else None
    best_tracker = BestModelTracker(min_delta) if val_loader is not None else None

    if log_prefix:
        epoch_key = f"{log_prefix.rstrip('/')}_epoch"
        wandb.define_metric(epoch_key, hidden=True)
        wandb.define_metric(f"{log_prefix}*", step_metric=epoch_key)
    else:
        epoch_key = None

    final_rl: float | None = None
    rl_reached = False

    for epoch in tqdm(range(cfg.epochs), desc=f"{log_prefix}Epoch"):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for inputs_, targets_ in train_loader:
            inputs = inputs_.to(device, non_blocking=True)
            targets = targets_.to(device, non_blocking=True)
            if device.type == "cuda" and inputs.ndim >= 4:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            running_loss += train_epoch_ce(
                model,
                inputs,
                targets,
                optimizer,
                grad_clip_norm=grad_clip_norm,
                amp_enabled=amp_enabled,
                scaler=scaler,
                loss_reducer=loss_reducer,
                kl_penalty=kl_penalty,
            )
            n_batches += 1

            if rl_stop is not None and rl_stop[2]:
                target_rl, max_ll, _ = rl_stop
                final_rl = _current_rl(model, train_loader, device, amp_enabled, max_ll)
                if final_rl >= target_rl:
                    rl_reached = True
                    break
        running_loss /= max(n_batches, 1)

        if rl_stop is not None and not rl_reached:
            target_rl, max_ll, _ = rl_stop
            final_rl = _current_rl(model, train_loader, device, amp_enabled, max_ll)
            if final_rl >= target_rl:
                rl_reached = True

        log_data: dict[str, Any] = {f"{log_prefix}train_loss": running_loss}
        if epoch_key is not None:
            log_data[epoch_key] = epoch
        if extra_metrics:
            log_data.update(extra_metrics)
        if final_rl is not None:
            log_data[f"{log_prefix}relative_likelihood"] = final_rl

        val_loss: float | None = None
        if val_loader is not None:
            metrics = validate(model, val_loader, device, amp_enabled, loss_reducer=loss_reducer)
            log_data.update({f"{log_prefix}val_{k}": v for k, v in metrics.items()})
            val_loss = metrics["loss"]
        run.log(data=log_data)

        if best_tracker is not None and val_loss is not None:
            best_tracker.update(val_loss, model)

        if scheduler is not None:
            scheduler.step()

        if rl_reached:
            run.summary[f"{log_prefix}rl_stopped"] = True
            run.summary[f"{log_prefix}stopped_epoch"] = epoch
            logger.info("[%s] hit RL target (RL=%.4f) at epoch %d", log_prefix, final_rl, epoch)
            break

        if early_stopper is not None and val_loss is not None and early_stopper.should_stop(val_loss):
            run.summary[f"{log_prefix}early_stopped"] = True
            logger.info("[%s] early stopping at epoch %d", log_prefix, epoch)
            break

    if best_tracker is not None:
        best_tracker.restore(model)
        if best_tracker.best_state_dict is not None:
            run.summary[f"{log_prefix}best_val_loss"] = best_tracker.best_loss

    return final_rl


@torch.no_grad()
def _current_rl(
    model: nn.Module, train_loader: DataLoader, device: torch.device, amp_enabled: bool, max_ll: float
) -> float:
    """exp(current_ll - max_ll) on the training set."""
    current_ll = compute_log_likelihood(model, train_loader, device, amp_enabled)
    return float(torch.exp(torch.tensor(current_ll - max_ll)).item())


_OPTIMIZERS = {"sgd": optim.SGD, "adam": optim.Adam, "adamw": optim.AdamW}


def _get_optimizer(name: str, params: Any, **kwargs: Any) -> optim.Optimizer:  # noqa: ANN401
    """Look up the optimizer class by name and instantiate it."""
    key = name.lower()
    if key not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    return _OPTIMIZERS[key](params, **kwargs)


def _get_scheduler(
    name: str | None,
    optimizer: optim.Optimizer,
    epochs: int,
    **kwargs: Any,
) -> optim.lr_scheduler.LRScheduler | None:
    """Look up the LR scheduler by name. Supports 'cosine' and 'multistep' (add more as needed)."""
    if name is None or str(name).lower() == "none":
        return None
    key = name.lower()
    if key == "cosine":
        kwargs.setdefault("T_max", epochs)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if key == "multistep":
        return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    raise ValueError(f"Unknown scheduler: {name}")


def train_epoch_ce(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    grad_clip_norm: float | None = None,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    loss_reducer: nn.Module | None = None,
    kl_penalty: float | None = None,
) -> float:
    """One CE training step. Returns the loss value.

    When loss_reducer is None the loss is the standard mean cross-entropy. Otherwise
    loss_reducer is applied to the per-instance CE losses (used by sqwash to aggregate
    via SuperquantileReducer / CVaR instead of the mean). kl_penalty, if set, adds
    kl_penalty times the summed KL divergence of the model's Bayesian layers (probly's
    ELBO of Blundell et al.; used by credal_bnn members, whose forward samples weights).
    """
    optimizer.zero_grad(set_to_none=True)
    if amp_enabled:
        if scaler is None:
            raise ValueError("scaler must be provided when amp_enabled=True")
        with torch.amp.autocast(inputs.device.type):
            logits = model(inputs)  # (B, C)
            loss = _ce_loss(logits, targets, loss_reducer)
            if kl_penalty is not None:
                loss = loss + kl_penalty * collect_kl_divergence(model)  # ty: ignore[unsupported-operator]
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model(inputs)  # (B, C)
        loss = _ce_loss(logits, targets, loss_reducer)
        if kl_penalty is not None:
            loss = loss + kl_penalty * collect_kl_divergence(model)  # ty: ignore[unsupported-operator]
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
    return loss.item()


def _reduce(per_instance: torch.Tensor, loss_reducer: nn.Module | None) -> torch.Tensor:
    """Apply loss_reducer to per-instance losses, or fall back to the mean."""
    return per_instance.mean() if loss_reducer is None else loss_reducer(per_instance)


def _ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_reducer: nn.Module | None,
) -> torch.Tensor:
    """Cross-entropy reduced by mean, or by loss_reducer applied to per-instance losses."""
    return _reduce(F.cross_entropy(logits, targets, reduction="none"), loss_reducer)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    loss_reducer: nn.Module | None = None,
) -> dict[str, float]:
    """Validation loss and top-1 accuracy.

    When loss_reducer is None the reported loss is the mean per-instance CE on the
    full val set. Otherwise loss_reducer is applied once to the concatenation of all
    per-instance CE losses (the population reduction, not a per-batch average), so that
    BestModelTracker selects on the same objective the training loop optimizes.
    """
    model.eval()
    per_instance_chunks: list[torch.Tensor] = []
    total_correct = 0
    total_n = 0
    for inputs_, targets_ in val_loader:
        inputs = inputs_.to(device, non_blocking=True)
        targets = targets_.to(device, non_blocking=True)
        if device.type == "cuda" and inputs.ndim >= 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            logits = model(inputs)  # (B, C)
            losses = F.cross_entropy(logits, targets, reduction="none")  # (B,)
        per_instance_chunks.append(losses)
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()
        total_n += targets.size(0)
    all_losses = torch.cat(per_instance_chunks)  # (N,)
    loss = _reduce(all_losses, loss_reducer)
    return {"loss": float(loss.item()), "acc": total_correct / total_n}


@torch.no_grad()
def extract_point_predictor(model: nn.Module | list[nn.Module]) -> nn.Module | None:
    """The method's regular point predictor, or None when the representation barycenter is it.

    Shared by evaluate() and shift_set_size's accuracy overlay so the two accuracies keep the
    same semantics. Member lists use their first (MLE) member -- except Bayesian ensembles, whose
    single members have no argument-free representer and whose regular prediction is the hull
    mean anyway; predictors exposing mle_member use it (credal_rl_multinomial's vote);
    efficient_credal_prediction unwraps to its base classifier.
    None means: represent the full model and score categorical_from_mean of the representation
    (the wrapper's envelope barycenter, the ensembling and credal_bnn hull means).

    Args:
        model: Trained predictor (nn.Module, or a Python list of members).

    Returns:
        The point predictor module, or None when the full model's representation is the point.
    """
    if isinstance(model, list):
        first: nn.Module = model[0]  # ty: ignore[invalid-assignment]
        if any(isinstance(module, BayesLinear | BayesConv2d) for module in first.modules()):
            return None
        point: nn.Module = first
    else:
        point = model
    point = getattr(point, "mle_member", point)
    if isinstance(point, TorchEfficientCredalPredictor):
        point = point.predictor
    return None if point is model else point


@torch.no_grad()
def evaluate(
    model: nn.Module | list[nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    num_bins: int = 15,
) -> dict[str, float]:
    """Test accuracy, NLL, and ECE on the predictor's point distribution.

    CreRL's top-level object is a Python list of members; for it we take the first
    (unbiased MLE) member. All other methods are nn.Module instances and use the full
    model. Either way we go through representer + categorical_from_mean:
      - base: softmax of logits.
      - credal_wrapper: intersection probability of the credal set (the barycenter
        of the ProbabilityIntervalsCredalSet is intersection_probability(lower, upper)).
      - credal_relative_likelihood: MLE = first member's softmax.
      - efficient_credal_prediction: the wrapped base classifier's softmax, exactly; the
        box barycenter would only approximate it and can flip near-tied argmaxes.

    Args:
        model: Trained predictor (nn.Module, or a Python list of members for CreRL).
        test_loader: Test DataLoader.
        device: Inference device.
        amp_enabled: If True, run forward in autocast.
        num_bins: Number of bins for ECE.

    Returns:
        Dict with keys test_acc, test_nll, test_ece.
    """
    # Credal predictors may expose a cheap MLE point predictor (e.g. credal_rl_multinomial); using it
    # avoids triggering expensive credal machinery for sanity-check metrics. ECP unwraps to its
    # base classifier (exact base accuracy). None (wrapper, ensembling, credal_bnn) scores the
    # full model's representation barycenter.
    point = extract_point_predictor(model)
    if point is None:
        members = model if isinstance(model, list) else [model]
        for member in members:
            member.eval()  # ty: ignore[unresolved-attribute]
        rep = representer(model)
    else:
        point.eval()
        rep = representer(point)

    probs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    for inputs_, targets_ in test_loader:
        inputs = inputs_.to(device, non_blocking=True)
        if device.type == "cuda" and inputs.ndim >= 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            rep_out = rep.predict(inputs)
        probs = categorical_from_mean(rep_out).probabilities
        probs_list.append(probs.detach().float().cpu())  # ty: ignore[unresolved-attribute]
        targets_list.append(targets_.detach().cpu())
    metrics = _compute_metrics(torch.cat(probs_list), torch.cat(targets_list), num_bins=num_bins)
    return {f"test_{k}": v for k, v in metrics.items()}


def _accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy of probs against integer labels."""
    return (probs.argmax(dim=-1) == labels).float().mean().item()


def _compute_metrics(probs: torch.Tensor, labels: torch.Tensor, num_bins: int) -> dict[str, float]:
    """Compute accuracy, NLL, and ECE from probabilities and labels."""
    accuracy = _accuracy(probs, labels)
    eps = torch.finfo(probs.dtype).eps
    logprobs = torch.log(probs.clamp(min=eps))
    nll = F.nll_loss(logprobs, labels).item()
    ece = expected_calibration_error(probs, labels, num_bins=num_bins).item()  # ty: ignore[unresolved-attribute]
    return {"acc": accuracy, "nll": nll, "ece": ece}


@torch.no_grad()
def compute_log_likelihood(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> float:
    """Mean per-sample log-likelihood of model on loader. Used by CRL for the MLL reference."""
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for inputs_, targets_ in loader:
        inputs = inputs_.to(device, non_blocking=True)
        targets = targets_.to(device, non_blocking=True)
        if device.type == "cuda" and inputs.ndim >= 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            logits = model(inputs)  # (B, C)
            log_probs = F.log_softmax(logits.float(), dim=-1)
        total += log_probs[torch.arange(targets.size(0), device=device), targets].sum().item()
        count += targets.size(0)
    if was_training:
        model.train()
    return total / count


@torch.no_grad()
def collect_logits_targets(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack all logits and targets over loader. Used by EffCre for the bounds computation."""
    was_training = model.training
    model.eval()
    logits_chunks: list[torch.Tensor] = []
    targets_chunks: list[torch.Tensor] = []
    for inputs_, targets_ in tqdm(loader, desc="Collecting logits"):
        inputs = inputs_.to(device, non_blocking=True)
        targets = targets_.to(device, non_blocking=True)
        if device.type == "cuda" and inputs.ndim >= 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            logits = model(inputs).float()  # (B, C)
        logits_chunks.append(logits)
        targets_chunks.append(targets)
    if was_training:
        model.train()
    return torch.cat(logits_chunks, dim=0), torch.cat(targets_chunks, dim=0)


@torch.no_grad()
def collect_encoder_features(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack encoder features and integer targets over loader. Used by credal_rl_multinomial."""
    was_training = encoder.training
    encoder.eval()
    feature_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for inputs_, targets_ in tqdm(loader, desc="Caching encoder features"):
        inputs = inputs_.to(device, non_blocking=True)
        targets = targets_.to(device, non_blocking=True)
        if device.type == "cuda" and inputs.ndim >= 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            features = encoder(inputs).float()  # (B, D)
        feature_chunks.append(features)
        target_chunks.append(targets)
    if was_training:
        encoder.train()
    return torch.cat(feature_chunks, dim=0), torch.cat(target_chunks, dim=0).long()
