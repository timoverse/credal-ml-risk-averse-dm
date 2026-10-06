"""Deterministic W&B artifact name + download + model rebuild. Used by train.py and eval scripts."""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import wandb

import data
import models
from paths import CHECKPOINTS_PATH

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from omegaconf import DictConfig

# Where a checkpoint is fetched from. "wandb" downloads the :latest version of the artifact;
# "local" reads <CHECKPOINTS_PATH>/<artifact_name>.pt, the file train.py writes when
# save_to_disk is true. The two share resolve_artifact_name, so the same (method, seed) names
# the same thing either way -- that coupling is deliberate and must be preserved, since the one
# time local and remote naming drifted apart the driver silently skipped five of seven methods.
ARTIFACT_SOURCE_WANDB = "wandb"
ARTIFACT_SOURCE_LOCAL = "local"
ARTIFACT_SOURCES = (ARTIFACT_SOURCE_WANDB, ARTIFACT_SOURCE_LOCAL)


def resolve_artifact_source(cfg: DictConfig | dict[str, Any]) -> str:
    """Read cfg.artifact_source, defaulting to W&B, and reject anything unrecognised.

    Chosen EXPLICITLY by config rather than inferred from whether a local file happens to exist.
    An implicit fallback would mean that a W&B outage silently redirects an evaluation onto
    whatever stale checkpoint is lying around on disk, and the results would look entirely normal.
    So an offline run has to say so, and a wandb run that cannot reach wandb fails.

    Args:
        cfg: Any config with an optional top-level ``artifact_source`` key.

    Returns:
        One of :data:`ARTIFACT_SOURCES`.

    Raises:
        ValueError: The configured source is not a recognised one.
    """
    source = cfg.get("artifact_source") or ARTIFACT_SOURCE_WANDB
    if source not in ARTIFACT_SOURCES:
        raise ValueError(f"Unknown artifact_source={source!r}. Choose from {list(ARTIFACT_SOURCES)}.")
    return str(source)


def describe_run_source(run_id: str | None) -> str:
    """Human-readable provenance of a loaded checkpoint, for log lines.

    Exists so no log line has to interpolate a bare None into a sentence ending in "wandb run",
    which reads as a broken run id rather than as "there was no run".

    Args:
        run_id: The run id a load returned, or None for a local checkpoint.

    Returns:
        Either ``"wandb run <id>"`` or ``"a local checkpoint (no wandb run)"``.
    """
    return f"wandb run {run_id}" if run_id is not None else "a local checkpoint (no wandb run)"


def local_checkpoint_path(artifact_name: str) -> pathlib.Path:
    """Where train.py writes (and the local source reads) the checkpoint for an artifact name.

    Args:
        artifact_name: The name from :func:`resolve_artifact_name`.

    Returns:
        ``<CHECKPOINTS_PATH>/<artifact_name>.pt``.
    """
    return pathlib.Path(CHECKPOINTS_PATH) / f"{artifact_name}.pt"


def resolve_artifact_name(cfg: DictConfig | dict[str, Any]) -> str:
    """Build <method>[_alpha<a>][_<tag>][_deltag<d>]_<base_model>_<dataset>[_num<num_train>]_seed<seed>.

    The _num<num_train> suffix is omitted when num_train is null (full training set), so
    artifact names of full-data runs stay unchanged. Alpha is canonicalized to its float
    repr, so the CLI literals 1 and 1.0 name the same artifact (alpha1.0).

    THE OPTIONAL ``method.artifact_tag`` exists because alpha is the only *fit* parameter the name
    encodes, while some sweeps vary a parameter that lives under ``method.params`` instead --
    credal_rl_multinomial's ``bandwidth_mult`` is the motivating case. Without a tag every
    bandwidth would resolve to one name and the fits would silently overwrite each other, leaving
    a sweep that looks complete but stores a single model. Absent (the default for most configs)
    the name is byte-identical to what it was before this parameter existed, so no existing
    artifact is renamed.

    The tag sits DIRECTLY AFTER the alpha segment, which is where the artifacts trained so far put
    it. It is not a free choice: moving it would rename every tagged artifact and silently orphan
    the ones already trained on W&B.

    Args:
        cfg: live OmegaConf cfg, or the plain dict from checkpoint['config'].

    Returns:
        Artifact / wandb run name.
    """
    method = cfg["method"] if isinstance(cfg, dict) else cfg.method
    train_section = method.get("train") or {}
    alpha = train_section.get("alpha") if train_section else None
    alpha_suffix = f"_alpha{float(alpha)}" if alpha is not None else ""
    tag = method.get("artifact_tag")
    tag_suffix = f"_{tag}" if tag else ""
    delta_g = train_section.get("delta_g") if train_section else None
    deltag_suffix = f"_deltag{float(delta_g)}" if delta_g is not None else ""
    method_name = method["name"] if isinstance(method, dict) else method.name
    base_model = cfg["base_model"] if isinstance(cfg, dict) else cfg.base_model
    dataset = cfg["dataset"] if isinstance(cfg, dict) else cfg.dataset
    seed = cfg["seed"] if isinstance(cfg, dict) else cfg.seed
    num_train = cfg.get("num_train")
    num_suffix = f"_num{num_train}" if num_train is not None else ""
    return f"{method_name}{alpha_suffix}{tag_suffix}{deltag_suffix}_{base_model}_{dataset}{num_suffix}_seed{seed}"


def _download_checkpoint_from_wandb(
    artifact_name: str, entity: str, project: str, device: torch.device
) -> tuple[dict[str, Any], str]:
    """Download :latest of the artifact and torch.load its .pt.

    Args:
        artifact_name: wandb artifact name (without :version).
        entity: wandb entity.
        project: wandb project.
        device: target device; MPS gets mapped to CPU at load.

    Returns:
        (checkpoint dict, run_id of the wandb run that logged the artifact).

    Raises:
        RuntimeError: artifact not found, or it doesn't contain exactly one .pt file.
    """
    api = wandb.Api(timeout=60)
    full_name = f"{entity}/{project}/{artifact_name}:latest"
    try:
        artifact = api.artifact(full_name)
    except Exception as e:  # noqa: BLE001
        msg = f"No trained model found for {artifact_name!r} in {entity}/{project}. Original: {e}"
        raise RuntimeError(msg) from None

    artifact_dir = artifact.download()
    pt_files = list(pathlib.Path(artifact_dir).glob("*.pt"))
    if len(pt_files) != 1:
        raise RuntimeError(f"Expected exactly one .pt file in artifact, found {len(pt_files)}")
    load_device = torch.device("cpu") if device.type == "mps" else device
    checkpoint = torch.load(pt_files[0], map_location=load_device, weights_only=False)
    run_id = artifact.logged_by().id
    return checkpoint, run_id


def _load_checkpoint_from_disk(artifact_name: str, device: torch.device) -> tuple[dict[str, Any], None]:
    """torch.load the local .pt for an artifact name.

    Args:
        artifact_name: wandb artifact name, which is also the checkpoint's filename stem.
        device: target device; MPS gets mapped to CPU at load, as in the W&B path.

    Returns:
        (checkpoint dict, None). The second element is the run id, and there is no W&B run behind a
        local file, so it is None rather than a placeholder string -- callers that log or resume a
        run have to handle the absence explicitly instead of writing to a run named "local".

    Raises:
        RuntimeError: No checkpoint at that path. RuntimeError specifically, matching
            _download_checkpoint_from_wandb, so the driver's skip handling and load_base_predictor's
            "not trained yet" fallback treat a missing local file exactly like a missing artifact.
    """
    path = local_checkpoint_path(artifact_name)
    if not path.is_file():
        msg = (
            f"No local checkpoint for {artifact_name!r} at {path}. Train it with "
            f"save_to_disk=true, or set artifact_source={ARTIFACT_SOURCE_WANDB!r} to load from W&B."
        )
        raise RuntimeError(msg)
    load_device = torch.device("cpu") if device.type == "mps" else device
    checkpoint = torch.load(path, map_location=load_device, weights_only=False)
    return checkpoint, None


def _acquire_checkpoint(
    cfg: DictConfig | dict[str, Any], artifact_name: str, device: torch.device
) -> tuple[dict[str, Any], str | None]:
    """Fetch a checkpoint from whichever source cfg selects, and say which one in the log.

    The only difference between the two paths is where the dict comes from; everything downstream
    (_build_model_from_checkpoint and the callers) is shared.

    Args:
        cfg: Config carrying ``artifact_source`` and, for the W&B source, ``wandb.entity`` /
            ``wandb.project``.
        artifact_name: The name from :func:`resolve_artifact_name`.
        device: target device.

    Returns:
        (checkpoint dict, run_id), run_id being None for the local source.
    """
    source = resolve_artifact_source(cfg)
    if source == ARTIFACT_SOURCE_LOCAL:
        logger.info("Loading %s from local checkpoint %s", artifact_name, local_checkpoint_path(artifact_name))
        return _load_checkpoint_from_disk(artifact_name, device)
    wandb_cfg = cfg["wandb"]
    entity, project = wandb_cfg["entity"], wandb_cfg["project"]
    logger.info("Loading %s from W&B artifact %s/%s/%s:latest", artifact_name, entity, project, artifact_name)
    return _download_checkpoint_from_wandb(artifact_name, entity, project, device)


def _build_model_from_checkpoint(
    checkpoint: dict[str, Any], device: torch.device
) -> tuple[nn.Module | list[nn.Module], dict[str, Any]]:
    """Rebuild via models.build_model from checkpoint['config'], load weights, eval mode.

    Args:
        checkpoint: dict with 'config' and 'model_state_dict'.
        device: target device.

    Returns:
        (model, train_cfg_dict).
    """
    cfg = checkpoint["config"]
    num_classes = data.DATASET_NUM_CLASSES[cfg["dataset"]]
    method_params = cfg["method"].get("params") or {}
    model = models.build_model(
        cfg["method"]["name"],
        cfg["base_model"],
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", False),
        model_type=cfg["model_type"],
        params=method_params,
    )

    state = checkpoint["model_state_dict"]
    if isinstance(model, list):
        # CRL: plain Python list, state is list of dicts (see train.py:_get_state_dict).
        for member, member_state in zip(model, state, strict=True):
            member.load_state_dict(member_state)  # ty: ignore[unresolved-attribute]
            _to_eval_device(member, device)  # ty: ignore[invalid-argument-type]
    else:
        model.load_state_dict(state)
        _to_eval_device(model, device)
    return model, cfg


def _to_eval_device(model: nn.Module, device: torch.device) -> None:
    """Move a rebuilt model to its inference device in eval mode, in place.

    The MPS backend cannot represent float64 at all, so any double-precision buffer a method
    stores (efficient_credal_prediction keeps its logit bounds as float64) is downcast to float32
    first -- a no-op for the all-float32 methods, and far below the scale of any set-membership
    tolerance for the rest. CUDA and CPU keep the stored dtypes untouched.
    """
    if device.type == "mps":
        model.float()
    model.to(device)
    model.eval()


def load_model_for_evaluation(
    cfg: DictConfig, device: torch.device
) -> tuple[nn.Module | list[nn.Module], dict[str, Any], str | None]:
    """Fetch and rebuild the model artifact for cfg, from W&B or from local disk.

    Args:
        cfg: Hydra eval config with method / recipe / seed / wandb fields, and optionally
            ``artifact_source`` (see :func:`resolve_artifact_source`; defaults to W&B).
        device: target device.

    Returns:
        (model, train_cfg, run_id). run_id is the wandb run that logged the artifact; use it with
        wandb.init(resume='must') to append eval metrics. It is None when the checkpoint came off
        local disk, because there is no run to append to -- callers must check before resuming.
    """
    artifact_name = resolve_artifact_name(cfg)
    checkpoint, run_id = _acquire_checkpoint(cfg, artifact_name, device)
    model, train_cfg = _build_model_from_checkpoint(checkpoint, device)
    return model, train_cfg, run_id


def load_base_predictor(cfg: DictConfig, device: torch.device) -> tuple[nn.Module, str | None] | None:
    """Try to load the base-method counterpart artifact for cfg (same base_model/dataset/num_train/seed).

    Used by training routines that have a base-predictor step (e.g. efficient_credal_prediction)
    to skip retraining when a base artifact already exists in W&B. The lookup name is built via
    resolve_artifact_name on a base-method-stripped view of cfg, so it follows whatever convention
    resolve_artifact_name produces.

    Args:
        cfg: live training cfg.
        device: target device.

    Returns:
        (base_predictor, run_id) if a matching base artifact exists, else None.
    """
    base_view = {
        "method": {"name": "base"},
        "base_model": cfg.base_model,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "num_train": cfg.get("num_train"),
    }
    base_name = resolve_artifact_name(base_view)
    try:
        checkpoint, run_id = _acquire_checkpoint(cfg, base_name, device)
    except RuntimeError:
        return None
    model, _ = _build_model_from_checkpoint(checkpoint, device)
    if isinstance(model, list):
        raise TypeError(f"Base artifact {base_name!r} unexpectedly contains a list (CreRL-style) checkpoint.")
    return model, run_id


def load_credal_wrapper_ensemble(cfg: DictConfig, device: torch.device) -> tuple[nn.Module, str | None] | None:
    """Try to load the credal_wrapper counterpart ensemble for cfg.

    Used by credal_ensembling training: the two methods share the exact same trained
    nn.ModuleList (probly's ensemble(base, num_members)) and differ only in the representer
    type at inference. So if a matching credal_wrapper artifact exists in W&B, we reuse its
    weights and skip the per-member CE training.

    The matching key is (base_model, dataset, num_train, seed, method.params.num_members);
    method.params.num_members must agree between the wrapper artifact and the ensembling cfg
    or the state_dict keys won't line up.

    Returns:
        (ensemble_module, run_id) if a matching credal_wrapper artifact exists, else None.
    """
    wrapper_view = {
        "method": {"name": "credal_wrapper"},
        "base_model": cfg.base_model,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "num_train": cfg.get("num_train"),
    }
    wrapper_name = resolve_artifact_name(wrapper_view)
    try:
        checkpoint, run_id = _acquire_checkpoint(cfg, wrapper_name, device)
    except RuntimeError:
        return None
    model, _ = _build_model_from_checkpoint(checkpoint, device)
    if isinstance(model, list):
        raise TypeError(f"credal_wrapper artifact {wrapper_name!r} unexpectedly contains a list checkpoint.")
    return model, run_id
