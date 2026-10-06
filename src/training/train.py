"""Training entrypoint. Loads config, builds model, dispatches to train_funcs, saves checkpoint."""

from __future__ import annotations

# Put src/ on sys.path so the top-level modules (data, models, utils, paths, ...) resolve
# when this file is run directly as `python src/training/train.py ...` from project root.
# Has no effect when invoked as `python -m training.train` from src/.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import logging
import pathlib
import tempfile
from typing import Any

import hydra
import torch
import torch.nn as nn
import wandb
import wandb.util
from omegaconf import DictConfig, OmegaConf

import data
import models
import utils
from artifacts import resolve_artifact_name
from paths import CHECKPOINTS_PATH
from training import train_funcs

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def _move_to_device(model: nn.Module | list[nn.Module], device: torch.device) -> None:
    """Move the model (or each member of a list ensemble) onto device."""
    if isinstance(model, list):
        for member in model:
            member.to(device)  # ty: ignore[unresolved-attribute]
    else:
        model.to(device)


def _maybe_compile_forward(model: nn.Module | list[nn.Module], device: torch.device, enable: bool) -> None:
    """torch.compile(model.forward) when on CUDA and enabled. Recurses into list ensembles."""
    if not enable:
        return
    if device.type != "cuda":
        logger.info("Skipping torch.compile (device=%s).", device.type)
        return
    if isinstance(model, list):
        for member in model:
            member.forward = torch.compile(member.forward)  # ty: ignore[unresolved-attribute]
    else:
        model.forward = torch.compile(model.forward)


def _get_state_dict(model: nn.Module | list[nn.Module]) -> dict | list[dict]:
    """state_dict for a single module, or list-of-dicts when the model is a plain list.

    CRL is the one method whose top-level object is a Python list (probly's class_bias_ensemble
    returns a list, not nn.ModuleList; wrapping it would break the protocol dispatch).
    """
    if isinstance(model, list):
        return [m.state_dict() for m in model]  # ty: ignore[unresolved-attribute]
    return model.state_dict()


def _save_checkpoint(model: nn.Module, cfg: DictConfig, test_metrics: dict[str, float], run: Any) -> None:  # noqa: ANN401
    """Write checkpoint to disk (if save_to_disk) and upload as a W&B artifact."""
    artifact_name = resolve_artifact_name(cfg)
    # OmegaConf.to_container returns a wider union (could be list/None/str); we know it's a dict.
    cfg_container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    checkpoint = {
        "model_state_dict": _get_state_dict(model),
        "config": cfg_container,
        "metrics": test_metrics,
    }
    if cfg.save_to_disk:
        path = pathlib.Path(CHECKPOINTS_PATH) / f"{artifact_name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        _log_wandb_artifact(path, artifact_name, cfg_container, run)  # ty: ignore[invalid-argument-type]
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / f"{artifact_name}.pt"
            torch.save(checkpoint, path)
            _log_wandb_artifact(path, artifact_name, cfg_container, run)  # ty: ignore[invalid-argument-type]


def _log_wandb_artifact(path: pathlib.Path, artifact_name: str, metadata: dict[str, Any], run: Any) -> None:  # noqa: ANN401
    """Upload a .pt file as a W&B model artifact."""
    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    artifact.add_file(str(path))
    run.log_artifact(artifact)


@hydra.main(version_base=None, config_path="../../configs/", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run training. Requires method=<m> and recipe=<r> on the command line."""
    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))

    utils.set_seed(cfg.seed)
    device = utils.get_device(cfg.get("device"))
    logger.info("Running on device: %s", device)

    run_id = wandb.util.generate_id()
    run_name = f"{resolve_artifact_name(cfg)}_{run_id}"
    with wandb.init(
        id=run_id,
        name=run_name,
        entity=cfg.wandb.get("entity"),
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore[invalid-argument-type]
        mode="online" if cfg.wandb.enabled else "disabled",
        save_code=True,
    ) as run:
        loader_kwargs: dict[str, Any] = {
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
        }
        train_loader, val_loader, test_loader = data.get_train_data(
            cfg.dataset, val_split=cfg.val_split, num_train=cfg.num_train, seed=cfg.seed, **loader_kwargs
        )

        num_classes = data.DATASET_NUM_CLASSES[cfg.dataset]
        params_section = cfg.method.get("params", {})
        # OmegaConf.to_container is wide-typed; method.params resolves to dict[str, Any] for us.
        params = OmegaConf.to_container(params_section, resolve=True) if params_section else {}
        model = models.build_model(
            cfg.method.name,
            cfg.base_model,
            num_classes=num_classes,
            pretrained=cfg.pretrained,
            model_type=cfg.model_type,
            params=params,  # ty: ignore[invalid-argument-type]
        )
        _move_to_device(model, device)
        _maybe_compile_forward(model, device, cfg.compile_forward)

        train_funcs.train_model(model, train_loader, val_loader, cfg, device, run)

        # Release training DataLoaders and force GC so any persistent workers shut down
        # before the test DataLoader spawns its own. On CUDA, this avoids
        # `cudaErrorInitializationError` in the test workers (they fork from a CUDA state
        # polluted by the long-lived training workers).
        del train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        test_metrics = train_funcs.evaluate(model, test_loader, device, bool(cfg.amp))
        run.summary.update(test_metrics)
        run.log(data=test_metrics)
        logger.info("Test metrics: %s", test_metrics)

        _save_checkpoint(model, cfg, test_metrics, run)

        # Shut down test_loader workers before the next multirun iteration to keep
        # the file-descriptor count from drifting upward across iterations.
        del test_loader
        gc.collect()


if __name__ == "__main__":
    main()
