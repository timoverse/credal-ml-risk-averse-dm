"""Collection of utility functions."""

from __future__ import annotations

import argparse
import copy
import inspect
import logging
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torch.optim as optim
from probly.quantification.measure.credal_set import upper_entropy
from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution
from scipy.special import log_softmax
from torch import Tensor, nn
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_device(name: str | None = None) -> torch.device:
    """Resolve a torch.device from a user spec, or auto-pick if None.

    Args:
        name: If given, used as-is (e.g. "cuda:0", "cuda:1", "cpu", "mps"). Raises if the
            requested device is unavailable. If None, picks the least-utilized CUDA device,
            else MPS, else CPU.

    Returns:
        A torch.device.

    Raises:
        ValueError: If a requested CUDA/MPS device is unavailable.
    """
    if name is not None:
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"Requested {name} but CUDA is not available.")
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested {name} but only {torch.cuda.device_count()} CUDA device(s) available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError(f"Requested {name} but MPS is not available.")
        return device

    if torch.cuda.is_available():
        utilizations = [torch.cuda.utilization(i) for i in range(torch.cuda.device_count())]
        return torch.device(f"cuda:{utilizations.index(min(utilizations))}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def printargs(args: argparse.Namespace) -> None:
    """Print the arguments in a formatted way.

    Args:
        args: The argparse.Namespace object containing the arguments to print.
    """
    print("\n" + "=" * 25)
    print("Starting experiment with: ")
    print("=" * 25)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 25 + "\n")


@torch.no_grad()
def torch_get_logits(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> tuple[Tensor, Tensor]:
    """Get the logits and targets from a torch model and data loader.

    Args:
        model: The torch model to get the logits from.
        loader: The data loader to get the data from.
        device: The device to run the model on.

    Returns:
        tuple[Tensor, Tensor]:
        A tuple containing the logits and targets.
    """
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device, dtype=torch.long)
    for input, target in tqdm(loader, desc="Batch of instances:"):
        input, target = input.to(device), target.to(device)
        targets = torch.cat((targets, target), dim=0)
        outputs = torch.cat((outputs, model(input)), dim=0)
    return outputs, targets


@torch.no_grad()
def collect_credal_sets_targets(
    rep: Any,  # noqa: ANN401
    loader: DataLoader[Any],
    device: torch.device,
    amp_enabled: bool = False,
) -> tuple[list[TorchConvexCredalSet | TorchProbabilityIntervalsCredalSet], list[Tensor]]:
    """Run a representer over a loader, gathering the per-batch credal sets and targets.

    Feeds cvar_minimax calibration, which needs the raw credal sets (not reduced predictions) so it
    can run the solver at many thresholds v. Returns aligned per-batch lists, both on the device.

    Args:
        rep: A representer whose predict returns a credal set for a batch of inputs.
        loader: The data loader to run over.
        device: The device to run inference on.
        amp_enabled: Whether to wrap the forward pass in autocast.

    Returns:
        A tuple of the per-batch credal sets and the per-batch targets.
    """
    rep_outs: list[TorchConvexCredalSet | TorchProbabilityIntervalsCredalSet] = []
    targets: list[Tensor] = []
    for inputs, batch_targets in tqdm(loader, desc="Collecting credal sets:"):
        x = inputs.to(device, non_blocking=True)
        if device.type == "cuda" and x.ndim >= 4:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            rep_outs.append(rep.predict(x))
        targets.append(batch_targets.to(device))
    return rep_outs, targets


@torch.no_grad()
def torch_get_logits_ensemble(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    logits: bool = True,
    alpha: None | float = None,
) -> tuple[Tensor, Tensor]:
    """Get the logits and targets from a torch ensemble model and data loader.

    The model must expose a ``predict_representation`` method. Historically this was the
    locally defined ``Ensemble`` class (deleted in the rewrite to ``train.py``); the type
    is left as ``nn.Module`` so legacy ``exp_*`` scripts that still call this function
    continue to type-check while we phase them out.

    Args:
        model: A torch ensemble model exposing ``predict_representation(x, logits, alpha?)``.
        loader: The data loader to get the data from.
        device: The device to run the model on.
        logits: Whether to return logits or probabilities.
        alpha: The alpha value to use for the ensemble prediction (DesterckeEnsemble).

    Returns:
        A tuple containing the logits and targets.
    """
    model.to(device)
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device, dtype=torch.long)
    for input, target in tqdm(loader, desc="Batch of instances:"):
        input, target = input.to(device), target.to(device)
        targets = torch.cat((targets, target), dim=0)
        # ``predict_representation`` is a method on legacy local ensemble classes that no longer exist;
        # ``nn.Module.__getattr__`` resolves it to a ``Tensor | Module`` union that ty cannot
        # narrow. This function only survives for legacy ``exp_*`` consumers and will be removed
        # in a follow-up pass.
        if alpha is not None:
            outputs = torch.cat(
                (outputs, model.predict_representation(input, logits=logits, alpha=alpha)),  # ty: ignore[call-non-callable]
                dim=0,
            )
        else:
            outputs = torch.cat(
                (outputs, model.predict_representation(input, logits=logits)),  # ty: ignore[call-non-callable]
                dim=0,
            )
    return outputs, targets


def set_seed(seed: int) -> None:
    """Set the seed for random, numpy, and torch.

    Args:
        seed: The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_beta_from_aggregation(name: str) -> float:
    """Extract the CVaR tail level beta from a cvar_<beta> aggregation name.

    For example parse_beta_from_aggregation("cvar_0.1") returns 0.1. Shared across experiments that
    need the tail level behind a cvar aggregation key from metrics.AGGREGATION_REGISTRY.

    Args:
        name: An aggregation name of the form cvar_<float in (0, 1]>.

    Returns:
        The parsed beta.

    Raises:
        ValueError: name is not cvar_<float in (0, 1]> (for example mean, or a fixed-count cvar_50).
    """
    if not name.startswith("cvar_"):
        raise ValueError(f"Expected a cvar_<beta> aggregation name, got {name!r}.")
    try:
        beta = float(name.removeprefix("cvar_"))
    except ValueError as e:
        raise ValueError(f"Could not parse beta from aggregation={name!r}; expected cvar_<float>.") from e
    if not 0.0 < beta <= 1.0:
        raise ValueError(f"Parsed beta={beta} from aggregation={name!r} is not in (0, 1].")
    return beta


def str2bool(s: str | bool) -> bool:
    """Convert a string to a boolean.

    Args:
        s: The string to convert.

    Returns:
        The converted boolean.
    """
    if isinstance(s, bool):
        return s
    elif s.lower() in ("yes", "true", "t", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def printt(var: Any, name: str | None = None, max_items: int = 10) -> None:  # noqa: ANN401, Any is allowed here
    """Prints variable name, type, shape (if numpy or torch), and content.

    Args:
        var: variable to inspect
        name: variable name override
        max_items: max number of elements to print for large arrays/tensors
    """
    # Try to infer variable name from caller if not provided
    if name is None:
        frame = inspect.currentframe().f_back  # type: ignore
        for k, v in frame.f_locals.items():  # type: ignore
            if v is var:
                name = k
                break
        else:
            name = "<unknown>"

    print(f"   Variable: {name}")
    print(f"   Type: {type(var)}")

    # NumPy
    try:
        import numpy as np

        if isinstance(var, np.ndarray):
            print(f"   Shape: {var.shape}")
            print(f"   Dtype: {var.dtype}")
            data = var.flatten()
            print(f"   Content: {data[:max_items]}{' ...' if data.size > max_items else ''}")
            return
    except ImportError:
        pass

    # PyTorch
    try:
        import torch

        if isinstance(var, torch.Tensor):
            print(f"   Shape: {tuple(var.shape)}")
            print(f"   Dtype: {var.dtype}")
            data = var.flatten()
            print(f"   Content: {data[:max_items]}{' ...' if data.numel() > max_items else ''}")
            return
    except ImportError:
        pass

    # Fallback
    print(f"   Content: {var}")


def get_optimizer(name: str, params: Iterator[nn.Parameter], lr: float, wd: float) -> optim.Optimizer:
    """Get the optimizer based on the name.

    Args:
        name: Name of the optimizer.
        params: Parameters to optimize.
        lr: Learning rate.
        wd: Weight decay.

    Returns:
        The optimizer instance.
    """
    match name:
        case "sgd":
            op = optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        case "adam":
            op = optim.Adam(params, lr=lr, weight_decay=wd)
        case _:
            raise ValueError(f"Unknown optimizer name: {name}")
    return op


def log_likelihood(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the average log-likelihood given model outputs and targets.

    Args:
        outputs: The model outputs (logits) of shape (N, C) with N samples and C classes.
        targets: The target labels of shape (N,).

    Returns:
        A scalar tensor representing the average log-likelihood.
    """
    outputs = F.log_softmax(outputs, dim=1)
    ll = torch.mean(outputs[torch.arange(outputs.shape[0]), targets])
    return ll


def log_likelihood_scipy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Computes the log-likelihood with scipy for the given logits and targets.

    The function takes the logits (unnormalized log probabilities) as inputs
    and applies log-softmax to convert them into log-probabilities. It then
    computes the mean log-probability corresponding to the target labels,
    which represents the overall log-likelihood.

    Args:
        logits: An array containing the logits for each class.
            It should have shape (N, C), where N is the number of samples,
            and C is the number of classes.
        targets: An array containing the target class indices for
            each sample. It should have shape (N,).

    Returns:
        A scalar representing the mean log-likelihood of the
        target labels.
    """
    probs = log_softmax(logits, axis=1)
    ll = np.mean(probs[np.arange(probs.shape[0]), targets])
    return ll


def log_likelihood_sum(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute the log-likelihood sum given model outputs and targets.

    Args:
        outputs: The model outputs (logits) of shape (N, C) with N samples and C classes.
        targets: The target labels of shape (N,).

    Returns:
        A scalar tensor representing the log-likelihood sum.
    """
    outputs = F.log_softmax(outputs, dim=1)
    ll = torch.sum(outputs[torch.arange(outputs.shape[0]), targets.int()])
    return ll


def loader_to_tensor(loader: DataLoader[Any]) -> tuple[Tensor, Tensor]:
    """Convert a DataLoader to tensors for inputs and targets.

    Args:
        loader: DataLoader to convert.

    Returns:
        A tuple containing tensors for inputs and targets.
    """
    x = []
    y = []
    for data in loader:
        x.append(data[0])
        y.append(data[1])
    x = torch.cat(x)
    y = torch.cat(y)
    return x, y


def convex_upper_entropy(probs: np.ndarray) -> npt.NDArray[np.floating]:
    """Compute the upper entropy of a batch of convex credal sets (numpy interface).

    Wraps probly's :func:`probly.quantification.measure.credal_set.upper_entropy`,
    which now requires a :class:`CredalSet` object rather than a raw array.

    Args:
        probs: shape ``(N, M, K)`` -- ``M`` categorical distributions per instance.

    Returns:
        shape ``(N,)`` natural-log upper entropy per instance.
    """
    t = torch.as_tensor(probs)
    cset = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(t))
    out: Tensor = upper_entropy(cset)  # ty:ignore[invalid-assignment]
    return out.cpu().numpy()


def credal_set_size(csets: np.ndarray) -> float:
    """Compute the size of the credal set based on the average interval size between upper and lower probabilities."""
    upper_probs = np.max(csets, axis=1)
    lower_probs = np.min(csets, axis=1)
    set_size = np.mean(upper_probs - lower_probs)
    return set_size


def get_probs_from_model(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Get the probabilities from a model and data loader."""
    with torch.no_grad():
        logits = model(x)
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits.squeeze(-1))
            probs = torch.stack([1 - probs, probs], dim=1).numpy()
        else:
            probs = logits.softmax(dim=1).numpy()
    return probs


# Generate grid of possible categorical distributions for num_classes
COMMON_HEAD_ATTRS = ["fc", "linear", "head", "classifier", "output_layer", "out"]


def get_last_layer(model: nn.Module) -> tuple[nn.Module, str]:
    """Get the last layer of a model. Logs which attribute it picked."""
    for attr in COMMON_HEAD_ATTRS:
        layer = getattr(model, attr, None)
        if isinstance(layer, nn.Linear):
            logger.info("get_last_layer: picked %s.%s (%s).", type(model).__name__, attr, type(layer).__name__)
            return layer, attr

    raise AttributeError(f"Could not find last layer in the model. Checked attributes: {COMMON_HEAD_ATTRS}")


class EarlyStopping:
    """Stop training after `patience` consecutive epochs without val-loss improvement."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        """patience: epochs without improvement to tolerate. min_delta: minimum drop counted as improvement."""
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def should_stop(self, val_loss: float) -> bool:
        """Update with current val_loss, return True if patience exhausted."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class BestModelTracker:
    """Snapshot the best-on-val state_dict on CPU."""

    def __init__(self, min_delta: float = 0.0) -> None:
        """min_delta: minimum val-loss drop counted as improvement."""
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state_dict: dict[str, torch.Tensor] | None = None

    def update(self, val_loss: float, model: nn.Module) -> None:
        """Snapshot model.state_dict() if val_loss is the best so far."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def restore(self, model: nn.Module) -> None:
        """Load the best snapshot back into model, if one exists."""
        if self.best_state_dict is not None:
            model.load_state_dict(copy.deepcopy(self.best_state_dict))
