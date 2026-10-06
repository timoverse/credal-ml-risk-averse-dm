"""Base models and method-wrapped model construction."""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any

import torch.nn as nn
from probly.method.credal_bnn import credal_bnn
from probly.method.credal_ensembling import credal_ensembling
from probly.method.credal_relative_likelihood import credal_relative_likelihood
from probly.method.credal_wrapper import credal_wrapper
from probly.method.efficient_credal_prediction import efficient_credal_prediction
from probly.predictor import LogitDistributionPredictor

from methods import CredalRLMultinomialPredictor

if TYPE_CHECKING:
    from collections.abc import Callable


@LogitDistributionPredictor.register_factory
def get_base_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Return a base logit classifier by name. Registered as a LogitDistributionPredictor."""
    match name:
        case "resnet18":
            if pretrained:
                raise NotImplementedError("Pretrained weights are not supported for the local ResNet18.")
            from kuangliu_resnet import ResNet18

            model = ResNet18()
            model.linear = nn.Linear(model.linear.in_features, num_classes)
        case "resnet18_encoder":
            if pretrained:
                raise NotImplementedError("Pretrained weights are not supported for the local ResNet18.")
            from kuangliu_resnet import ResNet18

            model = ResNet18()
            model.linear = nn.Identity()
        case _:
            raise ValueError(f"Unknown base model: {name}")
    return model


def credal_rl_multinomial(
    base: nn.Module,
    *,
    num_classes: int,
    predictor_type: str | None = None,  # noqa: ARG001
    **params: Any,  # noqa: ANN401
) -> nn.Module:
    """Wrap an encoder as a CredalRLMultinomialPredictor. Bandwidth and ridge come from method.params."""
    return CredalRLMultinomialPredictor(base, num_classes, **params)


# Method factories: take a pre-built base and return the wrapped predictor. None = passthrough.
# Typed as Any because probly's wrappers return heterogeneous Predictor protocol types that
# don't unify with nn.Module statically (though they all ARE nn.Modules at runtime).
_METHODS: dict[str, Callable[..., Any] | None] = {
    "base": None,
    "credal_bnn": credal_bnn,
    "credal_ensembling": credal_ensembling,
    "credal_relative_likelihood": credal_relative_likelihood,
    "credal_wrapper": credal_wrapper,
    # credal_dro: same predictor and representer as credal_wrapper (probly's ensemble +
    # box credal set); the method differs only in training (per-member top-delta CE),
    # which train_funcs.train_credal_dro implements.
    "credal_dro": credal_wrapper,
    "efficient_credal_prediction": efficient_credal_prediction,
    "credal_rl_multinomial": credal_rl_multinomial,
    # sqwash: trains a plain logit classifier with sqwash.SuperquantileReducer instead
    # of mean reduction on the per-instance CE losses. Inference is identical to base,
    # so no wrapper is needed -- the representer dispatches via LogitDistributionPredictor.
    "sqwash": None,
    # adacvar: trains a plain logit classifier with mean CE on minibatches drawn by an
    # Exp3-style two-stage sampler. Inference is identical to base; no wrapper needed.
    "adacvar": None,
}


def _default_build(
    method_fn: Callable[..., Any] | None,
    base_name: str,
    num_classes: int,
    pretrained: bool,
    model_type: str,
    params: dict[str, Any],
) -> nn.Module:
    """Construct the full base by name and wrap it via method_fn. Used for full-classifier methods."""
    base = get_base_model(base_name, num_classes=num_classes, pretrained=pretrained)
    if method_fn is None:
        return base
    sig = inspect.signature(method_fn)
    accepts_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_keyword:
        accepted = dict(params)
    else:
        accepted_keys = set(sig.parameters)
        accepted = {k: v for k, v in params.items() if k in accepted_keys}
        dropped = set(params) - set(accepted)
        if dropped:
            name = getattr(method_fn, "__name__", repr(method_fn))
            warnings.warn(f"{name} does not accept {dropped}; dropping.", stacklevel=2)
    return method_fn(base, predictor_type=model_type, **accepted)


def _build_credal_rl_multinomial(
    method_fn: Callable[..., Any] | None,
    base_name: str,
    num_classes: int,
    pretrained: bool,
    model_type: str,
    params: dict[str, Any],
) -> nn.Module:
    """Head-less encoder variant that also needs the class count (its buffers carry no head at all)."""
    encoder = get_base_model(f"{base_name}_encoder", num_classes=num_classes, pretrained=pretrained)
    return method_fn(encoder, num_classes=num_classes, predictor_type=model_type, **params)  # ty: ignore[call-non-callable]


_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "credal_rl_multinomial": _build_credal_rl_multinomial,
}


def build_model(
    method_name: str,
    base_name: str,
    *,
    num_classes: int,
    pretrained: bool,
    model_type: str,
    params: dict[str, Any],
) -> nn.Module:
    """Dispatch to the method's builder, which constructs its own base (encoder or classifier) and wraps it."""
    if method_name not in _METHODS:
        raise ValueError(f"Unknown method: {method_name}")
    method_fn = _METHODS[method_name]
    builder = _BUILDERS.get(method_name, _default_build)
    return builder(method_fn, base_name, num_classes, pretrained, model_type, params)
