"""Fast-path-compatible Mamba-LoRA weight parametrization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils import parametrize


class LoRAWeightParametrization(nn.Module):
    """Represent ``W + (alpha/rank) * B @ A`` through the ``weight`` attribute."""

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        # Parametrizations are evaluated while Mamba runs inside autocast. Build
        # the effective weight deterministically in FP32 so training-time fused
        # kernels and offline merged export use exactly the same LoRA delta.
        with torch.autocast(device_type=base_weight.device.type, enabled=False):
            effective = base_weight.float() + (
                self.lora_B.float() @ self.lora_A.float()
            ) * self.scaling
        return effective.to(dtype=base_weight.dtype)


@dataclass(frozen=True)
class LinearLoRAInjection:
    module_path: str
    target_name: str


@dataclass(frozen=True)
class LoRAInjection:
    layer_index: int
    target_name: str
    forward_path: str
    reverse_path: str


def _mamba_pair(backbone: nn.Module, layer_index: int):
    try:
        wrapper = backbone.layers[layer_index].mixer.mixer
        return wrapper.mamba_fwd, wrapper.mamba_rev
    except (AttributeError, IndexError) as error:
        raise TypeError(
            f"Backbone layer {layer_index} is not the expected BiMamba wrapper."
        ) from error


def _register_one(linear: nn.Linear, *, rank: int, alpha: float):
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"LoRA target must be nn.Linear, got {type(linear).__name__}.")
    if parametrize.is_parametrized(linear, "weight"):
        raise RuntimeError("LoRA target weight is already parametrized.")
    out_features, in_features = linear.weight.shape
    parametrization = LoRAWeightParametrization(
        out_features,
        in_features,
        rank=rank,
        alpha=alpha,
    )
    parametrize.register_parametrization(linear, "weight", parametrization)
    linear.parametrizations.weight.original.requires_grad = False
    return parametrization


def inject_mamba_lora(
    backbone: nn.Module,
    *,
    layers: Sequence[int],
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    share_bidirectional: bool,
) -> Tuple[LoRAInjection, ...]:
    """Inject LoRA into selected BiMamba projections without replacing forward()."""

    injections: List[LoRAInjection] = []
    for layer_index in layers:
        mamba_fwd, mamba_rev = _mamba_pair(backbone, int(layer_index))
        for target_name in target_modules:
            fwd_linear = getattr(mamba_fwd, target_name, None)
            rev_linear = getattr(mamba_rev, target_name, None)
            if not isinstance(fwd_linear, nn.Linear) or not isinstance(rev_linear, nn.Linear):
                raise TypeError(
                    f"Layer {layer_index} target {target_name!r} is not Linear in both directions."
                )
            base_shared = fwd_linear.weight is rev_linear.weight
            if share_bidirectional and not base_shared:
                raise RuntimeError(
                    f"Layer {layer_index} {target_name} base weights are not bidirectionally shared."
                )

            fwd_lora = _register_one(fwd_linear, rank=rank, alpha=alpha)
            rev_lora = _register_one(rev_linear, rank=rank, alpha=alpha)
            if share_bidirectional:
                if (
                    fwd_linear.parametrizations.weight.original
                    is not rev_linear.parametrizations.weight.original
                ):
                    raise RuntimeError(
                        f"Parametrization broke shared base weight at layer {layer_index} {target_name}."
                    )
                rev_lora.lora_A = fwd_lora.lora_A
                rev_lora.lora_B = fwd_lora.lora_B

            prefix = f"caduceus_ph.backbone.layers.{layer_index}.mixer.mixer"
            injections.append(
                LoRAInjection(
                    layer_index=int(layer_index),
                    target_name=str(target_name),
                    forward_path=f"{prefix}.mamba_fwd.{target_name}",
                    reverse_path=f"{prefix}.mamba_rev.{target_name}",
                )
            )
    return tuple(injections)


def inject_linear_lora(
    root: nn.Module,
    *,
    target_paths: Sequence[str],
    rank: int,
    alpha: float,
) -> Tuple[LinearLoRAInjection, ...]:
    """Inject LoRA into explicit Linear paths relative to ``root``."""

    injections: List[LinearLoRAInjection] = []
    for target_path in target_paths:
        try:
            target = root.get_submodule(str(target_path))
        except AttributeError as error:
            raise TypeError(f"Transformer-LoRA target is missing: {target_path}.") from error
        _register_one(target, rank=rank, alpha=alpha)
        module_path, _, target_name = str(target_path).rpartition(".")
        injections.append(
            LinearLoRAInjection(module_path=module_path, target_name=target_name)
        )
    return tuple(injections)


def iter_lora_parameters(module: nn.Module):
    """Yield unique named LoRA parameters in deterministic model order."""

    seen = set()
    for name, parameter in module.named_parameters():
        if not (
            name.endswith(".lora_A") or name.endswith(".lora_B")
        ):
            continue
        if "parametrizations.weight" not in name or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        yield name, parameter


def lora_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for _, parameter in iter_lora_parameters(module))


def effective_state_dict_for_unparametrized_model(
    parametrized_model: nn.Module,
    reference_model: nn.Module,
):
    """Build a normal state dict with every LoRA delta merged into base weights."""

    source_state = parametrized_model.state_dict()
    merged = {}
    for key, reference_value in reference_model.state_dict().items():
        if key in source_state:
            value = source_state[key]
        elif key.endswith(".weight"):
            module_path = key[: -len(".weight")]
            source_module = parametrized_model.get_submodule(module_path)
            if not parametrize.is_parametrized(source_module, "weight"):
                raise KeyError(f"Missing non-parametrized state key: {key}")
            value = source_module.weight.detach()
        else:
            raise KeyError(f"Cannot construct merged state key: {key}")
        if tuple(value.shape) != tuple(reference_value.shape):
            raise ValueError(
                f"Merged tensor shape mismatch for {key}: {tuple(value.shape)} vs "
                f"{tuple(reference_value.shape)}."
            )
        merged[key] = value.detach().to(device="cpu", dtype=reference_value.dtype)
    return merged
