"""Strict YAML configuration for GeneAnn PEFT and full-parameter tuning."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar, Union

import yaml

from .loss import (
    LOSS_MASKED_FOCAL_F1,
    LOSS_TIBERIUS_CCE_F1,
    SUPPORTED_LOSS_NAMES,
)


PEFT_CONFIG_NAME = "peft_tuning_config.yml"
HALF_CONFIG_NAME = "half_tuning_config.yml"
FULL_CONFIG_NAME = "full_tuning_config.yml"
RESOLVED_PEFT_CONFIG_NAME = "resolved_peft_tuning_config.yml"
RESOLVED_HALF_CONFIG_NAME = "resolved_half_tuning_config.yml"
RESOLVED_FULL_CONFIG_NAME = "resolved_full_tuning_config.yml"
# Backward-compatible alias used by existing PEFT callers.
RESOLVED_CONFIG_NAME = RESOLVED_PEFT_CONFIG_NAME
SUPPORTED_SCHEDULER_NAMES = ("cosine", "linear")
FORMAT_VERSION = 1
_ALLOWED_TARGETS = {"in_proj", "out_proj"}
_FIXED_MAMBA_LORA_TARGET_MODULES: Tuple[str, ...] = ("in_proj", "out_proj")
# GeneAnn-v2 plants backbone depth used by the fixed PEFT/half Mamba-LoRA scope.
EXPECTED_MAMBA_LORA_N_LAYER = 16
_T = TypeVar("_T")
TuningConfig = Union["PeftTuningConfig", "HalfTuningConfig", "FullTuningConfig"]


def _strict_dataclass(cls: Type[_T], data: Mapping[str, Any], path: str) -> _T:
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must be a mapping, got {type(data).__name__}.")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {unknown}.")
    return cls(**dict(data))


def _require_positive(name: str, value: float) -> None:
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_non_negative(name: str, value: float) -> None:
    if float(value) < 0:
        raise ValueError(f"{name} must be non-negative.")


def resolve_all_mamba_lora_layers(n_layer: int) -> Tuple[int, ...]:
    """Return the immutable full-backbone Mamba-LoRA layer indices.

    PEFT and half tuning always inject LoRA into every BiMamba block. The
    GeneAnn-v2 plants checkpoint is required to expose exactly
    ``EXPECTED_MAMBA_LORA_N_LAYER`` layers.
    """

    n_layer = int(n_layer)
    if n_layer <= 0:
        raise ValueError("Model n_layer must be positive.")
    if n_layer != EXPECTED_MAMBA_LORA_N_LAYER:
        raise ValueError(
            "Mamba-LoRA expects the GeneAnn-v2 plants backbone depth "
            f"n_layer={EXPECTED_MAMBA_LORA_N_LAYER}, got {n_layer}."
        )
    return tuple(range(n_layer))


@dataclass(frozen=True)
class MambaLoRAConfig:
    """Internal fixed PEFT Mamba-LoRA recipe.

    Layer selection is intentionally absent: PEFT always injects LoRA into every
    backbone BiMamba block. Target modules remain fixed to bidirectionally
    shared ``in_proj`` / ``out_proj`` projections.
    """

    enabled: bool = True
    target_modules: Tuple[str, ...] = _FIXED_MAMBA_LORA_TARGET_MODULES
    rank: int = 8
    alpha: float = 8.0
    dropout: float = 0.0
    share_bidirectional: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_modules", tuple(str(x) for x in self.target_modules)
        )
        if not self.enabled:
            raise ValueError("This tuning method requires mamba_lora.enabled: true.")
        if self.target_modules != _FIXED_MAMBA_LORA_TARGET_MODULES:
            raise ValueError(
                "Mamba-LoRA target_modules are fixed to "
                f"{list(_FIXED_MAMBA_LORA_TARGET_MODULES)}; got "
                f"{list(self.target_modules)}."
            )
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if self.dropout != 0.0:
            raise ValueError(
                "Mamba fast-path weight parametrization requires LoRA dropout=0.0."
            )

    def to_resolved_dict(self, *, n_layer: int) -> Dict[str, Any]:
        """Return the full runtime Mamba-LoRA recipe for provenance dumps."""

        return {
            "enabled": bool(self.enabled),
            "layers": list(resolve_all_mamba_lora_layers(n_layer)),
            "scope": "all_backbone_bimamba_layers",
            "target_modules": list(self.target_modules),
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "dropout": float(self.dropout),
            "share_bidirectional": bool(self.share_bidirectional),
        }


@dataclass(frozen=True)
class TransformerLoRAConfig:
    """Fixed LoRA recipe for every decoder TransformerEncoderBlock."""

    target_modules: Tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
        "ffn.gate_proj",
        "ffn.up_proj",
        "ffn.down_proj",
    )
    rank: int = 8
    alpha: float = 16.0

    def __post_init__(self) -> None:
        if not self.target_modules or len(set(self.target_modules)) != len(
            self.target_modules
        ):
            raise ValueError(
                "Transformer-LoRA target_modules must be non-empty and unique."
            )
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("Transformer-LoRA rank and alpha must be positive.")


@dataclass(frozen=True)
class FixedPeftDecoderConfig:
    """Fixed selective decoder recipe used only by ``PeftTuningConfig``."""

    def enabled_module_flags(self) -> Dict[str, bool]:
        return {
            "stem_block": False,
            "down_conv_blocks": False,
            "transformer_bridge": False,
            "up_conv_blocks": False,
            "last_up_conv_block": True,
            "local_refine_block": True,
            "final_norm": True,
            "prediction_head": True,
        }


_FIXED_PEFT_MAMBA_LORA = MambaLoRAConfig()
_FIXED_PEFT_TRANSFORMER_LORA = TransformerLoRAConfig()
_FIXED_PEFT_DECODER = FixedPeftDecoderConfig()


@dataclass(frozen=True)
class HalfMambaLoRAConfig:
    """Half-tuning Mamba-LoRA knobs with an immutable injection scope.

    Injection always covers all backbone BiMamba blocks on bidirectionally
    shared ``in_proj`` / ``out_proj`` projections. YAML may only configure rank,
    alpha, dropout, and sharing. Optimizer values come from the method-level
    ``half`` block.
    """

    enabled: bool = True
    rank: int = 8
    alpha: float = 8.0
    dropout: float = 0.0
    share_bidirectional: bool = True

    def __post_init__(self) -> None:
        if not self.enabled:
            raise ValueError("Half tuning requires mamba_lora.enabled: true.")
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if self.dropout != 0.0:
            raise ValueError(
                "Mamba fast-path weight parametrization requires LoRA dropout=0.0."
            )

    @property
    def target_modules(self) -> Tuple[str, ...]:
        return _FIXED_MAMBA_LORA_TARGET_MODULES

    def to_resolved_dict(self, *, n_layer: int) -> Dict[str, Any]:
        """Return the full runtime Mamba-LoRA recipe for provenance dumps."""

        return {
            "enabled": bool(self.enabled),
            "layers": list(resolve_all_mamba_lora_layers(n_layer)),
            "scope": "all_backbone_bimamba_layers",
            "target_modules": list(self.target_modules),
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "dropout": float(self.dropout),
            "share_bidirectional": bool(self.share_bidirectional),
        }


@dataclass(frozen=True)
class HalfDecoderFinetuningConfig:
    """Fixed fully-trainable half-tuning decoder scope.

    Module-selection flags are intentionally absent: the complete embeddings
    decoder and prediction head are always trainable. Optimizer values come
    from the method-level ``half`` block.
    """

    def enabled_module_flags(self) -> Dict[str, bool]:
        """Return the immutable trainable scope for resolved-run metadata."""
        return {
            "stem_block": True,
            "down_conv_blocks": True,
            "transformer_bridge": True,
            "up_conv_blocks": True,
            "last_up_conv_block": False,
            "local_refine_block": True,
            "final_norm": True,
            "prediction_head": True,
        }


@dataclass(frozen=True)
class FullTuningBodyConfig:
    """Naive full-parameter FT: single global LR / weight decay for all params."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        _require_positive("full.learning_rate", self.learning_rate)
        _require_non_negative("full.weight_decay", self.weight_decay)


@dataclass(frozen=True)
class LossConfig:
    """Per-strand loss selection shared by PEFT and full tuning.

    Supported ``name`` values come from ``tuning.loss.SUPPORTED_LOSS_NAMES``:
      - ``masked_focal_f1``: original GeneAnn hierarchical focal + F1 loss
        (uses gamma, state/intron/cds_f1_weight, gate_temperature, epsilon)
      - ``tiberius_cce_f1``: Tiberius CCE + coding-class soft F1
        (uses f1_factor, coding_start, epsilon)
    Unused fields for the selected loss are retained for YAML compatibility
    but ignored at construction time.
    """

    name: str = LOSS_MASKED_FOCAL_F1
    # masked_focal_f1 hyperparameters
    gamma: float = 2.0
    state_f1_weight: float = 0.5
    intron_f1_weight: float = 0.5
    cds_f1_weight: float = 0.5
    gate_temperature: float = 0.20
    # shared
    epsilon: float = 1e-5
    # tiberius_cce_f1 hyperparameters
    f1_factor: float = 2.0
    coding_start: int = 4

    def __post_init__(self) -> None:
        name = str(self.name)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coding_start", int(self.coding_start))
        if name == LOSS_MASKED_FOCAL_F1:
            if self.gamma < 0 or min(
                self.state_f1_weight, self.intron_f1_weight, self.cds_f1_weight
            ) < 0:
                raise ValueError("Loss gamma and F1 weights must be non-negative.")
            if self.gate_temperature <= 0 or self.epsilon <= 0:
                raise ValueError(
                    "Loss gate_temperature and epsilon must be positive."
                )
        elif name == LOSS_TIBERIUS_CCE_F1:
            if self.f1_factor < 0:
                raise ValueError("loss.f1_factor must be non-negative.")
            if not 0 < int(self.coding_start) < 15:
                raise ValueError(
                    "loss.coding_start must be in [1, 14] for 15-class labels."
                )
            if self.epsilon <= 0:
                raise ValueError("loss.epsilon must be positive.")
        else:
            raise ValueError(
                f"Unsupported loss.name={name!r}; expected one of "
                f"{list(SUPPORTED_LOSS_NAMES)}."
            )


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.99
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    train_batch_size: int = 1
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    epochs: int = 20
    eval_steps: int = 1000
    logging_steps: int = 1000
    early_stopping_patience: int = 2
    metric_for_best_model: str = "macro_f1"
    seed: int = 42
    num_workers: int = 4

    def __post_init__(self) -> None:
        if self.optimizer != "adamw":
            raise ValueError("Only the AdamW optimizer is supported.")
        if self.scheduler not in SUPPORTED_SCHEDULER_NAMES:
            raise ValueError(
                "Unsupported scheduler; expected one of "
                f"{list(SUPPORTED_SCHEDULER_NAMES)}."
            )
        if self.mixed_precision not in {"bf16", "fp16", "no"}:
            raise ValueError("mixed_precision must be bf16, fp16, or no.")
        integer_fields = (
            "train_batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "epochs",
            "eval_steps",
            "logging_steps",
            "early_stopping_patience",
            "seed",
            "num_workers",
        )
        invalid_integer_fields = [
            name
            for name in integer_fields
            if isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), int)
        ]
        if invalid_integer_fields:
            raise TypeError(
                f"Training integer fields have invalid types: {invalid_integer_fields}."
            )
        numeric_fields = (
            "beta1",
            "beta2",
            "warmup_ratio",
            "max_grad_norm",
        )
        invalid_numeric_fields = [
            name
            for name in numeric_fields
            if isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), (int, float))
        ]
        if invalid_numeric_fields:
            raise TypeError(
                f"Training numeric fields have invalid types: {invalid_numeric_fields}."
            )
        if not 0 < float(self.beta1) < 1 or not 0 < float(self.beta2) < 1:
            raise ValueError("AdamW beta1 and beta2 must be in (0, 1).")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1).")
        positive = (
            self.train_batch_size,
            self.eval_batch_size,
            self.gradient_accumulation_steps,
            self.epochs,
            self.eval_steps,
            self.logging_steps,
            self.early_stopping_patience,
        )
        if min(positive) <= 0 or self.max_grad_norm <= 0 or self.num_workers < 0:
            raise ValueError(
                "Training counts/max_grad_norm must be positive; num_workers non-negative."
            )
        if self.metric_for_best_model not in {
            "macro_f1",
            "boundary_f1",
            "cds_f1",
            "intron_f1",
            "nucleotide_accuracy",
        }:
            raise ValueError("Unsupported metric_for_best_model.")


@dataclass(frozen=True)
class CheckpointConfig:
    save_peft_only: bool = True
    save_full_inference_model: bool = True
    safe_serialization: bool = False
    save_last: bool = False
    # Full-tuning optional archival/export bundle under best/full/.
    save_trainable_bundle: bool = False

    def __post_init__(self) -> None:
        # Method-specific constraints are enforced by PeftTuningConfig / FullTuningConfig.
        for name in (
            "save_peft_only",
            "save_full_inference_model",
            "safe_serialization",
            "save_last",
            "save_trainable_bundle",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool.")


@dataclass(frozen=True)
class UniformOptimizationConfig:
    """Global optimizer values shared by every trainable module in a method."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        _require_positive("learning_rate", self.learning_rate)
        _require_non_negative("weight_decay", self.weight_decay)


@dataclass(frozen=True)
class PeftTuningConfig:
    """PEFT strategy with a uniform, user-configurable optimizer policy."""

    format_version: int = FORMAT_VERSION
    method: str = "peft"
    peft: UniformOptimizationConfig = UniformOptimizationConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()

    @property
    def optimization(self) -> UniformOptimizationConfig:
        return self.peft

    @property
    def mamba_lora(self) -> MambaLoRAConfig:
        return _FIXED_PEFT_MAMBA_LORA

    @property
    def transformer_lora(self) -> TransformerLoRAConfig:
        return _FIXED_PEFT_TRANSFORMER_LORA

    @property
    def decoder(self) -> FixedPeftDecoderConfig:
        return _FIXED_PEFT_DECODER

    def __post_init__(self) -> None:
        nested = (
            ("peft", UniformOptimizationConfig),
            ("loss", LossConfig),
            ("training", TrainingConfig),
            ("checkpoint", CheckpointConfig),
        )
        for name, cls in nested:
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, _strict_dataclass(cls, value, name))
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported PEFT config format_version={self.format_version}; "
                f"expected {FORMAT_VERSION}."
            )
        if self.method != "peft":
            raise ValueError("PeftTuningConfig.method must be 'peft'.")
        if not self.checkpoint.save_peft_only or not self.checkpoint.save_full_inference_model:
            raise ValueError(
                "PEFT checkpoints require save_peft_only=true and "
                "save_full_inference_model=true."
            )
        if self.checkpoint.save_trainable_bundle:
            raise ValueError(
                "PEFT config must set checkpoint.save_trainable_bundle=false "
                "(use best/peft instead)."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PeftTuningConfig":
        forbidden = sorted(
            set(data)
            & {
                "mamba_lora",
                "backbone_ffn_lora",
                "transformer_lora",
                "decoder",
                "full",
            }
        )
        if forbidden:
            raise ValueError(
                f"Fixed PEFT config must not expose strategy blocks: {forbidden}."
            )
        return _strict_dataclass(cls, data, "root")

    @classmethod
    def from_yaml(cls, path: str) -> "PeftTuningConfig":
        config_path = os.path.join(path, PEFT_CONFIG_NAME) if os.path.isdir(path) else path
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data is None:
            raise ValueError(f"Empty PEFT configuration: {config_path}")
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str) -> str:
        output_path = (
            os.path.join(path, PEFT_CONFIG_NAME) if os.path.isdir(path) else path
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)
        return output_path

    def resolve_lora_layers(self, n_layer: int) -> Tuple[int, ...]:
        return resolve_all_mamba_lora_layers(n_layer)

    def resolved_dict(self, *, n_layer: int, trainable: int, total: int) -> Dict[str, Any]:
        data = self.to_dict()
        mamba_lora = self.mamba_lora.to_resolved_dict(n_layer=n_layer)
        transformer_lora = asdict(self.transformer_lora)
        decoder = asdict(self.decoder)
        data["strategy"] = {
            "optimizer": asdict(self.peft),
            "mamba_lora": mamba_lora,
            "backbone_feed_forward": {
                "trainable": False,
                "lora": False,
                "scope": "all_dense_ffn_moe_experts_and_routers",
            },
            "transformer_lora": transformer_lora,
            "decoder": decoder,
            "decoder_train_flags": self.decoder.enabled_module_flags(),
        }
        data["runtime"] = {
            "method": self.method,
            "model_n_layer": int(n_layer),
            "resolved_lora_layers": list(self.resolve_lora_layers(n_layer)),
            "trainable_parameters": int(trainable),
            "total_parameters": int(total),
        }
        return data

    def save_resolved_yaml(
        self, output_dir: str, *, n_layer: int, trainable: int, total: int
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, RESOLVED_PEFT_CONFIG_NAME)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.resolved_dict(n_layer=n_layer, trainable=trainable, total=total),
                handle,
                sort_keys=False,
            )
        return path


@dataclass(frozen=True)
class HalfTuningConfig:
    """Mamba-LoRA backbone plus an unconditionally fully-trainable decoder."""

    format_version: int = FORMAT_VERSION
    method: str = "half"
    half: UniformOptimizationConfig = UniformOptimizationConfig()
    mamba_lora: HalfMambaLoRAConfig = HalfMambaLoRAConfig()
    decoder: HalfDecoderFinetuningConfig = HalfDecoderFinetuningConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()

    @property
    def optimization(self) -> UniformOptimizationConfig:
        return self.half

    def __post_init__(self) -> None:
        nested = (
            ("half", UniformOptimizationConfig),
            ("mamba_lora", HalfMambaLoRAConfig),
            ("decoder", HalfDecoderFinetuningConfig),
            ("loss", LossConfig),
            ("training", TrainingConfig),
            ("checkpoint", CheckpointConfig),
        )
        for name, cls in nested:
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, _strict_dataclass(cls, value, name))
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported half-tuning config format_version={self.format_version}; "
                f"expected {FORMAT_VERSION}."
            )
        if self.method != "half":
            raise ValueError("HalfTuningConfig.method must be 'half'.")
        if not self.checkpoint.save_peft_only or not self.checkpoint.save_full_inference_model:
            raise ValueError(
                "Half-tuning checkpoints require save_peft_only=true and "
                "save_full_inference_model=true."
            )
        if self.checkpoint.save_trainable_bundle:
            raise ValueError(
                "Half-tuning config must set checkpoint.save_trainable_bundle=false."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HalfTuningConfig":
        forbidden = sorted(set(data) & {"backbone_ffn_lora", "full"})
        if forbidden:
            raise ValueError(
                f"Half-tuning config must not expose strategy blocks: {forbidden}."
            )
        payload = dict(data)
        mamba_block = payload.get("mamba_lora")
        if isinstance(mamba_block, Mapping):
            forbidden_mamba = sorted(
                set(mamba_block)
                & {"layer_selection", "target_modules", "learning_rate", "weight_decay"}
            )
            if forbidden_mamba:
                if set(forbidden_mamba) & {"learning_rate", "weight_decay"}:
                    raise ValueError(
                        "Half-tuning module-level optimizer fields "
                        f"{forbidden_mamba} were removed; set learning_rate and "
                        "weight_decay in the top-level half block."
                    )
                raise ValueError(
                    "Half-tuning mamba_lora must not expose fixed scope fields "
                    f"{forbidden_mamba}; injection is fixed to all 16 BiMamba "
                    "in_proj/out_proj modules."
                )
        decoder_block = payload.get("decoder")
        if isinstance(decoder_block, Mapping):
            legacy_decoder_optimizer = sorted(
                key
                for key in decoder_block
                if key.endswith("_learning_rate") or key == "weight_decay"
            )
            if legacy_decoder_optimizer:
                raise ValueError(
                    "Half-tuning decoder module-level optimizer fields "
                    f"{legacy_decoder_optimizer} were removed; set learning_rate "
                    "and weight_decay in the top-level half block."
                )
        return _strict_dataclass(cls, payload, "root")

    @classmethod
    def from_yaml(cls, path: str) -> "HalfTuningConfig":
        config_path = os.path.join(path, HALF_CONFIG_NAME) if os.path.isdir(path) else path
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data is None:
            raise ValueError(f"Empty half-tuning configuration: {config_path}")
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str) -> str:
        output_path = (
            os.path.join(path, HALF_CONFIG_NAME) if os.path.isdir(path) else path
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)
        return output_path

    def resolve_lora_layers(self, n_layer: int) -> Tuple[int, ...]:
        return resolve_all_mamba_lora_layers(n_layer)

    def resolved_dict(self, *, n_layer: int, trainable: int, total: int) -> Dict[str, Any]:
        data = self.to_dict()
        mamba_lora = self.mamba_lora.to_resolved_dict(n_layer=n_layer)
        decoder = asdict(self.decoder)
        # YAML/user-facing mamba_lora omits fixed scope fields; provenance dumps
        # must still record the immutable full-layer in_proj/out_proj recipe.
        data["strategy"] = {
            "optimizer": asdict(self.half),
            "mamba_lora": mamba_lora,
            "backbone_feed_forward": {
                "trainable": False,
                "lora": False,
                "scope": "all_dense_ffn_moe_experts_and_routers",
            },
            "decoder": decoder,
            "decoder_train_flags": self.decoder.enabled_module_flags(),
            "decoder_transformer_lora": False,
        }
        data["runtime"] = {
            "method": self.method,
            "model_n_layer": int(n_layer),
            "resolved_lora_layers": list(self.resolve_lora_layers(n_layer)),
            "trainable_parameters": int(trainable),
            "total_parameters": int(total),
        }
        return data

    def save_resolved_yaml(
        self, output_dir: str, *, n_layer: int, trainable: int, total: int
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, RESOLVED_HALF_CONFIG_NAME)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.resolved_dict(n_layer=n_layer, trainable=trainable, total=total),
                handle,
                sort_keys=False,
            )
        return path


@dataclass(frozen=True)
class FullTuningConfig:
    format_version: int = FORMAT_VERSION
    method: str = "full"
    full: FullTuningBodyConfig = FullTuningBodyConfig()
    loss: LossConfig = LossConfig()
    # Full tuning uses a shorter warmup than PEFT and half tuning.
    training: TrainingConfig = TrainingConfig(warmup_ratio=0.025)
    checkpoint: CheckpointConfig = CheckpointConfig(
        save_peft_only=False,
        save_full_inference_model=True,
        safe_serialization=False,
        save_last=False,
        save_trainable_bundle=True,
    )

    @property
    def optimization(self) -> FullTuningBodyConfig:
        return self.full

    def __post_init__(self) -> None:
        nested = (
            ("full", FullTuningBodyConfig),
            ("loss", LossConfig),
            ("training", TrainingConfig),
            ("checkpoint", CheckpointConfig),
        )
        for name, cls in nested:
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, _strict_dataclass(cls, value, name))
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported full-tuning config format_version={self.format_version}; "
                f"expected {FORMAT_VERSION}."
            )
        if self.method != "full":
            raise ValueError("FullTuningConfig.method must be 'full'.")
        if self.checkpoint.save_peft_only:
            raise ValueError(
                "Full-tuning config must set checkpoint.save_peft_only=false."
            )
        if not self.checkpoint.save_full_inference_model:
            raise ValueError(
                "Full-tuning requires checkpoint.save_full_inference_model=true."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FullTuningConfig":
        if "mamba_lora" in data:
            raise ValueError(
                "Full-tuning config must not contain a 'mamba_lora' block; "
                "use method: peft with PeftTuningConfig."
            )
        if "decoder" in data:
            raise ValueError(
                "Full-tuning config must not contain a top-level 'decoder' block; "
                "naive full FT trains all parameters with full.learning_rate."
            )
        return _strict_dataclass(cls, data, "root")

    @classmethod
    def from_yaml(cls, path: str) -> "FullTuningConfig":
        config_path = os.path.join(path, FULL_CONFIG_NAME) if os.path.isdir(path) else path
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data is None:
            raise ValueError(f"Empty full-tuning configuration: {config_path}")
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str) -> str:
        output_path = (
            os.path.join(path, FULL_CONFIG_NAME) if os.path.isdir(path) else path
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)
        return output_path

    def resolved_dict(
        self,
        *,
        n_layer: int,
        trainable: int,
        total: int,
        group_counts: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        data = self.to_dict()
        runtime: Dict[str, Any] = {
            "method": self.method,
            "model_n_layer": int(n_layer),
            "learning_rate": float(self.full.learning_rate),
            "weight_decay": float(self.full.weight_decay),
            "trainable_parameters": int(trainable),
            "total_parameters": int(total),
            "trainable_scope": "all_parameters",
        }
        if group_counts is not None:
            runtime["trainable_group_counts"] = {
                key: int(value) for key, value in group_counts.items()
            }
        data["runtime"] = runtime
        return data

    def save_resolved_yaml(
        self,
        output_dir: str,
        *,
        n_layer: int,
        trainable: int,
        total: int,
        group_counts: Optional[Mapping[str, int]] = None,
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, RESOLVED_FULL_CONFIG_NAME)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.resolved_dict(
                    n_layer=n_layer,
                    trainable=trainable,
                    total=total,
                    group_counts=group_counts,
                ),
                handle,
                sort_keys=False,
            )
        return path



def _load_yaml_mapping(path: str) -> Dict[str, Any]:
    config_path = path
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, name)
            for name in (PEFT_CONFIG_NAME, HALF_CONFIG_NAME, FULL_CONFIG_NAME)
            if os.path.isfile(os.path.join(path, name))
        ]
        if len(candidates) == 1:
            config_path = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                f"Directory {path} contains multiple tuning configs; "
                "pass an explicit file path."
            )
        else:
            raise FileNotFoundError(
                f"No {PEFT_CONFIG_NAME}, {HALF_CONFIG_NAME}, or "
                f"{FULL_CONFIG_NAME} under {path}."
            )
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ValueError(f"Empty tuning configuration: {config_path}")
    if not isinstance(data, Mapping):
        raise TypeError(f"Tuning configuration root must be a mapping: {config_path}")
    return dict(data)


def load_tuning_config(path: str) -> TuningConfig:
    """Load a PEFT, half, or full-tuning YAML by inspecting ``method``."""

    data = _load_yaml_mapping(path)
    method = data.get("method", None)
    if method == "peft":
        return PeftTuningConfig.from_dict(data)
    if method == "half":
        return HalfTuningConfig.from_dict(data)
    if method == "full":
        return FullTuningConfig.from_dict(data)
    raise ValueError(
        f"Unsupported or missing method={method!r}; expected "
        "'peft', 'half', or 'full'."
    )
