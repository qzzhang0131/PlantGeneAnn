"""Tuning wrappers for loss-free GeneAnn models (PEFT and full-parameter)."""

from __future__ import annotations

import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from transformers import AutoModel

from .config import (
    FULL_CONFIG_NAME,
    PEFT_CONFIG_NAME,
    FullTuningConfig,
    HalfTuningConfig,
    PeftTuningConfig,
    load_tuning_config,
)
from .lora import (
    effective_state_dict_for_unparametrized_model,
    inject_linear_lora,
    inject_mamba_lora,
    iter_lora_parameters,
    lora_parameter_count,
)
from .loss import GeneAnnTrainingCriterion, TuningOutput


PEFT_WEIGHTS_NAME = "peft_model.bin"
FULL_WEIGHTS_NAME = "pytorch_model.bin"
TRAINABLE_PARAMETER_NAMES_NAME = "trainable_parameter_names.json"
REMOTE_CODE_FILES = (
    "configuration_caduceus_ph.py",
    "modeling_caduceus_moe.py",
    "modeling_segment_caduceus_v2.py",
    "tokenization_caduceus.py",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

@dataclass(frozen=True)
class ParameterSummary:
    total: int
    trainable: int
    lora: int
    decoder: int
    embeddings: int = 0
    backbone: int = 0
    extra: Dict[str, int] = field(default_factory=dict)

    @property
    def trainable_fraction(self) -> float:
        return self.trainable / self.total if self.total else 0.0

    def to_dict(self) -> Dict[str, float]:
        payload = {
            "total": self.total,
            "trainable": self.trainable,
            "lora": self.lora,
            "decoder": self.decoder,
            "embeddings": self.embeddings,
            "backbone": self.backbone,
            "trainable_fraction": self.trainable_fraction,
            "trainable_percent": 100.0 * self.trainable_fraction,
        }
        payload.update(self.extra)
        return payload


def _set_module_trainable(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


def _copy_remote_code(source_model_path: str, output_dir: str) -> None:
    for filename in REMOTE_CODE_FILES:
        source = os.path.join(source_model_path, filename)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(output_dir, filename))


def _state_dict_without_shared_storage(module: nn.Module) -> OrderedDict:
    """Materialize only duplicate state-dict aliases for safetensors export.

    GeneAnn ties forward/reverse Mamba projection parameters. Hugging Face cannot
    infer these custom ties and rejects safe serialization unless duplicate keys
    have distinct storage. Cloning only later aliases avoids a second full-model
    copy while preserving every ordinary HF state key.
    """

    output = OrderedDict()
    seen_storage = set()
    for name, tensor in module.state_dict().items():
        if not torch.is_tensor(tensor):
            output[name] = tensor
            continue
        storage_key = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )
        output[name] = tensor.clone() if storage_key in seen_storage else tensor
        seen_storage.add(storage_key)
    return output


def _validate_geneann_structure(base_model: nn.Module) -> None:
    if not bool(getattr(base_model.config, "inference_only", False)):
        raise TypeError("Tuning requires config.inference_only=true.")
    if hasattr(base_model, "loss_fn"):
        raise TypeError("The supplied base model contains loss_fn.")
    required = (
        "caduceus_ph.backbone.layers",
        "caduceus_ph.backbone.embeddings",
        "embeddings_decoder.up_conv_blocks",
        "embeddings_decoder.local_refine_block",
        "embeddings_decoder.norm_final",
        "prediction_head",
    )
    for path in required:
        value = base_model
        for component in path.split("."):
            if not hasattr(value, component):
                raise TypeError(f"Unsupported GeneAnn model: missing {path}.")
            value = getattr(value, component)
    decoder = base_model.embeddings_decoder
    if len(decoder.up_conv_blocks) == 0:
        raise ValueError("Decoder has no up blocks.")


def _split_decay_params(
    named_parameters: Iterable[Tuple[str, nn.Parameter]],
    *,
    seen: Set[int],
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim <= 1 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay


def _append_decay_groups(
    groups: List[dict],
    *,
    group_name: str,
    named_parameters: Iterable[Tuple[str, nn.Parameter]],
    lr: float,
    weight_decay: float,
    seen: Set[int],
) -> None:
    decay, no_decay = _split_decay_params(named_parameters, seen=seen)
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": lr,
                "weight_decay": weight_decay,
                "group_name": f"{group_name}_decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": lr,
                "weight_decay": 0.0,
                "group_name": f"{group_name}_no_decay",
            }
        )


class _TuningModelBase(nn.Module):
    """Shared forward / criterion utilities for PEFT and full wrappers."""

    base_model: nn.Module
    criterion: GeneAnnTrainingCriterion
    source_model_path: Optional[str]

    @property
    def config(self):
        return self.base_model.config

    def _backbone(self):
        return self.base_model.caduceus_ph.backbone

    def _decoder(self):
        return self.base_model.embeddings_decoder

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        attention_mask=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        loss = None
        if labels is not None:
            if attention_mask is None:
                raise ValueError("attention_mask loss mask is required with labels.")
            loss = self.criterion(outputs.logits, labels, attention_mask)
        elif attention_mask is not None:
            raise ValueError("labels are required when attention_mask is supplied.")
        if return_dict is False:
            result = (outputs.logits,)
            if output_hidden_states:
                result += (outputs.hidden_states,)
            return ((loss,) + result) if loss is not None else result
        return TuningOutput(
            loss=loss, logits=outputs.logits, hidden_states=outputs.hidden_states
        )

    def prepare_targets(self, labels, loss_mask, *, output_length):
        return self.criterion.prepare_targets(
            labels, loss_mask, output_length=output_length
        )

    def trainable_named_parameters(self):
        seen = set()
        for name, parameter in self.base_model.named_parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                yield name, parameter

    def _build_criterion(self, loss_cfg) -> GeneAnnTrainingCriterion:
        return GeneAnnTrainingCriterion(
            num_features=int(self.base_model.config.num_features),
            flank_length=int(self.base_model.config.flank_length),
            loss_cfg=loss_cfg,
        )


class MambaLoRATuningModel(_TuningModelBase):
    """Wrap GeneAnn with Mamba-LoRA and PEFT- or half-tuning decoder scope."""

    def __init__(
        self,
        base_model: nn.Module,
        peft_config: PeftTuningConfig | HalfTuningConfig,
        *,
        source_model_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.peft_config = peft_config
        self.tuning_config = peft_config
        self.source_model_path = (
            os.path.abspath(source_model_path) if source_model_path else None
        )
        _validate_geneann_structure(self.base_model)
        self.resolved_lora_layers = peft_config.resolve_lora_layers(
            int(base_model.config.n_layer)
        )
        lora = peft_config.mamba_lora
        self.lora_injections = inject_mamba_lora(
            self._backbone(),
            layers=self.resolved_lora_layers,
            target_modules=lora.target_modules,
            rank=lora.rank,
            alpha=lora.alpha,
            share_bidirectional=lora.share_bidirectional,
        )
        self.transformer_lora_injections = ()
        if isinstance(peft_config, PeftTuningConfig):
            transformer = peft_config.transformer_lora
            bridge = self._decoder().transformer_layers
            if not hasattr(bridge, "layers"):
                raise TypeError(
                    "PEFT Transformer-LoRA requires decoder.transformer_layers.layers."
                )
            target_paths = tuple(
                f"layers.{layer_index}.{target}"
                for layer_index in range(len(bridge.layers))
                for target in transformer.target_modules
            )
            self.transformer_lora_injections = inject_linear_lora(
                bridge,
                target_paths=target_paths,
                rank=transformer.rank,
                alpha=transformer.alpha,
            )
        self.criterion = self._build_criterion(peft_config.loss)
        self._configure_trainable_parameters()
        self._validate_parameter_count()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        peft_config: Optional[PeftTuningConfig | HalfTuningConfig] = None,
        *,
        peft_path: Optional[str] = None,
        trust_remote_code: bool = True,
        local_files_only: bool = True,
        torch_dtype=None,
        **model_kwargs,
    ) -> "MambaLoRATuningModel":
        if peft_config is None:
            if peft_path is None:
                raise ValueError("Provide peft_config or peft_path.")
            peft_config = load_tuning_config(peft_path)
        if not isinstance(peft_config, (PeftTuningConfig, HalfTuningConfig)):
            raise TypeError(
                "MambaLoRATuningModel requires PeftTuningConfig or HalfTuningConfig."
            )
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        base_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            **model_kwargs,
        )
        wrapped = cls(base_model, peft_config, source_model_path=model_path)
        if peft_path is not None:
            weights_path = (
                os.path.join(peft_path, PEFT_WEIGHTS_NAME)
                if os.path.isdir(peft_path)
                else None
            )
            if weights_path and os.path.isfile(weights_path):
                wrapped.load_peft(peft_path)
        return wrapped

    def _iter_decoder_train_modules(self):
        """Yield (group_name, module) for the resolved decoder scope."""
        cfg = self.peft_config.decoder
        flags = cfg.enabled_module_flags()
        decoder = self._decoder()
        specs = []
        if flags["stem_block"] and hasattr(decoder, "stem_block"):
            specs.append(("decoder_stem_block", decoder.stem_block))
        if flags["down_conv_blocks"] and hasattr(decoder, "down_conv_blocks"):
            specs.append(("decoder_down_conv_blocks", decoder.down_conv_blocks))
        if flags["transformer_bridge"] and hasattr(decoder, "transformer_layers"):
            specs.append(("decoder_transformer_bridge", decoder.transformer_layers))
        if flags["up_conv_blocks"] and hasattr(decoder, "up_conv_blocks"):
            specs.append(("decoder_up_conv_blocks", decoder.up_conv_blocks))
        elif flags["last_up_conv_block"] and hasattr(decoder, "up_conv_blocks"):
            if len(decoder.up_conv_blocks) == 0:
                raise ValueError("Decoder has no up_conv_blocks to train.")
            specs.append(("decoder_last_up_conv_block", decoder.up_conv_blocks[-1]))
        if flags["local_refine_block"] and hasattr(decoder, "local_refine_block"):
            specs.append(("decoder_local_refine_block", decoder.local_refine_block))
        if flags["final_norm"] and hasattr(decoder, "norm_final"):
            specs.append(("decoder_final_norm", decoder.norm_final))
        if flags["prediction_head"]:
            specs.append(("prediction_head", self.base_model.prediction_head))
        return specs

    def _configure_trainable_parameters(self) -> None:
        # Start from a deny-by-default policy. Only injected Mamba/decoder-
        # Transformer LoRA tensors and the explicitly selected decoder/head
        # modules may become trainable. In particular, every original backbone
        # dense FFN, MoE expert, and MoE router tensor remains frozen.
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        for _, parameter in iter_lora_parameters(self.base_model):
            parameter.requires_grad = True
        for _, module in self._iter_decoder_train_modules():
            _set_module_trainable(module, True)
        self._validate_frozen_backbone_feed_forward()

    def _validate_frozen_backbone_feed_forward(self) -> None:
        """Enforce the immutable PEFT/half backbone feed-forward freeze policy."""
        trainable = []
        parametrized = []
        for name, parameter in self._backbone().named_parameters():
            if ".feed_forward." not in f".{name}":
                continue
            if parameter.requires_grad:
                trainable.append(name)
            if "parametrizations." in name or name.endswith((".lora_A", ".lora_B")):
                parametrized.append(name)
        if trainable or parametrized:
            raise RuntimeError(
                "Backbone dense FFN, MoE expert, and MoE router parameters must "
                "remain frozen and must not contain LoRA parametrizations; "
                f"trainable={trainable[:5]}, parametrized={parametrized[:5]}."
            )

    def _decoder_trainable_count(self) -> int:
        lora_ids = {id(p) for _, p in iter_lora_parameters(self.base_model)}
        return sum(
            p.numel()
            for p in self.base_model.parameters()
            if p.requires_grad and id(p) not in lora_ids
        )

    def decoder_module_counts(self) -> Dict[str, int]:
        """Per enabled decoder module parameter counts (for resolved YAML)."""
        counts: Dict[str, int] = {}
        for group_name, module in self._iter_decoder_train_modules():
            counts[group_name] = sum(p.numel() for p in module.parameters())
        counts["decoder_total"] = self._decoder_trainable_count()
        return counts

    def _validate_parameter_count(self) -> None:
        head_ok = any(p.requires_grad for p in self.base_model.prediction_head.parameters())
        if not head_ok:
            raise RuntimeError("prediction_head parameters are not trainable.")
        if isinstance(self.peft_config, HalfTuningConfig):
            frozen_decoder = [
                name
                for name, parameter in self._decoder().named_parameters()
                if not parameter.requires_grad
            ]
            frozen_head = [
                name
                for name, parameter in self.base_model.prediction_head.named_parameters()
                if not parameter.requires_grad
            ]
            if frozen_decoder or frozen_head:
                raise RuntimeError(
                    "Half tuning requires every decoder/head parameter trainable; "
                    f"frozen_decoder={frozen_decoder[:5]}, frozen_head={frozen_head[:5]}."
                )
        lora = self.peft_config.mamba_lora
        if not self.resolved_lora_layers:
            raise RuntimeError("No LoRA layers resolved for PEFT tuning.")
        per_layer = sum(
            lora.rank
            * (
                getattr(
                    self._backbone()
                    .layers[self.resolved_lora_layers[0]]
                    .mixer.mixer.mamba_fwd,
                    target,
                ).weight.shape[0]
                + getattr(
                    self._backbone()
                    .layers[self.resolved_lora_layers[0]]
                    .mixer.mixer.mamba_fwd,
                    target,
                ).weight.shape[1]
            )
            for target in lora.target_modules
        )
        multiplier = 1 if lora.share_bidirectional else 2
        expected = per_layer * len(self.resolved_lora_layers) * multiplier
        actual = sum(
            parameter.numel()
            for name, parameter in iter_lora_parameters(self._backbone())
            if ".mixer.mixer.mamba_" in name
        )
        if actual != expected:
            raise RuntimeError(
                f"Expected {expected:,} Mamba-LoRA parameters, got {actual:,}."
            )
        self._validate_frozen_backbone_feed_forward()

        if isinstance(self.peft_config, PeftTuningConfig):
            transformer = self.peft_config.transformer_lora
            bridge = self._decoder().transformer_layers
            expected_transformer = sum(
                transformer.rank * (module.weight.shape[0] + module.weight.shape[1])
                for injection in self.transformer_lora_injections
                for module in [
                    bridge.get_submodule(
                        ".".join(
                            part
                            for part in (injection.module_path, injection.target_name)
                            if part
                        )
                    )
                ]
            )
            actual_transformer = lora_parameter_count(bridge)
            if actual_transformer != expected_transformer:
                raise RuntimeError(
                    "Transformer-LoRA parameter count mismatch: "
                    f"expected={expected_transformer:,}, actual={actual_transformer:,}."
                )

    def parameter_summary(self) -> ParameterSummary:
        total = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        lora = lora_parameter_count(self.base_model)
        return ParameterSummary(total, trainable, lora, trainable - lora)

    def get_optimizer_grouped_parameters(self):
        cfg = self.peft_config
        optimizer_cfg = cfg.optimization
        learning_rate = float(optimizer_cfg.learning_rate)
        weight_decay = float(optimizer_cfg.weight_decay)
        groups = []
        seen: Set[int] = set()

        mamba_lora_params = []
        for name, parameter in iter_lora_parameters(self._backbone()):
            if ".mixer.mixer.mamba_" not in name:
                continue
            if id(parameter) not in seen:
                mamba_lora_params.append(parameter)
        if not mamba_lora_params:
            raise RuntimeError("No Mamba-LoRA parameters found for optimizer grouping.")
        _append_decay_groups(
            groups,
            group_name="mamba_lora",
            named_parameters=[
                (name, parameter)
                for name, parameter in iter_lora_parameters(self._backbone())
                if ".mixer.mixer.mamba_" in name
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
            seen=seen,
        )

        if isinstance(cfg, PeftTuningConfig):
            transformer_lora_params = []
            for _, parameter in iter_lora_parameters(
                self._decoder().transformer_layers
            ):
                if id(parameter) not in seen:
                    transformer_lora_params.append(parameter)
            if not transformer_lora_params:
                raise RuntimeError(
                    "No decoder Transformer-LoRA parameters found for optimizer grouping."
                )
            _append_decay_groups(
                groups,
                group_name="transformer_lora",
                named_parameters=list(
                    iter_lora_parameters(self._decoder().transformer_layers)
                ),
                lr=learning_rate,
                weight_decay=weight_decay,
                seen=seen,
            )

        for group_name, module in self._iter_decoder_train_modules():
            _append_decay_groups(
                groups,
                group_name=group_name,
                named_parameters=module.named_parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                seen=seen,
            )
        # Recheck immediately before optimizer construction so later accidental
        # unfreezing cannot silently introduce backbone FFN/MoE/router tensors.
        self._validate_frozen_backbone_feed_forward()
        expected = {id(p) for _, p in self.trainable_named_parameters()}
        if seen != expected:
            raise RuntimeError(
                f"Optimizer grouping mismatch: missing={len(expected - seen)}, "
                f"extra={len(seen - expected)}."
            )
        return groups

    def save_peft(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.peft_config.save_yaml(output_dir)
        names = [name for name, _ in self.trainable_named_parameters()]
        state = OrderedDict(
            (name, parameter.detach().cpu())
            for name, parameter in self.trainable_named_parameters()
        )
        torch.save(state, os.path.join(output_dir, PEFT_WEIGHTS_NAME))
        with open(
            os.path.join(output_dir, TRAINABLE_PARAMETER_NAMES_NAME),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(names, handle, indent=2)
            handle.write("\n")

    def load_peft(self, peft_path: str) -> None:
        weights_path = (
            os.path.join(peft_path, PEFT_WEIGHTS_NAME)
            if os.path.isdir(peft_path)
            else peft_path
        )
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location="cpu")
        expected = {name for name, _ in self.trainable_named_parameters()}
        if set(state) != expected:
            raise RuntimeError(
                f"PEFT checkpoint mismatch: missing={sorted(expected - set(state))[:10]}, "
                f"unexpected={sorted(set(state) - expected)[:10]}."
            )
        result = self.base_model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected PEFT keys: {result.unexpected_keys}")
        # ``strict=False`` is intentional because a PEFT bundle contains only
        # trainable tensors. Every omitted key must therefore belong to a frozen
        # parameter/buffer; a missing trainable key would indicate checkpoint
        # corruption or a strategy mismatch.
        missing_trainable = sorted(expected & set(result.missing_keys))
        if missing_trainable:
            raise RuntimeError(
                f"Missing trainable PEFT keys after load: {missing_trainable[:10]}"
            )

    def save_pretrained(self, output_dir: str, *, safe_serialization: bool = False) -> None:
        """Save a merged, ordinary loss-free SegmentCaduceusPh checkpoint."""
        if not self.source_model_path or not os.path.isdir(self.source_model_path):
            raise FileNotFoundError(
                f"Source model directory unavailable: {self.source_model_path!r}"
            )
        os.makedirs(output_dir, exist_ok=True)
        reference = AutoModel.from_pretrained(
            self.source_model_path, trust_remote_code=True, local_files_only=True
        )
        merged = effective_state_dict_for_unparametrized_model(self.base_model, reference)
        load_result = reference.load_state_dict(merged, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(f"Merged state load failed: {load_result}")
        reference.save_pretrained(
            output_dir,
            safe_serialization=safe_serialization,
            state_dict=(
                _state_dict_without_shared_storage(reference)
                if safe_serialization
                else None
            ),
        )
        _copy_remote_code(self.source_model_path, output_dir)
        self.peft_config.save_yaml(output_dir)

    def save_checkpoint_bundle(
        self,
        checkpoint_dir: str,
        *,
        safe_serialization: bool = False,
    ) -> None:
        """PEFT layout: peft/ + merged model/."""
        peft_dir = os.path.join(checkpoint_dir, "peft")
        model_dir = os.path.join(checkpoint_dir, "model")
        self.save_peft(peft_dir)
        self.save_pretrained(model_dir, safe_serialization=safe_serialization)


class FullTuningModel(_TuningModelBase):
    """Naive full-parameter fine-tuning: train all weights with one LR/WD."""

    def __init__(
        self,
        base_model: nn.Module,
        full_config: FullTuningConfig,
        *,
        source_model_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.full_config = full_config
        self.tuning_config = full_config
        self.source_model_path = (
            os.path.abspath(source_model_path) if source_model_path else None
        )
        _validate_geneann_structure(self.base_model)
        self.criterion = self._build_criterion(full_config.loss)
        self._configure_trainable_parameters()
        self._validate_trainable_scope()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        full_config: Optional[FullTuningConfig] = None,
        *,
        resume_path: Optional[str] = None,
        trust_remote_code: bool = True,
        local_files_only: bool = True,
        torch_dtype=None,
        **model_kwargs,
    ) -> "FullTuningModel":
        if full_config is None:
            if resume_path is None:
                raise ValueError("Provide full_config or resume_path.")
            full_config = FullTuningConfig.from_yaml(resume_path)
        if not isinstance(full_config, FullTuningConfig):
            raise TypeError("FullTuningModel requires FullTuningConfig.")
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        base_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            **model_kwargs,
        )
        wrapped = cls(base_model, full_config, source_model_path=model_path)
        if resume_path is not None:
            wrapped.load_training_bundle(resume_path)
        return wrapped

    def _configure_trainable_parameters(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad = True

    def _validate_trainable_scope(self) -> None:
        trainable = list(self.trainable_named_parameters())
        if not trainable:
            raise RuntimeError("Full-tuning configured zero trainable parameters.")
        total = sum(1 for _ in self.base_model.parameters())
        trainable_count = sum(1 for p in self.base_model.parameters() if p.requires_grad)
        if trainable_count != total:
            raise RuntimeError(
                f"Naive full FT requires all parameters trainable; "
                f"got {trainable_count}/{total}."
            )
        for name, _ in self.base_model.named_parameters():
            if "lora_A" in name or "lora_B" in name or "parametrizations" in name:
                raise RuntimeError(
                    f"Full-tuning model must not contain LoRA/parametrization keys: {name}"
                )

    def parameter_group_counts(self) -> Dict[str, int]:
        decay = 0
        no_decay = 0
        for name, parameter in self.trainable_named_parameters():
            if parameter.ndim <= 1 or name.endswith("bias"):
                no_decay += parameter.numel()
            else:
                decay += parameter.numel()
        return {
            "decay": decay,
            "no_decay": no_decay,
            "trainable": decay + no_decay,
        }

    def parameter_summary(self) -> ParameterSummary:
        total = sum(p.numel() for p in self.base_model.parameters())
        trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        counts = self.parameter_group_counts()
        backbone = sum(p.numel() for p in self._backbone().parameters())
        embeddings = sum(
            p.numel() for p in self._backbone().embeddings.parameters()
        )
        decoder = sum(p.numel() for p in self._decoder().parameters()) + sum(
            p.numel() for p in self.base_model.prediction_head.parameters()
        )
        return ParameterSummary(
            total=total,
            trainable=trainable,
            lora=0,
            decoder=decoder,
            embeddings=embeddings,
            backbone=backbone,
            extra={f"group_{k}": int(v) for k, v in counts.items()},
        )

    def get_optimizer_grouped_parameters(self):
        """Single global LR; standard AdamW decay / no_decay split."""
        lr = float(self.full_config.optimization.learning_rate)
        weight_decay = float(self.full_config.optimization.weight_decay)
        decay: List[nn.Parameter] = []
        no_decay: List[nn.Parameter] = []
        seen: Set[int] = set()
        for name, parameter in self.base_model.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.ndim <= 1 or name.endswith("bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        groups = []
        if decay:
            groups.append(
                {
                    "params": decay,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "group_name": "all_decay",
                }
            )
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "group_name": "all_no_decay",
                }
            )
        expected = {id(p) for _, p in self.trainable_named_parameters()}
        if seen != expected:
            raise RuntimeError(
                f"Optimizer grouping mismatch: missing={len(expected - seen)}, "
                f"extra={len(seen - expected)}."
            )
        if not groups:
            raise RuntimeError("No optimizer parameter groups were constructed.")
        return groups

    def save_pretrained(self, output_dir: str, *, safe_serialization: bool = False) -> None:
        """Save an ordinary loss-free SegmentCaduceusPh checkpoint."""
        if not self.source_model_path or not os.path.isdir(self.source_model_path):
            raise FileNotFoundError(
                f"Source model directory unavailable: {self.source_model_path!r}"
            )
        os.makedirs(output_dir, exist_ok=True)
        forbidden = [
            key
            for key in self.base_model.state_dict()
            if "lora_A" in key or "lora_B" in key or "parametrizations" in key
        ]
        if forbidden:
            raise RuntimeError(
                f"Refusing to save full-tuning model with LoRA keys: {forbidden[:5]}"
            )
        self.base_model.save_pretrained(
            output_dir,
            safe_serialization=safe_serialization,
            state_dict=(
                _state_dict_without_shared_storage(self.base_model)
                if safe_serialization
                else None
            ),
        )
        _copy_remote_code(self.source_model_path, output_dir)
        self.full_config.save_yaml(os.path.join(output_dir, FULL_CONFIG_NAME))

    def save_training_bundle(self, output_dir: str) -> None:
        """Save a resume bundle: full weights + trainable names + config."""
        os.makedirs(output_dir, exist_ok=True)
        self.full_config.save_yaml(output_dir)
        state = OrderedDict(
            (name, tensor.detach().cpu())
            for name, tensor in self.base_model.state_dict().items()
        )
        torch.save(state, os.path.join(output_dir, FULL_WEIGHTS_NAME))
        names = [name for name, _ in self.trainable_named_parameters()]
        with open(
            os.path.join(output_dir, TRAINABLE_PARAMETER_NAMES_NAME),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(names, handle, indent=2)
            handle.write("\n")
        if self.source_model_path:
            _copy_remote_code(self.source_model_path, output_dir)

    def load_training_bundle(self, bundle_path: str) -> None:
        """Load weights from best/full or best/model style directories."""
        if os.path.isdir(bundle_path):
            candidates = [
                os.path.join(bundle_path, FULL_WEIGHTS_NAME),
                os.path.join(bundle_path, "model.safetensors"),
            ]
            weights_path = next((path for path in candidates if os.path.isfile(path)), None)
            if weights_path is None:
                nested = os.path.join(bundle_path, "model", FULL_WEIGHTS_NAME)
                if os.path.isfile(nested):
                    weights_path = nested
            if weights_path is None:
                raise FileNotFoundError(
                    f"No full-tuning weights under {bundle_path}."
                )
        else:
            weights_path = bundle_path

        if weights_path.endswith(".safetensors"):
            raise ValueError(
                "safetensors resume is not supported in this path; use pytorch_model.bin."
            )
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location="cpu")
        result = self.base_model.load_state_dict(state, strict=False)
        unexpected = [
            key
            for key in result.unexpected_keys
            if "lora_" not in key and "parametrizations" not in key
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected keys in full bundle: {unexpected[:10]}")
        if result.missing_keys:
            raise RuntimeError(
                f"Missing keys in full bundle: {result.missing_keys[:10]}"
            )
        self._configure_trainable_parameters()

    def save_checkpoint_bundle(
        self,
        checkpoint_dir: str,
        *,
        safe_serialization: bool = False,
    ) -> None:
        """Full layout: model/ (+ optional full/ resume bundle)."""
        model_dir = os.path.join(checkpoint_dir, "model")
        self.save_pretrained(model_dir, safe_serialization=safe_serialization)
        if self.full_config.checkpoint.save_trainable_bundle:
            full_dir = os.path.join(checkpoint_dir, "full")
            self.save_training_bundle(full_dir)
