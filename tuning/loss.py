"""Training-only label preparation and loss for inference-only GeneAnn models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers.utils import ModelOutput


NUM_CLASSES = 15
LOSS_MASKED_FOCAL_F1 = "masked_focal_f1"
LOSS_TIBERIUS_CCE_F1 = "tiberius_cce_f1"
SUPPORTED_LOSS_NAMES = (LOSS_MASKED_FOCAL_F1, LOSS_TIBERIUS_CCE_F1)


@dataclass
class TuningOutput(ModelOutput):
    """Training output produced outside the inference-only base model."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[torch.FloatTensor] = None


# Backward-compatible alias used by existing imports and call sites.
PeftTuningOutput = TuningOutput


def crop_strand_labels_and_masks(
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    flank_length: int,
    output_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Center-crop two-strand labels and masks to the model output interval.

    The inference-only model crops ``flank_length`` positions from both sides of
    its logits. Training labels have shape ``[B, 2, input_sequence_length]`` and
    must receive exactly the same crop before strand-specific loss computation.
    ``output_length`` is checked against the result to catch geometry drift.
    """

    if labels.ndim != 3 or labels.shape[1] != 2:
        raise ValueError(
            f"labels must have shape [B, 2, L], got {tuple(labels.shape)}."
        )
    if tuple(loss_mask.shape) != tuple(labels.shape):
        raise ValueError(
            "loss_mask must have the same [B, 2, L] shape as labels, got "
            f"{tuple(loss_mask.shape)} and {tuple(labels.shape)}."
        )
    flank_length = int(flank_length)
    if flank_length < 0:
        raise ValueError(f"flank_length must be non-negative, got {flank_length}.")

    sequence_length = int(labels.shape[-1])
    if flank_length == 0:
        cropped_labels = labels
        cropped_mask = loss_mask
    else:
        if 2 * flank_length >= sequence_length:
            raise ValueError(
                f"flank_length={flank_length} is too large for label length "
                f"{sequence_length}; a non-empty center interval is required."
            )
        cropped_labels = labels[..., flank_length : sequence_length - flank_length]
        cropped_mask = loss_mask[..., flank_length : sequence_length - flank_length]

    if output_length is not None and cropped_labels.shape[-1] != int(output_length):
        raise ValueError(
            "Cropped label/model output length mismatch: labels become "
            f"{cropped_labels.shape[-1]} after removing {flank_length} positions "
            f"from each side, but model logits have length {int(output_length)}."
        )
    return cropped_labels, cropped_mask


class MaskedFocalF1Loss(nn.Module):
    """Original PlantGeneAnn hierarchical masked focal-F1 loss."""

    NUM_CLASSES = NUM_CLASSES
    INTRON_START = 1
    CDS_START = 4
    FOCAL_CLASS_WEIGHTS = (
        1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        2.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0,
    )

    def __init__(
        self,
        gamma: float = 2.0,
        state_f1_weight: float = 0.5,
        intron_f1_weight: float = 0.5,
        cds_f1_weight: float = 0.5,
        gate_temperature: float = 0.20,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}.")
        if min(state_f1_weight, intron_f1_weight, cds_f1_weight) < 0:
            raise ValueError("F1 loss weights must be non-negative.")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.gamma = float(gamma)
        self.state_f1_weight = float(state_f1_weight)
        self.intron_f1_weight = float(intron_f1_weight)
        self.cds_f1_weight = float(cds_f1_weight)
        self.gate_temperature = float(gate_temperature)
        self.epsilon = float(epsilon)
        self.register_buffer(
            "focal_class_weights",
            torch.tensor(self.FOCAL_CLASS_WEIGHTS, dtype=torch.float32),
            persistent=False,
        )

    def _group_f1_loss(
        self,
        group_probs: torch.Tensor,
        group_true: torch.Tensor,
        loss_mask: torch.Tensor,
        normalized_lengths: torch.Tensor,
        valid_sample_weights: torch.Tensor,
        valid_sample_count: torch.Tensor,
    ) -> torch.Tensor:
        intersection = (group_probs * group_true * loss_mask).sum(dim=1)
        predicted_positives = (group_probs * loss_mask).sum(dim=1)
        possible_positives = (group_true * loss_mask).sum(dim=1)
        present_group_loss = 1.0 - (
            2.0 * intersection + self.epsilon
        ) / (predicted_positives + possible_positives + self.epsilon)
        absent_group_loss = predicted_positives / normalized_lengths
        group_loss_per_sample = torch.where(
            possible_positives > 0,
            present_group_loss,
            absent_group_loss,
        )
        return (
            group_loss_per_sample * valid_sample_weights
        ).sum() / valid_sample_count

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(
                f"logits must have shape [B, L, C], got {tuple(logits.shape)}."
            )
        if tuple(labels.shape) != tuple(logits.shape[:2]):
            raise ValueError(
                f"labels shape {tuple(labels.shape)} does not match logits "
                f"{tuple(logits.shape)}."
            )
        if tuple(loss_mask.shape) != tuple(labels.shape):
            raise ValueError("loss_mask must have the same shape as labels.")
        if logits.shape[-1] != self.NUM_CLASSES:
            raise ValueError(
                f"Expected {self.NUM_CLASSES} classes, got {logits.shape[-1]}."
            )

        labels = labels.to(device=logits.device, dtype=torch.long)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits_fp32 = logits.float()
            loss_mask_fp32 = loss_mask.to(
                device=logits.device, dtype=torch.float32
            )
            safe_labels = labels.masked_fill(loss_mask_fp32 <= 0, 0)
            if torch.any((safe_labels < 0) | (safe_labels >= self.NUM_CLASSES)):
                raise ValueError(
                    f"Unmasked labels must be in [0, {self.NUM_CLASSES - 1}]."
                )

            valid_lengths = loss_mask_fp32.sum(dim=1)
            valid_samples = valid_lengths > 0
            valid_sample_weights = valid_samples.to(dtype=torch.float32)
            valid_sample_count = valid_sample_weights.sum().clamp_min(1.0)
            normalized_lengths = valid_lengths.clamp_min(self.epsilon)

            log_probs = F.log_softmax(logits_fp32, dim=-1)
            true_log_probs = log_probs.gather(
                dim=-1, index=safe_labels.unsqueeze(-1)
            ).squeeze(-1)
            true_probs = true_log_probs.exp()
            focal_per_position = -(
                (1.0 - true_probs) ** self.gamma
            ) * true_log_probs
            class_weights = self.focal_class_weights.to(device=logits.device)
            focal_position_weights = (
                loss_mask_fp32 * class_weights[safe_labels]
            )
            focal_weight_sums = focal_position_weights.sum(dim=1).clamp_min(
                self.epsilon
            )
            focal_per_sample = (
                focal_per_position * focal_position_weights
            ).sum(dim=1) / focal_weight_sums
            focal_loss = (
                focal_per_sample * valid_sample_weights
            ).sum() / valid_sample_count

            probs = log_probs.exp()
            one_hot_true = F.one_hot(
                safe_labels, num_classes=self.NUM_CLASSES
            ).to(dtype=torch.float32)
            positive_probs = probs[..., self.INTRON_START :]
            positive_true = one_hot_true[..., self.INTRON_START :]
            expanded_mask = loss_mask_fp32.unsqueeze(-1)
            intersection = (
                positive_probs * positive_true * expanded_mask
            ).sum(dim=1)
            predicted_positives = (
                positive_probs * expanded_mask
            ).sum(dim=1)
            possible_positives = (
                positive_true * expanded_mask
            ).sum(dim=1)
            present_f1_loss = 1.0 - (
                2.0 * intersection + self.epsilon
            ) / (predicted_positives + possible_positives + self.epsilon)

            soft_gate = F.softmax(
                logits_fp32 / self.gate_temperature, dim=-1
            )
            positive_soft_gate = soft_gate[..., self.INTRON_START :]
            masked_positive_gate = positive_soft_gate * expanded_mask
            false_positive_total = (
                positive_probs * masked_positive_gate
            ).sum(dim=1)
            false_positive_weight = masked_positive_gate.sum(dim=1)
            absent_state_loss = false_positive_total / (
                false_positive_weight + self.epsilon
            )
            classwise_f1_loss = torch.where(
                possible_positives > 0,
                present_f1_loss,
                absent_state_loss,
            )
            state_f1_loss = (
                classwise_f1_loss.sum(dim=1) * valid_sample_weights
            ).sum() / valid_sample_count

            intron_probs = probs[
                ..., self.INTRON_START : self.CDS_START
            ].sum(dim=-1)
            intron_true = one_hot_true[
                ..., self.INTRON_START : self.CDS_START
            ].sum(dim=-1)
            intron_f1_loss = self._group_f1_loss(
                intron_probs,
                intron_true,
                loss_mask_fp32,
                normalized_lengths,
                valid_sample_weights,
                valid_sample_count,
            )
            cds_probs = probs[..., self.CDS_START :].sum(dim=-1)
            cds_true = one_hot_true[..., self.CDS_START :].sum(dim=-1)
            cds_f1_loss = self._group_f1_loss(
                cds_probs,
                cds_true,
                loss_mask_fp32,
                normalized_lengths,
                valid_sample_weights,
                valid_sample_count,
            )
            return (
                focal_loss
                + self.state_f1_weight * state_f1_loss
                + self.intron_f1_weight * intron_f1_loss
                + self.cds_f1_weight * cds_f1_loss
            )


class TiberiusCceF1Loss(nn.Module):
    """Masked PyTorch version of Tiberius' CCE-F1 loss for 15-state labels.

    Combines token-averaged categorical cross-entropy with a soft F1 term over
    coding-related classes (default indices ``[coding_start, num_classes)``,
    i.e. CDS + boundary states when ``coding_start=4``), plus a false-positive
    penalty on classes that are absent in a sample.
    """

    NUM_CLASSES = NUM_CLASSES

    def __init__(
        self,
        f1_factor: float = 2.0,
        coding_start: int = 4,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if f1_factor < 0:
            raise ValueError(f"f1_factor must be non-negative, got {f1_factor}.")
        coding_start = int(coding_start)
        if not 0 < coding_start < self.NUM_CLASSES:
            raise ValueError(
                f"coding_start must be in [1, {self.NUM_CLASSES - 1}], got {coding_start}."
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        self.f1_factor = float(f1_factor)
        self.coding_start = coding_start
        self.epsilon = float(epsilon)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(
                f"logits must have [B, L, C] dims, got {tuple(logits.shape)}."
            )
        if labels.shape != logits.shape[:2]:
            raise ValueError(
                f"labels must have [B, L] dims matching logits, got {tuple(labels.shape)} "
                f"for logits {tuple(logits.shape)}."
            )
        if loss_mask.shape != labels.shape:
            raise ValueError(
                f"loss_mask must have [B, L] dims matching labels, got "
                f"{tuple(loss_mask.shape)} for labels {tuple(labels.shape)}."
            )
        if logits.shape[-1] != self.NUM_CLASSES:
            raise ValueError(
                f"Expected {self.NUM_CLASSES} classes, got {logits.shape[-1]}."
            )

        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits_fp32 = logits.float()
            labels = labels.to(device=logits.device, dtype=torch.long)
            loss_mask = loss_mask.to(device=logits.device, dtype=torch.float32)
            safe_labels = labels.masked_fill(loss_mask <= 0, 0)
            if torch.any((safe_labels < 0) | (safe_labels >= self.NUM_CLASSES)):
                raise ValueError(
                    f"Unmasked labels must be in [0, {self.NUM_CLASSES - 1}]."
                )

            batch_size, _, num_classes = logits_fp32.shape
            valid_lengths = loss_mask.sum(dim=1).clamp_min(self.epsilon)

            cce_loss = F.cross_entropy(
                logits_fp32.reshape(-1, num_classes),
                safe_labels.reshape(-1),
                reduction="none",
            ).view_as(labels)
            cce_loss = ((cce_loss * loss_mask).sum(dim=1) / valid_lengths).sum() / batch_size

            probs = F.softmax(logits_fp32, dim=-1)
            coding_probs = probs[..., self.coding_start :]
            coding_true = F.one_hot(safe_labels, num_classes=num_classes).to(
                dtype=logits_fp32.dtype
            )[..., self.coding_start :]
            coding_mask = loss_mask.unsqueeze(-1)

            true_positives = (coding_probs * coding_true * coding_mask).sum(dim=1)
            predicted_positives = (coding_probs * coding_mask).sum(dim=1)
            possible_positives = (coding_true * coding_mask).sum(dim=1)

            precision = true_positives / (predicted_positives + self.epsilon)
            recall = true_positives / (possible_positives + self.epsilon)
            f1_score = (
                2.0 * precision * recall / (precision + recall + self.epsilon)
            )

            has_positives = (possible_positives > 0).to(dtype=logits_fp32.dtype)
            f1_loss = ((1.0 - f1_score) * has_positives).sum() / batch_size

            absent_class_probs = (
                coding_probs * (1.0 - has_positives).unsqueeze(1) * coding_mask
            )
            false_positive_rate = (
                absent_class_probs.sum(dim=(1, 2)) / valid_lengths
            ).sum() / batch_size

            return cce_loss + self.f1_factor * (f1_loss + false_positive_rate)


def build_strand_loss(loss_cfg: Union[Mapping[str, Any], Any]) -> nn.Module:
    """Construct a per-strand loss module from a LossConfig-like object/mapping."""

    if isinstance(loss_cfg, Mapping):
        name = str(loss_cfg.get("name", LOSS_MASKED_FOCAL_F1))
        get = loss_cfg.get
    else:
        name = str(getattr(loss_cfg, "name", LOSS_MASKED_FOCAL_F1))
        get = lambda key, default=None: getattr(loss_cfg, key, default)

    if name == LOSS_MASKED_FOCAL_F1:
        return MaskedFocalF1Loss(
            gamma=float(get("gamma", 2.0)),
            state_f1_weight=float(get("state_f1_weight", 0.5)),
            intron_f1_weight=float(get("intron_f1_weight", 0.5)),
            cds_f1_weight=float(get("cds_f1_weight", 0.5)),
            gate_temperature=float(get("gate_temperature", 0.20)),
            epsilon=float(get("epsilon", 1e-5)),
        )
    if name == LOSS_TIBERIUS_CCE_F1:
        return TiberiusCceF1Loss(
            f1_factor=float(get("f1_factor", 2.0)),
            coding_start=int(get("coding_start", 4)),
            epsilon=float(get("epsilon", 1e-6)),
        )
    raise ValueError(
        f"Unsupported loss.name={name!r}; expected one of {list(SUPPORTED_LOSS_NAMES)}."
    )


class GeneAnnTrainingCriterion(nn.Module):
    """Crop full-window labels and average the chosen per-strand objective."""

    def __init__(
        self,
        *,
        num_features: int,
        flank_length: int,
        loss_cfg: Optional[Any] = None,
        # Backward-compatible kwargs for MaskedFocalF1 (used when loss_cfg is None).
        name: str = LOSS_MASKED_FOCAL_F1,
        gamma: float = 2.0,
        state_f1_weight: float = 0.5,
        intron_f1_weight: float = 0.5,
        cds_f1_weight: float = 0.5,
        gate_temperature: float = 0.20,
        epsilon: float = 1e-5,
        f1_factor: float = 2.0,
        coding_start: int = 4,
    ) -> None:
        super().__init__()
        if int(num_features) != NUM_CLASSES:
            raise ValueError(
                f"GeneAnn loss requires {NUM_CLASSES} states per strand, "
                f"got {num_features}."
            )
        self.num_features = int(num_features)
        self.flank_length = int(flank_length)
        if loss_cfg is None:
            loss_cfg = {
                "name": name,
                "gamma": gamma,
                "state_f1_weight": state_f1_weight,
                "intron_f1_weight": intron_f1_weight,
                "cds_f1_weight": cds_f1_weight,
                "gate_temperature": gate_temperature,
                "epsilon": epsilon,
                "f1_factor": f1_factor,
                "coding_start": coding_start,
            }
        self.loss_name = (
            str(loss_cfg["name"])
            if isinstance(loss_cfg, Mapping)
            else str(getattr(loss_cfg, "name", name))
        )
        self.loss_fn = build_strand_loss(loss_cfg)

    def prepare_targets(
        self,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        output_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return crop_strand_labels_and_masks(
            labels,
            loss_mask,
            flank_length=self.flank_length,
            output_length=output_length,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[-1] != 2 * self.num_features:
            raise ValueError(
                f"Expected logits [B, L, {2 * self.num_features}], got "
                f"{tuple(logits.shape)}."
            )
        labels, loss_mask = self.prepare_targets(
            labels,
            loss_mask,
            output_length=logits.shape[1],
        )
        labels = labels.to(device=logits.device)
        loss_mask = loss_mask.to(device=logits.device)
        positive_logits = logits[..., : self.num_features]
        negative_logits = logits[..., self.num_features :]
        return 0.5 * (
            self.loss_fn(
                positive_logits,
                labels[:, 0],
                loss_mask[:, 0],
            )
            + self.loss_fn(
                negative_logits,
                labels[:, 1],
                loss_mask[:, 1],
            )
        )
