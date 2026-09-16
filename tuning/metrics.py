"""Memory-bounded position-level metrics for GeneAnn PEFT tuning."""

from __future__ import annotations

from typing import Dict

import torch


NUM_CLASSES = 15


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1),
        torch.zeros_like(numerator, dtype=torch.float64),
    )


class MaskedSegmentationMetrics:
    """Accumulate per-strand confusion matrices without retaining logits.

    Designed for single-process use and multi-GPU evaluation via all-reduce of
    the compact confusion tensor (see :meth:`distributed_state` /
    :meth:`load_distributed_state`).
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = int(num_classes)
        self.confusion = torch.zeros(
            (2, self.num_classes, self.num_classes), dtype=torch.int64
        )

    def reset(self) -> None:
        """Zero the accumulated confusion matrices."""
        self.confusion.zero_()

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> None:
        if logits.ndim != 3 or logits.shape[-1] != 2 * self.num_classes:
            raise ValueError(
                f"Expected logits [B, L, {2 * self.num_classes}], got {tuple(logits.shape)}."
            )
        if labels.ndim != 3 or labels.shape[1] != 2:
            raise ValueError(f"Expected labels [B, 2, L], got {tuple(labels.shape)}.")
        if tuple(loss_mask.shape) != tuple(labels.shape):
            raise ValueError("loss_mask must have the same shape as labels.")

        predictions = torch.stack(
            (
                logits[..., : self.num_classes].argmax(dim=-1),
                logits[..., self.num_classes :].argmax(dim=-1),
            ),
            dim=1,
        )
        length = predictions.shape[-1]
        if labels.shape[-1] != length:
            raise ValueError(
                "Metrics require labels already cropped by the training criterion: "
                f"label length={labels.shape[-1]}, logits length={length}."
            )

        for strand in range(2):
            valid = loss_mask[:, strand].to(dtype=torch.bool)
            true = labels[:, strand][valid].to(dtype=torch.int64)
            pred = predictions[:, strand][valid].to(dtype=torch.int64)
            if true.numel() == 0:
                continue
            indices = true * self.num_classes + pred
            counts = torch.bincount(
                indices,
                minlength=self.num_classes * self.num_classes,
            ).reshape(self.num_classes, self.num_classes)
            self.confusion[strand] += counts.detach().cpu()

    def distributed_state(self, device: torch.device) -> torch.Tensor:
        """Return the confusion tensor on *device* for ``accelerator.reduce``."""
        return self.confusion.to(device=device, dtype=torch.float64)

    def load_distributed_state(self, confusion: torch.Tensor) -> None:
        """Load an all-reduced confusion tensor produced across processes."""
        expected = (2, self.num_classes, self.num_classes)
        if tuple(confusion.shape) != expected:
            raise ValueError(
                f"Distributed confusion shape {tuple(confusion.shape)} != {expected}."
            )
        self.confusion = confusion.detach().to(device="cpu", dtype=torch.int64)

    def compute(self) -> Dict[str, float]:
        confusion = self.confusion.to(dtype=torch.float64)
        true_positive = confusion.diagonal(dim1=-2, dim2=-1)
        predicted = confusion.sum(dim=-2)
        actual = confusion.sum(dim=-1)
        precision = _safe_divide(true_positive, predicted)
        recall = _safe_divide(true_positive, actual)
        # Compute F1 directly from confusion counts. Reusing _safe_divide on
        # precision + recall would clamp valid fractional denominators below 1.
        f1 = _safe_divide(2.0 * true_positive, predicted + actual)
        # Include classes with false-positive predictions so macro F1 penalizes
        # predicted-only states. Classes absent from both truth and prediction
        # have no evidence and are excluded from the average.
        evaluated = (actual > 0) | (predicted > 0)

        result: Dict[str, float] = {}
        strand_names = ("positive", "negative")
        for strand, strand_name in enumerate(strand_names):
            result[f"{strand_name}_macro_f1"] = float(
                f1[strand][evaluated[strand]].mean().item()
                if evaluated[strand].any()
                else 0.0
            )
            for class_index in range(self.num_classes):
                result[f"{strand_name}_class_{class_index}_f1"] = float(
                    f1[strand, class_index].item()
                )

        all_tp = true_positive.sum(dim=0)
        all_predicted = predicted.sum(dim=0)
        all_actual = actual.sum(dim=0)
        all_precision = _safe_divide(all_tp, all_predicted)
        all_recall = _safe_divide(all_tp, all_actual)
        all_f1 = _safe_divide(2.0 * all_tp, all_predicted + all_actual)
        # Apply the same evidence rule after pooling the two strands.
        all_evaluated = (all_actual > 0) | (all_predicted > 0)
        result["macro_f1"] = float(
            all_f1[all_evaluated].mean().item()
            if all_evaluated.any()
            else 0.0
        )
        result["nucleotide_accuracy"] = float(
            all_tp.sum().item() / all_actual.sum().item()
            if all_actual.sum() > 0
            else 0.0
        )

        combined_confusion = confusion.sum(dim=0)
        for name, class_slice in (
            ("intron", slice(1, 4)),
            ("cds", slice(4, 7)),
            ("boundary", slice(7, 15)),
        ):
            # Treat each state family as a binary set. A frame/state error within
            # the same family is still a true positive for the grouped metric;
            # the class-wise F1 values above retain the stricter distinction.
            group_tp = combined_confusion[class_slice, class_slice].sum()
            group_predicted = combined_confusion[:, class_slice].sum()
            group_actual = combined_confusion[class_slice, :].sum()
            group_precision = float(group_tp / group_predicted) if group_predicted > 0 else 0.0
            group_recall = float(group_tp / group_actual) if group_actual > 0 else 0.0
            result[f"{name}_precision"] = group_precision
            result[f"{name}_recall"] = group_recall
            result[f"{name}_f1"] = (
                2.0 * group_precision * group_recall / (group_precision + group_recall)
                if group_precision + group_recall > 0
                else 0.0
            )

        return result
