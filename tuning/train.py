"""Train GeneAnn models with PEFT (Mamba-LoRA) or full-parameter fine-tuning.

Supports single-GPU and multi-GPU training/evaluation via HuggingFace
Accelerate (``accelerate>=0.32``, matched to the caduceus_env stack).

Launch examples::

    # PEFT
    python -m tuning.train --model_path ... --tuning_config tuning/tuning_config/peft_tuning_config.yml \\
        --train_dataset ... --output_dir ...

    # Full-parameter
    python -m tuning.train --model_path ... --tuning_config tuning/tuning_config/full_tuning_config.yml \\
        --train_dataset ... --output_dir ...

    # multi-GPU
    accelerate launch --num_processes 4 -m tuning.train --model_path ... \\
        --tuning_config tuning/tuning_config/full_tuning_config.yml --train_dataset ... --output_dir ...
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import default_data_collator, get_scheduler

from .config import (
    FullTuningConfig,
    HalfTuningConfig,
    PeftTuningConfig,
    load_tuning_config,
)
from .data import (
    REQUIRED_COLUMNS,
    LoadedDatasetCollection,
    load_dataset_collection,
    save_dataset_manifest,
    split_dataset_collection,
)
from .metrics import MaskedSegmentationMetrics
from .model import FullTuningModel, MambaLoRATuningModel


logger = get_logger("PlantGeneAnn.tuning")
TuningModel = Union[MambaLoRATuningModel, FullTuningModel]
TuningConfig = Union[PeftTuningConfig, HalfTuningConfig, FullTuningConfig]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GeneAnn-v2 PEFT (Mamba-LoRA) or full-parameter tuning "
        "(single- or multi-GPU via HuggingFace Accelerate)."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--tuning_config",
        required=True,
        help=(
            "Path to peft_tuning_config.yml, half_tuning_config.yml, "
            "or full_tuning_config.yml."
        ),
    )
    parser.add_argument("--train_dataset", required=True)
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument(
        "--valid_dataset",
        default=None,
        help="Independent validation Dataset/root; mutually exclusive with --valid_fraction.",
    )
    parser.add_argument("--output_dir", required=True)
    validation_group.add_argument(
        "--valid_fraction",
        type=float,
        default=0.0,
        help=(
            "Development-only random split; mutually exclusive with "
            "--valid_dataset."
        ),
    )
    parser.add_argument(
        "--save_last",
        action="store_true",
        help="Override checkpoint.save_last=true for this run.",
    )
    return parser.parse_args()


def load_datasets(
    args: argparse.Namespace,
    *,
    seed: int,
) -> Tuple[LoadedDatasetCollection, LoadedDatasetCollection]:
    """Load one Dataset or recursively discovered multi-Dataset collections."""

    train_collection = load_dataset_collection(args.train_dataset, role="train")
    if args.valid_dataset:
        validation_collection = load_dataset_collection(
            args.valid_dataset, role="validation"
        )
        if train_collection.sequence_length != validation_collection.sequence_length:
            raise ValueError(
                "Train/validation sequence lengths differ: "
                f"{train_collection.sequence_length} vs "
                f"{validation_collection.sequence_length}."
            )
        return train_collection, validation_collection

    if not 0.0 < args.valid_fraction < 1.0:
        raise ValueError(
            "Provide --valid_dataset (recommended) or an explicit "
            "--valid_fraction between 0 and 1."
        )
    return split_dataset_collection(
        train_collection,
        validation_fraction=args.valid_fraction,
        seed=seed,
    )


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Build a per-process DataLoader.

    Accelerate injects a distributed sampler in ``accelerator.prepare``; the
    generator only seeds any residual CPU-side randomness (e.g. worker init).
    """

    generator = torch.Generator()
    generator.manual_seed(seed)

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        import random

        import numpy as np

        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": default_data_collator,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
        kwargs["worker_init_fn"] = _worker_init_fn
    return DataLoader(**kwargs)


def filter_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only the columns required by the tuning forward pass."""
    return {key: value for key, value in batch.items() if key in REQUIRED_COLUMNS}


def pad_validation_dataset(
    dataset: Dataset,
    *,
    sequence_length: int,
    batch_size: int,
    num_processes: int,
) -> Dataset:
    """Pad validation rows to a whole number of distributed batches.

    Accelerate's prepared validation loader must yield the same number of
    batches on every rank so DDP never enters its uneven-input join hook. The
    synthetic rows carry an all-zero loss mask, which the training criterion
    and metrics intentionally ignore. Real validation rows are never copied.
    """
    global_batch_size = int(batch_size) * max(1, int(num_processes))
    if global_batch_size <= 0:
        raise ValueError("Validation batch_size and num_processes must be positive.")
    padding_rows = (-len(dataset)) % global_batch_size
    if padding_rows == 0:
        return dataset

    sequence_length = int(sequence_length)
    if sequence_length <= 0:
        raise ValueError("Validation sequence_length must be positive.")
    # Reuse one real tokenized sequence so the model sees valid vocabulary IDs
    # even though the all-zero mask makes this row invisible to supervision.
    dummy_input_ids = list(dataset[0]["input_ids"])
    if len(dummy_input_ids) != sequence_length + 2:
        raise ValueError(
            "Validation Dataset first input_ids row does not match sequence_length."
        )
    dummy_labels = [[0] * sequence_length, [0] * sequence_length]
    dummy_mask = [[0] * sequence_length, [0] * sequence_length]
    padding = Dataset.from_dict(
        {
            "input_ids": [dummy_input_ids] * padding_rows,
            "labels": [dummy_labels] * padding_rows,
            "attention_mask": [dummy_mask] * padding_rows,
        },
        features=dataset.features,
    )
    return concatenate_datasets([dataset, padding])


def resolve_mixed_precision(precision: str) -> str:
    if precision not in {"bf16", "fp16", "no"}:
        raise ValueError(f"Unsupported mixed_precision={precision!r}.")
    return precision


def build_ddp_kwargs(method: str) -> DistributedDataParallelKwargs:
    """Build the fixed method-specific DDP policy.

    Full tuning trains dynamically routed MoE experts, so a rank may leave an
    expert unused in a given synchronized update. DDP must discover those
    parameters dynamically. PEFT and half tuning freeze every backbone expert
    and retain the faster default reducer path.
    """

    if method not in {"peft", "half", "full"}:
        raise ValueError(f"Unsupported tuning method for DDP policy: {method!r}.")
    is_full = method == "full"
    return DistributedDataParallelKwargs(
        find_unused_parameters=is_full,
        static_graph=False,
        gradient_as_bucket_view=is_full,
    )


def build_optimizer(
    optimizer_groups: List[dict],
    *,
    training,
    method: str,
) -> torch.optim.AdamW:
    """Construct AdamW with a memory-stable full-tuning implementation."""

    kwargs = {"betas": (training.beta1, training.beta2)}
    if method == "full":
        # Avoid foreach tensor-list temporaries during the first 195M-parameter
        # AdamW update. This preserves AdamW math while lowering peak memory.
        kwargs["foreach"] = False
    return torch.optim.AdamW(optimizer_groups, **kwargs)


def empty_cuda_cache_after_validation(
    *,
    device: torch.device,
    reason: str,
) -> None:
    """Release local CUDA cache strictly after a validation pass completes."""

    if device.type != "cuda":
        return
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(
        "Cleared CUDA caching allocator after validation (%s) on device=%s",
        reason,
        device,
        main_process_only=True,
    )


def _tqdm_disable(accelerator: Accelerator) -> bool:
    """Show progress bars only on the main process to avoid multi-rank clutter."""
    return not accelerator.is_local_main_process


def build_model(
    args: argparse.Namespace,
    tuning_config: TuningConfig,
) -> TuningModel:
    if isinstance(tuning_config, (PeftTuningConfig, HalfTuningConfig)):
        return MambaLoRATuningModel.from_pretrained(
            args.model_path,
            peft_config=tuning_config,
            torch_dtype=torch.float32,
        )
    if isinstance(tuning_config, FullTuningConfig):
        return FullTuningModel.from_pretrained(
            args.model_path,
            full_config=tuning_config,
            torch_dtype=torch.float32,
        )
    raise TypeError(f"Unsupported tuning config type: {type(tuning_config)!r}")


def save_resolved_config(
    tuning_config: TuningConfig,
    model: TuningModel,
    output_dir: str,
    summary,
) -> None:
    n_layer = int(model.config.n_layer)
    if isinstance(tuning_config, (PeftTuningConfig, HalfTuningConfig)):
        tuning_config.save_resolved_yaml(
            output_dir,
            n_layer=n_layer,
            trainable=summary.trainable,
            total=summary.total,
        )
        return
    group_counts = (
        model.parameter_group_counts()
        if isinstance(model, FullTuningModel)
        else None
    )
    tuning_config.save_resolved_yaml(
        output_dir,
        n_layer=n_layer,
        trainable=summary.trainable,
        total=summary.total,
        group_counts=group_counts,
    )


def log_model_summary(model: TuningModel, summary) -> None:
    if isinstance(model, MambaLoRATuningModel):
        optimizer_config = model.tuning_config.optimization
        logger.info(
            "Model parameters: method=%s total=%s trainable=%s (%.4f%%), "
            "LoRA=%s, decoder=%s, lr=%.3e, weight_decay=%.3f",
            model.tuning_config.method,
            f"{summary.total:,}",
            f"{summary.trainable:,}",
            100.0 * summary.trainable_fraction,
            f"{summary.lora:,}",
            f"{summary.decoder:,}",
            optimizer_config.learning_rate,
            optimizer_config.weight_decay,
            main_process_only=True,
        )
        return
    logger.info(
        "Model parameters: method=full total=%s trainable=%s (%.4f%%), "
        "lr=%s, weight_decay=%s, scope=all_parameters",
        f"{summary.total:,}",
        f"{summary.trainable:,}",
        100.0 * summary.trainable_fraction,
        f"{model.full_config.optimization.learning_rate:.3e}",
        f"{model.full_config.optimization.weight_decay}",
        main_process_only=True,
    )


@torch.no_grad()
def evaluate(
    model: TuningModel,
    dataloader: DataLoader,
    accelerator: Accelerator,
    *,
    desc: str = "eval",
) -> Dict[str, float]:
    """Run distributed evaluation and return globally reduced metrics."""

    model.eval()
    metrics = MaskedSegmentationMetrics()
    loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    sample_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)

    unwrapped = accelerator.unwrap_model(model)

    progress = tqdm(
        dataloader,
        desc=desc,
        total=len(dataloader),
        disable=_tqdm_disable(accelerator),
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        batch = filter_batch(batch)
        # The validation padding rows have an all-zero mask. Count only rows
        # with at least one supervised position so they do not dilute loss.
        valid_samples = batch["attention_mask"].to(dtype=torch.bool).any(dim=(1, 2))
        valid_count = int(valid_samples.sum().item())
        outputs = model(**batch)
        if outputs.loss is not None and valid_count > 0:
            loss_sum += (
                outputs.loss.detach().float().to(torch.float64) * valid_count
            )
            sample_count += valid_count
        cropped_labels, cropped_mask = unwrapped.prepare_targets(
            batch["labels"],
            batch["attention_mask"],
            output_length=outputs.logits.shape[1],
        )
        metrics.update(outputs.logits, cropped_labels, cropped_mask)
        if not _tqdm_disable(accelerator) and sample_count.item() > 0:
            progress.set_postfix(
                loss=f"{(loss_sum / sample_count).item():.4f}",
                refresh=False,
            )

    reduced_stats = accelerator.reduce(
        torch.stack([loss_sum, sample_count]),
        reduction="sum",
    )
    reduced_confusion = accelerator.reduce(
        metrics.distributed_state(accelerator.device),
        reduction="sum",
    )
    metrics.load_distributed_state(reduced_confusion)

    total_loss = float(reduced_stats[0].item())
    total_count = float(reduced_stats[1].item())
    result = metrics.compute()
    result["loss"] = total_loss / max(1.0, total_count)
    return result


def evaluate_and_cleanup(
    model: TuningModel,
    dataloader: DataLoader,
    accelerator: Accelerator,
    *,
    desc: str,
    reason: str,
) -> Dict[str, float]:
    """Evaluate, then clear CUDA cache exactly once after all ranks finish."""

    result = evaluate(model, dataloader, accelerator, desc=desc)
    accelerator.wait_for_everyone()
    empty_cuda_cache_after_validation(
        device=accelerator.device,
        reason=reason,
    )
    accelerator.wait_for_everyone()
    return result


def save_checkpoint(
    model: TuningModel,
    accelerator: Accelerator,
    output_dir: str,
    name: str,
    metrics: Dict[str, float],
    global_step: int,
) -> None:
    """Save method-specific checkpoint bundles from the main process only."""

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        checkpoint_dir = os.path.join(output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        safe = bool(
            getattr(unwrapped.tuning_config.checkpoint, "safe_serialization", False)
        )
        unwrapped.save_checkpoint_bundle(checkpoint_dir, safe_serialization=safe)
        with open(
            os.path.join(checkpoint_dir, "metrics.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {"global_step": global_step, **metrics},
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    accelerator.wait_for_everyone()


def _trainable_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def main() -> None:
    args = parse_args()
    tuning_config = load_tuning_config(args.tuning_config)
    training = tuning_config.training
    mixed_precision = resolve_mixed_precision(training.mixed_precision)
    method = tuning_config.method

    # Full tuning needs dynamic unused-parameter discovery for routed MoE
    # experts. PEFT/half retain the cheaper default DDP reducer behavior.
    ddp_kwargs = build_ddp_kwargs(method)

    dataloader_config = DataLoaderConfiguration(even_batches=True)

    # step_scheduler_with_optimizer=False: one LR step per optimizer update.
    accelerator = Accelerator(
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        dataloader_config=dataloader_config,
        step_scheduler_with_optimizer=False,
        log_with=None,
        kwargs_handlers=[ddp_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
        force=True,
    )
    logger.info(
        "Accelerate initialized: method=%s, processes=%d, mixed_precision=%s, "
        "gradient_accumulation_steps=%d, device=%s",
        method,
        accelerator.num_processes,
        accelerator.mixed_precision,
        accelerator.gradient_accumulation_steps,
        accelerator.device,
        main_process_only=True,
    )

    if (
        training.mixed_precision == "bf16"
        and accelerator.device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "mixed_precision=bf16 requested, but this GPU lacks BF16 support."
        )

    set_seed(training.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    train_collection, validation_collection = load_datasets(args, seed=training.seed)
    if accelerator.is_main_process:
        save_dataset_manifest(
            args.output_dir,
            train=train_collection,
            validation=validation_collection,
        )
    train_dataset = train_collection.dataset
    validation_dataset = pad_validation_dataset(
        validation_collection.dataset,
        sequence_length=validation_collection.sequence_length,
        batch_size=training.eval_batch_size,
        num_processes=accelerator.num_processes,
    )
    if len(validation_dataset) != len(validation_collection.dataset):
        logger.info(
            "Padded validation loader with %d zero-mask row(s) for %d process(es).",
            len(validation_dataset) - len(validation_collection.dataset),
            accelerator.num_processes,
            main_process_only=True,
        )
    accelerator.wait_for_everyone()

    model = build_model(args, tuning_config)
    summary = model.parameter_summary()
    if accelerator.is_main_process:
        save_resolved_config(tuning_config, model, args.output_dir, summary)
    log_model_summary(model, summary)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=training.train_batch_size,
        shuffle=True,
        num_workers=training.num_workers,
        seed=training.seed,
    )
    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=training.eval_batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        seed=training.seed,
    )

    optimizer_groups = model.get_optimizer_grouped_parameters()
    optimizer = build_optimizer(
        optimizer_groups,
        training=training,
        method=method,
    )

    model, optimizer, train_loader, validation_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        validation_loader,
    )

    updates_per_epoch = max(
        1,
        math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps),
    )
    total_updates = updates_per_epoch * training.epochs
    warmup_steps = int(total_updates * training.warmup_ratio)
    scheduler = get_scheduler(
        training.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    logger.info(
        "Training schedule: method=%s, scheduler=%s, epochs=%d, "
        "per-process train batches/epoch=%d, updates/epoch=%d, total_updates=%d, "
        "lr=%.3e, weight_decay=%.3f, lr_warmup_steps=%d, world_size=%d",
        method,
        training.scheduler,
        training.epochs,
        len(train_loader),
        updates_per_epoch,
        total_updates,
        tuning_config.optimization.learning_rate,
        tuning_config.optimization.weight_decay,
        warmup_steps,
        accelerator.num_processes,
        main_process_only=True,
    )

    global_step = 0
    best_metric = -math.inf
    evaluations_without_improvement = 0
    stop_training = False
    running_loss = 0.0
    running_loss_count = 0
    last_train_loss: Optional[float] = None

    for epoch in range(training.epochs):
        model.train()
        epoch_progress = tqdm(
            train_loader,
            desc=f"train epoch {epoch + 1}/{training.epochs}",
            total=len(train_loader),
            disable=_tqdm_disable(accelerator),
            leave=True,
            dynamic_ncols=True,
        )
        for batch in epoch_progress:
            batch = filter_batch(batch)
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        _trainable_parameters(model),
                        training.max_grad_norm,
                    )

                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()

            gathered_loss = accelerator.gather(loss.detach().float()).mean().item()
            running_loss += gathered_loss
            running_loss_count += 1
            last_train_loss = gathered_loss

            postfix = {
                "loss": f"{last_train_loss:.4f}",
                "step": global_step,
            }
            if accelerator.is_local_main_process:
                postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
            epoch_progress.set_postfix(postfix, refresh=False)

            if not accelerator.sync_gradients:
                continue

            global_step += 1

            if global_step % training.logging_steps == 0:
                mean_loss = running_loss / max(1, running_loss_count)
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "epoch=%d step=%d train_loss=%.6f lr=%.3e method=%s",
                    epoch + 1,
                    global_step,
                    mean_loss,
                    current_lr,
                    method,
                    main_process_only=True,
                )
                running_loss = 0.0
                running_loss_count = 0

            if global_step % training.eval_steps == 0:
                epoch_progress.set_postfix_str("evaluating...", refresh=True)
                evaluation = evaluate_and_cleanup(
                    model,
                    validation_loader,
                    accelerator,
                    desc=f"eval step {global_step}",
                    reason=f"step={global_step}",
                )
                selected_metric = evaluation[training.metric_for_best_model]
                logger.info(
                    "evaluation step=%d %s=%.6f loss=%.6f boundary_f1=%.6f",
                    global_step,
                    training.metric_for_best_model,
                    selected_metric,
                    evaluation["loss"],
                    evaluation["boundary_f1"],
                    main_process_only=True,
                )

                if selected_metric > best_metric:
                    best_metric = selected_metric
                    evaluations_without_improvement = 0
                    save_checkpoint(
                        model,
                        accelerator,
                        args.output_dir,
                        "best",
                        evaluation,
                        global_step,
                    )
                else:
                    evaluations_without_improvement += 1

                if evaluations_without_improvement >= training.early_stopping_patience:
                    logger.info(
                        "Early stopping at step %d.",
                        global_step,
                        main_process_only=True,
                    )
                    stop_training = True

                model.train()

            if stop_training:
                break

        epoch_progress.close()

        if stop_training:
            break

        if global_step > 0 and global_step % training.eval_steps != 0:
            evaluation = evaluate_and_cleanup(
                model,
                validation_loader,
                accelerator,
                desc=f"eval epoch {epoch + 1}",
                reason=f"epoch={epoch + 1}",
            )
            selected_metric = evaluation[training.metric_for_best_model]
            logger.info(
                "epoch-end evaluation epoch=%d step=%d %s=%.6f loss=%.6f",
                epoch + 1,
                global_step,
                training.metric_for_best_model,
                selected_metric,
                evaluation["loss"],
                main_process_only=True,
            )
            if selected_metric > best_metric:
                best_metric = selected_metric
                evaluations_without_improvement = 0
                save_checkpoint(
                    model,
                    accelerator,
                    args.output_dir,
                    "best",
                    evaluation,
                    global_step,
                )
            else:
                evaluations_without_improvement += 1

            if evaluations_without_improvement >= training.early_stopping_patience:
                logger.info(
                    "Early stopping after epoch %d.",
                    epoch + 1,
                    main_process_only=True,
                )
                stop_training = True
                break

    if args.save_last or tuning_config.checkpoint.save_last:
        final_metrics = evaluate_and_cleanup(
            model,
            validation_loader,
            accelerator,
            desc="eval final",
            reason="final",
        )
        save_checkpoint(
            model,
            accelerator,
            args.output_dir,
            "last",
            final_metrics,
            global_step,
        )

    accelerator.wait_for_everyone()
    logger.info(
        "Training complete: method=%s best_%s=%.6f, best_model=%s, world_size=%d",
        method,
        training.metric_for_best_model,
        best_metric,
        os.path.join(args.output_dir, "best", "model"),
        accelerator.num_processes,
        main_process_only=True,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
