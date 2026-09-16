# PlantGeneAnn Fine-tuning Guide

This document describes species-specific adaptation of PlantGeneAnn using PEFT, half tuning, or full-parameter fine-tuning. For installation and inference, see the [main README](../README.md) and [inference guide](inference.md). For Dataset generation, see the [preprocessing guide](preprocessing.md).

## Contents

- [Overview](#overview)
- [Dataset requirements](#dataset-requirements)
- [Choosing a tuning strategy](#choosing-a-tuning-strategy)
- [Configuration files](#configuration-files)
- [Loss functions](#loss-functions)
- [Training configuration](#training-configuration)
- [Single-GPU training](#single-gpu-training)
- [Multi-GPU training](#multi-gpu-training)
- [Validation policy](#validation-policy)
- [Metrics and model selection](#metrics-and-model-selection)
- [Outputs and checkpoints](#outputs-and-checkpoints)
- [Inference with the tuned model](#inference-with-the-tuned-model)
- [Export](#export)
- [Starting another adaptation stage](#starting-another-adaptation-stage)
- [Resource guidance](#resource-guidance)

## Overview

Each released PlantGeneAnn checkpoint is a loss-free, inference-ready Hugging Face model. Choose the checkpoint matching the target plant clade (see the [pretrained model table](../README.md#-download-the-pretrained-model)); the `tuning` package wraps that base model with a training objective and exposes three adaptation strategies while preserving compatibility with the standard inference pipeline.

Fine-tuning can be useful when:

- the target lineage is evolutionarily distant from the species represented in the [training species table](species.md);
- GC content, repeat composition, intron lengths, or gene structure differs from the distribution represented by the [training species table](species.md);
- default predictions underperform on a trustworthy validation annotation;
- a limited amount of high-quality target-domain structural annotation is available.

All strategies use the same Dataset loader, training loop, losses, metrics, Accelerate integration, early stopping, and deployable model export.

## Dataset Requirements

Generate compatible data using:

```bash
python -m preprocess.cli \
  --input_dir /data/species_inputs \
  --output_dir /data/training_datasets \
  --model_path models/PlantGeneAnn-v2-Angiospermae
```

Each example must provide:

| Column | Shape | Description |
|---|---|---|
| `input_ids` | `[L+2]` | Token IDs with CLS and SEP; default `L=40960`, total 40,962 |
| `labels` | `[2, L]` | Positive- and negative-strand 15-state labels |
| `attention_mask` | `[2, L]` | Per-position loss weights |

`preprocess` generates the per-position loss weights (`0.0`, `0.5`, and
`1.0`). Provenance columns are permitted and automatically removed before the
model forward pass.

Before allocating the model, the loader performs only inexpensive startup checks:

- the source is a non-empty `datasets.Dataset`;
- the three required columns are present;
- the first row has the expected `input_ids`/`labels` geometry, CLS/SEP
  boundaries, and a sequence length divisible by the decoder factor (16);
- the sequence lengths reported from the training and validation first rows are
  compatible.

The loader does not scan every row and does not inspect `attention_mask` values,
shapes, or supervision coverage. `preprocess` is responsible for biological QC
and for removing windows whose positive- and negative-strand loss masks are both
zero. Masked positions may use sentinel labels outside the supervised class
range because they do not enter the objective.

### Dataset path discovery

`--train_dataset` and `--valid_dataset` accept either:

1. one complete `Dataset.save_to_disk()` directory; or
2. a parent directory containing multiple complete Dataset directories at any nested depth.

Sources are discovered recursively in stable path order, validated, column-pruned, and concatenated. Sampling is proportional to the number of windows, not balanced by species.

## Choosing a Tuning Strategy

| Strategy | Base-model adaptation | Decoder adaptation | Relative memory/compute | Suggested use |
|---|---|---|---|---|
| PEFT | Fixed Mamba-LoRA scope | Transformer LoRA plus selected trainable decoder/head blocks | Lowest | Small labeled datasets and rapid experiments |
| Half | Fixed Mamba-LoRA scope | Complete decoder and prediction head | Medium | Stronger structural adaptation without full backbone training |
| Full | Every parameter trainable | Complete model trainable | Highest | Larger high-quality datasets and sufficient GPU memory |

### PEFT

PEFT uses a fixed production strategy enforced by code:

1. apply Mamba-compatible LoRA to `in_proj` and `out_proj` in all 16 bidirectional backbone layers;
2. keep backbone dense SwiGLU, MoE experts, and MoE routers frozen;
3. apply LoRA to attention and feed-forward projections in every decoder Transformer block;
4. train the last up-convolution block, local-refinement block, final norm, and prediction head directly;
5. keep all remaining parameters frozen.

All trainable PEFT modules use the method-level optimizer settings:

```yaml
peft:
  learning_rate: 1.0e-4
  weight_decay: 0.01
```

The learning rate and weight decay are shared by the Mamba-LoRA parameters,
decoder Transformer-LoRA parameters, selected decoder blocks, and prediction
head. Biases and normalization parameters remain in the standard AdamW
no-decay groups. The structural scope remains fixed and is recorded in the
resolved configuration for provenance.

### Half tuning

Half tuning uses:

- Mamba-LoRA on all 16 backbone `in_proj/out_proj` modules;
- frozen backbone dense FFNs, MoE experts, and routers;
- no decoder Transformer LoRA;
- direct full-parameter training of the complete embeddings decoder and prediction head.

LoRA rank/alpha/dropout remain configurable, while all trainable half-tuning
modules use one method-level optimizer policy:

```yaml
half:
  learning_rate: 1.0e-4
  weight_decay: 0.01
```

The trainable decoder scope itself is fixed so users cannot accidentally train
only an inconsistent subset.

### Full-parameter tuning

Full tuning sets `requires_grad=True` for every model parameter. It uses one global learning rate and weight decay, separated only into decay and no-decay AdamW parameter groups. No LoRA or module-freezing policy is active.

Full tuning has the highest optimizer-state and gradient-memory requirements. The implementation uses a non-foreach AdamW path to reduce temporary optimizer memory for the approximately 195-million-parameter model.

## Configuration Files

The three released YAML files are:

```text
tuning/tuning_config/peft_tuning_config.yml
tuning/tuning_config/half_tuning_config.yml
tuning/tuning_config/full_tuning_config.yml
```

Select one with `--tuning_config`. Configuration parsing is strict: unknown keys, invalid values, and model-geometry mismatches fail immediately.

### PEFT defaults

```yaml
format_version: 1
method: peft

peft:
  learning_rate: 1.0e-4
  weight_decay: 0.01

loss:
  name: masked_focal_f1

training:
  optimizer: adamw
  beta1: 0.9
  beta2: 0.99
  scheduler: cosine  # cosine or linear
  warmup_ratio: 0.05
  max_grad_norm: 1.0
  mixed_precision: bf16
  train_batch_size: 1
  eval_batch_size: 8
  gradient_accumulation_steps: 16
  epochs: 20
  eval_steps: 1000
  logging_steps: 1000
  early_stopping_patience: 2
  metric_for_best_model: macro_f1
  seed: 42
  num_workers: 4

checkpoint:
  save_peft_only: true
  save_full_inference_model: true
  safe_serialization: false
  save_last: false
```

### Half defaults

The released half configuration uses Mamba-LoRA rank 8, alpha 8, no dropout,
and bidirectional sharing. Every trainable module uses learning rate `1e-4` and
weight decay `0.01` for eligible multidimensional weights; bias and
normalization parameters use the standard no-decay rule. These values are
configured in the `half` block and can be changed together. The complete
training and checkpoint defaults are:

```yaml
format_version: 1
method: half

half:
  learning_rate: 1.0e-4
  weight_decay: 0.01

loss:
  name: masked_focal_f1

training:
  optimizer: adamw
  beta1: 0.9
  beta2: 0.99
  scheduler: cosine  # cosine or linear
  warmup_ratio: 0.05
  max_grad_norm: 1.0
  mixed_precision: bf16
  train_batch_size: 1
  eval_batch_size: 8
  gradient_accumulation_steps: 16
  epochs: 20
  eval_steps: 1000
  logging_steps: 1000
  early_stopping_patience: 2
  metric_for_best_model: macro_f1
  seed: 42
  num_workers: 4

checkpoint:
  save_peft_only: true
  save_full_inference_model: true
  safe_serialization: false
  save_last: false
```

### Full defaults

```yaml
format_version: 1
method: full

full:
  learning_rate: 1.0e-4
  weight_decay: 0.01

loss:
  name: masked_focal_f1

training:
  optimizer: adamw
  beta1: 0.9
  beta2: 0.99
  scheduler: cosine  # cosine or linear
  warmup_ratio: 0.025
  max_grad_norm: 1.0
  mixed_precision: bf16
  train_batch_size: 1
  eval_batch_size: 8
  gradient_accumulation_steps: 16
  epochs: 20
  eval_steps: 1000
  logging_steps: 1000
  early_stopping_patience: 2
  metric_for_best_model: macro_f1
  seed: 42
  num_workers: 4

checkpoint:
  save_peft_only: false
  save_full_inference_model: true
  safe_serialization: false
  save_last: false
  save_trainable_bundle: true
```

## Loss Functions

Select the objective through `loss.name`.

| Name | Description |
|---|---|
| `masked_focal_f1` | PlantGeneAnn hierarchical masked focal cross-entropy plus class/group soft-F1 terms |
| `tiberius_cce_f1` | Tiberius-style masked categorical cross-entropy plus coding-related soft F1 and absent-class false-positive penalty |

Both objectives:

- operate separately on positive and negative strands;
- align labels/masks to the model output by identical flank cropping;
- respect per-position weights;
- average the two strand objectives equally;
- exist only in the training wrapper and are not shipped as dependencies of the exported inference model.

Loss hyperparameters use validated built-in defaults. The YAML selects the objective name rather than exposing every internal coefficient.

## Training Configuration

### Optimizer and scheduler

- optimizer: AdamW only;
- PEFT, half, and full each expose a method-level learning rate and weight decay in their YAML file;
- scheduler: cosine decay by default, or linear decay;
- warmup: linear warmup for `warmup_ratio × total_updates` optimizer steps (`0.05` for PEFT/half and `0.025` for full);
- gradient clipping: `max_grad_norm`;
- one scheduler update per synchronized optimizer update.

The supported scheduler values are `cosine` and `linear`. The default tuning
configurations use `cosine`; select linear decay by setting:

```yaml
training:
  scheduler: linear
```

### Effective batch size

An approximate global effective batch size is:

```text
train_batch_size × number_of_GPUs × gradient_accumulation_steps
```

`train_batch_size` and `eval_batch_size` are per process/GPU.

### Mixed precision

Supported values:

- `bf16`
- `fp16`
- `no`

Mixed precision is read from YAML, not a separate training CLI flag.

### DataLoader workers

`training.num_workers` is applied to each process. On multi-GPU nodes, total host worker count is approximately `num_workers × processes`; adjust it to the node's CPU and memory capacity.

### Reproducible initialization

The training seed initializes Python/NumPy/PyTorch data-side randomness through Accelerate and DataLoader generators. CUDA kernels may still have hardware- and implementation-dependent nondeterminism.

## Single-GPU Training

Run from the repository root.

### PEFT

```bash
python -m tuning.train \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --tuning_config tuning/tuning_config/peft_tuning_config.yml \
  --train_dataset /data/training_datasets \
  --valid_dataset /data/heldout_training_datasets \
  --output_dir /data/plantgeneann_peft
```

### Half tuning

```bash
python -m tuning.train \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --tuning_config tuning/tuning_config/half_tuning_config.yml \
  --train_dataset /data/training_datasets \
  --valid_dataset /data/heldout_training_datasets \
  --output_dir /data/plantgeneann_half
```

### Full tuning

```bash
python -m tuning.train \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --tuning_config tuning/tuning_config/full_tuning_config.yml \
  --train_dataset /data/training_datasets \
  --valid_dataset /data/heldout_training_datasets \
  --output_dir /data/plantgeneann_full
```

### CLI reference

| Option | Required/default | Description |
|---|---|---|
| `--model_path` | required | Base inference-ready Hugging Face model |
| `--tuning_config` | required | PEFT, half, or full YAML |
| `--train_dataset` | required | One Dataset or a recursively searched parent directory |
| `--valid_dataset` | recommended | Independent validation Dataset/root; mutually exclusive with `--valid_fraction` |
| `--valid_fraction` | default `0.0` | Development-only per-source random split when no validation path is supplied; mutually exclusive with `--valid_dataset` |
| `--output_dir` | required | Training configuration, manifests, and checkpoints |
| `--save_last` | off | Also save a final `last/` checkpoint |

Exactly one valid validation route is required: provide `--valid_dataset`, or set `--valid_fraction` strictly between 0 and 1. The two arguments are mutually exclusive; supplying both causes the CLI argument parser to exit with an error.

## Multi-GPU Training

PlantGeneAnn training uses Hugging Face Accelerate.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes 4 \
  -m tuning.train \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --tuning_config tuning/tuning_config/peft_tuning_config.yml \
  --train_dataset /data/training_datasets \
  --valid_dataset /data/heldout_training_datasets \
  --output_dir /data/plantgeneann_peft
```

Distributed behavior:

- gradients synchronize under `accelerator.accumulate`;
- PEFT and half tuning use the faster default DDP reducer because MoE experts are frozen;
- full tuning enables unused-parameter discovery because dynamic routing can leave some experts unused in an update;
- validation rows are padded to a common per-rank batch count with synthetic
  all-zero-mask sentinel rows;
- sentinel rows are excluded from the validation loss and all reported metrics;
- confusion matrices and loss totals are reduced across all ranks;
- checkpoints are written by the main process only;
- progress bars are shown on the local main process only;
- each rank clears its CUDA cache once after a completed validation pass.

## Validation Policy

For biological evaluation, prefer a validation set separated from training by:

1. species;
2. chromosome;
3. non-overlapping scaffolds or contigs;
4. another partition that prevents homologous/overlapping windows from crossing splits.

The optional `--valid_fraction` creates a per-source random row split. It is useful for implementation checks and rapid development, but neighboring 40,960-bp windows overlap and can leak nearly identical sequence/labels into both partitions. Do not use such a split as the sole evidence for biological generalization.

The trainer writes `dataset_manifest.json` at startup, recording source paths, row counts, and split information. Preserve this file with reported experiments.

## Metrics and Model Selection

Validation accumulates confusion matrices rather than retaining all nucleotide predictions in memory.

Macro F1 averages only classes with evidence in the validation labels or model
predictions. A class predicted by the model but absent from the labels is
therefore included with F1 equal to `0`; a class absent from both is excluded.
Per-class F1 is calculated directly from accumulated confusion counts as
`2TP / (2TP + FP + FN)`.

Reported metrics include:

- `macro_f1`;
- `nucleotide_accuracy`;
- `cds_f1`;
- `intron_f1`;
- `boundary_f1`;
- per-strand F1;
- per-class F1;
- validation loss.

Supported values for `training.metric_for_best_model` are:

```text
macro_f1
boundary_f1
cds_f1
intron_f1
nucleotide_accuracy
```

The selected metric determines:

- whether a validation result improves the best checkpoint;
- early-stopping patience;
- the final `best/model/` contents.

A checkpoint is saved only when the selected metric improves. Early stopping occurs after the configured number of completed evaluations without improvement.

These are nucleotide/segmentation metrics. Gene-level exact structure, exon-level exact match, BUSCO completeness, and evidence-supported annotation assessment can be performed separately for final biological evaluation.

## Outputs and Checkpoints

Conceptual output layout:

```text
output_dir/
├── resolved_peft_tuning_config.yml       # or half/full equivalent
├── dataset_manifest.json
├── best/
│   ├── metrics.json
│   ├── peft/                             # PEFT/half only
│   │   ├── peft_model.bin
│   │   ├── trainable_parameter_names.json
│   │   └── *_tuning_config.yml
│   ├── model/                            # Always deployable
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── modeling_*.py
│   │   ├── configuration_*.py
│   │   ├── tokenization_*.py
│   │   ├── tokenizer JSON files
│   │   └── resolved tuning configuration
│   └── full/                             # Optional full-tuning archive
│       ├── pytorch_model.bin
│       └── full_tuning_config.yml
└── last/                                 # Only when requested/configured
    └── ...
```

| Path | Purpose |
|---|---|
| `best/model/` | Inference, release, and subsequent adaptation stages |
| `best/peft/` | Compact PEFT/half trainable-weight archive and offline export |
| `best/full/` | Optional full-weight archival bundle |
| `best/metrics.json` | Best-step validation metrics and global step |
| `last/` | Final checkpoint when `--save_last` or `checkpoint.save_last` is enabled |

### Why `best/model/` is deployable

For PEFT and half tuning, LoRA is merged into base projections when the inference model is saved:

```text
W_saved = W + (alpha / rank) × B × A
```

Directly trained decoder/head weights are materialized in the same model state. The resulting directory contains a normal PlantGeneAnn state dictionary without runtime LoRA parametrization keys. Inference therefore does not depend on the tuning package or `best/peft/`.

## Inference with the Tuned Model

One-step annotation:

```bash
python run_annotator.py \
  --genome_file target_species.fa.gz \
  --model_path /data/plantgeneann_peft/best/model \
  --output_file target_species.gff3
```

Two-step prediction:

```bash
python run_model_prediction.py \
  --genome_file target_species.fa.gz \
  --model_path /data/plantgeneann_peft/best/model \
  --cache_path /data/target_prediction
```

Hugging Face loading:

```python
from transformers import AutoModel, AutoTokenizer

model_dir = "/data/plantgeneann_peft/best/model"
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
)
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=True,
)
model.eval()
```

## Export

Rebuild a publishable inference directory without rerunning training.

### PEFT or half tuning

```bash
python -m tuning.export \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --peft_path /data/plantgeneann_peft/best/peft \
  --output_dir /data/published_plantgeneann_model
```

This loads the original base model, restores the trainable bundle, merges LoRA, materializes directly trained weights, and writes a clean Hugging Face directory.

### Full tuning

```bash
python -m tuning.export \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --full_path /data/plantgeneann_full/best/model \
  --output_dir /data/published_plantgeneann_model
```

`--full_path` is mandatory so an untuned `--model_path` cannot be mistaken for the fine-tuned checkpoint.

## Starting Another Adaptation Stage

Every `python -m tuning.train` invocation starts a new optimization run from `--model_path`. The CLI does not restore:

- adapters into an active training session;
- optimizer state;
- scheduler state;
- global step;
- early-stopping state;
- DataLoader iteration state.

To perform a second adaptation stage, pass the previous deployable model explicitly:

```bash
python -m tuning.train \
  --model_path /data/first_stage/best/model \
  --tuning_config tuning/tuning_config/peft_tuning_config.yml \
  --train_dataset /data/second_stage_train \
  --valid_dataset /data/second_stage_valid \
  --output_dir /data/second_stage_output
```

This creates a fresh optimizer, scheduler, random initialization state, and step counter while starting model weights from the previous tuned checkpoint.

## Resource Guidance

### GPU memory

If CUDA out-of-memory occurs:

1. reduce `training.train_batch_size`;
2. reduce `training.eval_batch_size`;
3. increase `gradient_accumulation_steps` to preserve effective batch size;
4. prefer PEFT over half or full tuning;
5. use more GPUs with the same per-GPU batch only if each GPU can still hold its local model/batch;
6. avoid full tuning when optimizer state cannot fit.

### CPU and host memory

`training.num_workers` is per process. For four GPUs and four workers, up to approximately 16 DataLoader workers can be active. Reduce this value if the node has limited cores, memory, shared-memory capacity, or file descriptors.

### Precision

- use BF16 on supported Ampere-or-newer hardware;
- use FP16 only after confirming numerical stability;
- use `no` for diagnostic full-precision runs with substantially higher memory use.

### Strategy selection

A practical progression is:

1. begin with PEFT and strict independent validation;
2. use half tuning if decoder adaptation remains insufficient;
3. use full tuning only with enough high-quality data and hardware;
4. compare improvements on held-out biological units, not only training loss.
