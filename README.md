# 🌱 PlantGeneAnn: Plant Gene Annotator

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.38.1-yellow.svg)](https://huggingface.co/docs/transformers/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.06.25.733695-b31b1b.svg)](https://doi.org/10.64898/2026.06.25.733695)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-PlantGeneAnn--v2--Angiospermae-FFD21E.svg)](https://huggingface.co/qzzhang/PlantGeneAnn-v2-Angiospermae/tree/main)

**PlantGeneAnn** (Plant Gene Annotator) is a model for accurate **ab initio protein-coding gene structure annotation in plant genomes**. It predicts nucleotide-level gene states independently on both genomic strands and converts them into biologically valid gene models through **semi-Markov model (SMM) decoding**.

PlantGeneAnn accepts a plant genome in FASTA format and directly produces a standard GFF3 annotation. It does not require RNA-seq alignments, protein homology, or an existing genome annotation.

## 📑 Table of Contents

- [🗂️ Repository Structure](#-repository-structure)
- [🛠️ Installation](#-installation)
- [⬇️ Download the Pretrained Model](#-download-the-pretrained-model)
- [🚀 Quick Start](#-quick-start)
- [🧬 One-step Annotation](#-one-step-annotation)
- [🔬 Two-step Annotation](#-two-step-annotation)
- [🖥️ Multi-GPU Inference](#-multi-gpu-inference)
- [💻 Hardware Requirements](#-hardware-requirements)
- [⏱️ Runtime & Memory Consumption](#-runtime-memory-consumption)
- [🎛️ Custom Fine-tuning](#-custom-fine-tuning)
- [📚 Documentation](#-documentation)
- [📖 Citation](#-citation)
- [📜 License](#-license)
- [✉️ Contact](#-contact)

## 🗂️ Repository Structure

```text
PlantGeneAnn/
├── run_annotator.py                 # One-step FASTA-to-GFF3 annotation
├── run_model_prediction.py          # FASTA-to-probability-HDF5 prediction
├── run_prediction_decoding.py       # Probability-HDF5-to-GFF3 SMM decoding
├── fetch_model_weights.py           # Pretrained model downloader
├── annotator.py                     # Internal Accelerate inference worker
├── requirements.txt
├── src/                             # Inference and SMM decoding implementation
├── preprocess/                      # Fine-tuning data preprocessing
├── tuning/                          # PEFT, half, and full fine-tuning
├── scripts/
│   ├── inference.md                 # Detailed inference documentation
│   ├── preprocessing.md             # Detailed preprocessing documentation
│   ├── finetuning.md                # Detailed fine-tuning documentation
│   └── species.md                   # PlantGeneAnn training species and assemblies
├── example/
│   └── Arabidopsis_thaliana.TAIR10.dna.chromosome.5.fa.gz
└── LICENSE
```

Run the commands in this README from the PlantGeneAnn repository root. `annotator.py` is an internal worker launched by the public inference entry points and normally should not be invoked directly.

## 🛠️ Installation

PlantGeneAnn requires Linux, an NVIDIA CUDA GPU, and CUDA extensions provided by [mamba-ssm](https://github.com/state-spaces/mamba), [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d), and [FlashAttention](https://github.com/Dao-AILab/flash-attention).

```bash
# 1. Clone the repository and create the environment
git clone https://github.com/qzzhang0131/PlantGeneAnn.git
cd PlantGeneAnn
conda create -n PlantGeneAnn python=3.8 -y
conda activate PlantGeneAnn

# 2. Install the CUDA toolkit and Python dependencies
conda install -c nvidia -c conda-forge cuda-toolkit=12.1.0 libxcrypt -y
pip install -r requirements.txt

# 3. Compile the core CUDA libraries (typically 10–20 minutes)
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
MAX_JOBS=4 pip install \
  causal-conv1d==1.2.0.post2 \
  mamba-ssm==1.2.0.post1 \
  flash-attn==2.5.6 \
  --no-build-isolation
```

The provided environment pins PyTorch 2.2.2 with CUDA 12.1 wheels, Transformers 4.38.1, Accelerate 0.32.1, Datasets 2.15.0, and Triton 2.2.0. `MAX_JOBS` controls parallel compilation and may be reduced on machines with limited CPU memory.

After installation, verify that the GPU is visible:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## ⬇️ Download the Pretrained Model

Download the default (Angiospermae) checkpoint with:

```bash
python fetch_model_weights.py
```

PlantGeneAnn v2 provides three clade-specific pretrained checkpoints:

| `--model_type` | Hugging Face repository | Adapted plant group | Default local directory |
|---|---|---|---|
| `angiospermae` | [`qzzhang/PlantGeneAnn-v2-Angiospermae`](https://huggingface.co/qzzhang/PlantGeneAnn-v2-Angiospermae/tree/main) | Angiospermae (flowering plants) | `models/PlantGeneAnn-v2-Angiospermae` |
| `bryophyta` | [`qzzhang/PlantGeneAnn-v2-Bryophyta`](https://huggingface.co/qzzhang/PlantGeneAnn-v2-Bryophyta/tree/main) | Bryophyta (mosses, liverworts, and hornworts) | `models/PlantGeneAnn-v2-Bryophyta` |
| `chlorophyta` | [`qzzhang/PlantGeneAnn-v2-Chlorophyta`](https://huggingface.co/qzzhang/PlantGeneAnn-v2-Chlorophyta/tree/main) | Chlorophyta (green algae) | `models/PlantGeneAnn-v2-Chlorophyta` |

> [!NOTE]
> The Chlorophyta model is currently a development version.

The downloader tries [Hugging Face](https://huggingface.co/) first, falls back to [HF Mirror](https://hf-mirror.com/) when needed, and validates every downloaded model. If no `--model_type` is supplied, only the Angiospermae checkpoint is downloaded.

```bash
# Select a clade, or download all three checkpoints
python fetch_model_weights.py --model_type angiospermae
python fetch_model_weights.py --model_type bryophyta
python fetch_model_weights.py --model_type chlorophyta
python fetch_model_weights.py --model_type all
python fetch_model_weights.py --model_type all --model_dir /path/to/models

# Optional: choose an endpoint, output directory, or fresh download
python fetch_model_weights.py --endpoint mirror
python fetch_model_weights.py --model_dir /path/to/PlantGeneAnn-model
python fetch_model_weights.py --force
```

With `--model_type all`, `--model_dir` is treated as an output root and the
three model directories are created below it. For a single model, `--model_dir`
is the final model directory. Match the model directory to the clade of the
target genome; the inference and fine-tuning commands below use the default
Angiospermae directory.

For advanced use, `--repo_id` can override the repository for a single selected
model; it cannot be combined with `--model_type all`.

## 🚀 Quick Start

The repository includes *Arabidopsis thaliana* TAIR10 chromosome 5 as a test genome:

```bash
python run_annotator.py \
  -i example/Arabidopsis_thaliana.TAIR10.dna.chromosome.5.fa.gz \
  -m models/PlantGeneAnn-v2-Angiospermae \
  -o Arabidopsis_chr5.PlantGeneAnn.gff3
```

This command performs sequence extraction, tokenization, GPU inference, chromosome-level probability assembly, 15-state SMM decoding, and GFF3 writing. With the default automatic cache, runtime intermediates are removed after the pipeline finishes.

See [the detailed inference guide](scripts/inference.md) for the complete CLI reference and data-format specifications.

## 🧬 One-step Annotation

Use `run_annotator.py` for the recommended end-to-end workflow:

```bash
python run_annotator.py \
  -i genome.fa.gz \
  -m models/PlantGeneAnn-v2-Angiospermae \
  -o annotation.gff3 \
  -c 18 \
  --batch_size 16 \
  --num_processes 1 
```

### Common options

| Option | Default | Description |
|---|---:|---|
| `-i`, `--genome_file` | required | Input FASTA; gzip compression is supported |
| `-m`, `--model_path` | required | Local PlantGeneAnn Hugging Face model directory |
| `-o`, `--output_file` | required | Output path ending in `.gff` or `.gff3` |
| `-c`, `--num_cpu_threads` | available CPU affinity | Total CPU-thread budget for the pipeline |
| `--chunk_size` | 3,200 | Number of sequence windows in each inference chunk; not a length in bp |
| `--batch_size` | 16 | Per-process GPU inference batch size |
| `--num_processes` | visible GPUs | Number of Accelerate inference processes |
| `--cache_path` | `auto` | Cache directory; automatic caches are deleted after use |
| `--min_chromosome_size` | 40,960 | Skip FASTA records shorter than this number of bases |
| `--min_intron_length` | 20 | Hard minimum intron length accepted by SMM decoding |
| `--min_cds_length` | 60 | Hard minimum complete CDS length accepted by SMM decoding |
| `--min_mean_gene_log_odds` | 0.50 | Minimum span-normalized SMM gene score for GFF3 output |
| `-v`, `--verbose` | off | Enable debug-level logging |

> [!NOTE]
> `--num_cpu_threads`/`-c` must not exceed the number of physical CPU threads; otherwise, the decoding process may crash.

Increasing `--min_mean_gene_log_odds` generally makes annotation more conservative, whereas decreasing it may recover lower-confidence genes at the cost of additional false positives.

## 🔬 Two-step Annotation

The two-step workflow separates GPU prediction from CPU SMM decoding. It is useful when testing several decoding thresholds, scheduling inference and decoding independently, or retaining nucleotide-level probabilities for downstream analysis.

### Step 1: Generate chromosome-level probabilities

```bash
python run_model_prediction.py \
  -i genome.fa.gz \
  -m models/PlantGeneAnn-v2-Angiospermae \
  -c 18 \
  --cache_path prediction_cache \
  --output_h5 prediction_cache/chromosome_predictions.h5 \
  --batch_size 16 \
  --num_processes 1
```

If `--output_h5` is omitted, the default output is:

```text
<cache_path>/chromosome_predictions.h5
```

The completed HDF5 stores a `float16` probability array of shape `(2, genomic_record_length, 15)` for every processed FASTA record. Probabilities are independently normalized over the 15 states for each strand.

### Step 2: Decode probabilities into GFF3

```bash
python run_prediction_decoding.py \
  -i genome.fa.gz \
  -c 18 \
  --chromosome_h5 prediction_cache/chromosome_predictions.h5 \
  --output_file annotation.gff3
```

The decoding stage does not load the neural network and does not repeat GPU inference. You may run it multiple times with different thresholds and output paths.

> [!IMPORTANT]
> Use the same genome assembly for prediction and decoding. Every processed FASTA record identifier and length must agree with the HDF5 manifest. Legacy 5-state prediction caches are not compatible with the current 15-state schema and must be regenerated.

### Disk-space considerations

Two-step annotation stores a chromosome-level, uncompressed probability HDF5. The probability payload contains:

```text
2 strands × 15 states × 2 bytes (float16) = approximately 60 bytes/base
```

A genome with 1 billion (1GB) processed bases therefore requires approximately 60 GB for the probability payload alone, excluding HDF5 metadata and other temporary storage. Check available local disk space before annotating large genomes with the two-step workflow.

## 🖥️ Multi-GPU Inference

PlantGeneAnn supports multi-GPU data-parallel inference through [Hugging Face Accelerate](https://github.com/huggingface/accelerate). The public inference entry points launch the internal worker automatically; do not run `annotator.py` manually.

With `N` comparable GPUs, the model-inference component is expected to approach `1/N` of the single-GPU reference time. End-to-end scaling can be lower because sequence preparation, storage I/O, and SMM decoding also contribute to runtime.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_annotator.py \
  -i genome.fa.gz \
  -m models/PlantGeneAnn-v2-Angiospermae \
  -c 36 \
  -o annotation.gff3 \
  --num_processes 4 \
  --batch_size 16
```

Resource semantics:

- `CUDA_VISIBLE_DEVICES` controls which GPUs are available;
- `--num_processes` is normally set to the number of visible GPUs;
- `--batch_size` is the batch size **per inference process/GPU**;
- `-c` / `--num_cpu_threads` is one total CPU-thread budget shared by preprocessing and decoding, not a per-GPU allocation;
- if `--num_processes` is omitted, PlantGeneAnn uses the visible GPU count.

If inference runs out of GPU memory, reduce `--batch_size`. If host memory or process pressure is excessive, reduce `-c` / `--num_cpu_threads`.

## 💻 Hardware Requirements

The following hardware configuration applies to **PlantGeneAnn inference**:

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Linux (x86-64) | Linux (x86-64) |
| CUDA | 12.1 | 12.1 |
| GPU | NVIDIA RTX 3060 | NVIDIA RTX 4090 |
| CPU | Intel/AMD ≥ 4-core CPU | Intel/AMD 16-core CPU |
| System Memory | 32 GB | 48 GB |

PlantGeneAnn inference requires an **NVIDIA GPU with Ampere architecture or newer**, e.g., RTX 30-series, RTX 40-series, NVIDIA A40/A100, or NVIDIA L20/H20/H100. The minimum RTX 3060 and recommended RTX 4090 both meet this requirement. The inference pipeline uses BF16 mixed precision by default. CPU resources are used for FASTA indexing, sequence extraction, tokenization, DataLoader workers, candidate-region scanning, and SMM decoding.

## ⏱️ Runtime & Memory Consumption

The following end-to-end reference annotation times were measured using **one NVIDIA RTX 4090 GPU** and **18 vCPU cores from an AMD EPYC 9754**.

| Species | Genome Size | Memory Consumption | Runtime |
|---|---:|---:|---:|
| *Arabidopsis thaliana* | 138 MB | 9.97 GB | 14.3 min |
| *Oryza sativa* | 373 MB | 16.72 GB | 37.2 min |
| *Cannabis sativa* | 744 MB | 18.28 GB | 1.26 h |
| *Elaeis guineensis* | 1.74 GB | 16.49 GB | 3.01 h |
| *Papaver somniferum* | 2.57 GB | 18.73 GB | 4.40 h |
| *Hordeum vulgare* | 3.98 GB | 13.19 GB | 6.81 h |

Runtime and memory consumption depends on the number and length of FASTA records, storage throughput, CPU allocation, GPU/CPU model, and the number of candidate regions processed during SMM decoding. These values are reference measurements rather than guaranteed runtimes and memory consumptions.

## 🎛️ Custom Fine-tuning

The selected pretrained checkpoint may be fine-tuned when its default predictions are suboptimal for the target species. The complete list of training species, assemblies, and data sources is available in the [training species table](scripts/species.md). Fine-tuning is particularly useful when the target species is evolutionarily distant from the training species or has distinct genome composition and gene structure.

Fine-tuning GPU requirements depend on the selected strategy:

| Tuning Mode | Minimum GPU VRAM* | Minimum GPU | Recommended GPU |
|---|---:|---|---|
| PEFT tuning | 24 GB | RTX 3090 | RTX 4090 |
| Half tuning | 24 GB | RTX 3090 | RTX 4090 |
| Full tuning | 32 GB | A40 | A800/A100 |

\* Minimum practical VRAM per GPU with `training.train_batch_size=1`; it is not the combined memory of all GPUs. **Multi-GPU parallel fine-tuning is recommended**.

The workflow consists of:

1. preparing one or more species-specific Hugging Face Datasets from FASTA/GFF pairs;
2. selecting PEFT, half, or full-parameter tuning;
3. training and selecting the best validation checkpoint;
4. using the exported `best/model/` directory with the standard annotation commands.

### 1. Prepare FASTA/GFF training data

The input root must contain one first-level directory per species. Each species directory must contain exactly one FASTA and one GFF/GFF3 file.

```text
species_inputs/
├── Species_A/
│   ├── genome.fa
│   └── annotation.gff3
└── Species_B/
    ├── genome.fna.gz
    └── genes.gff3.gz
```

Supported FASTA suffixes are `.fa`, `.fna`, `.fasta`, and their gzip variants. Supported annotation suffixes are `.gff`, `.gff3`, and their gzip variants.

```bash
python -m preprocess.cli \
  --input_dir /data/species_inputs \
  --output_dir /data/training_datasets \
  --model_path models/PlantGeneAnn-v2-Angiospermae
```

Preprocessing writes a Hugging Face Dataset for each species together with QC reports, BED masks, resolved configuration, and provenance manifests. See [the preprocessing guide](scripts/preprocessing.md) for the complete input contract and QC criteria.

### 2. Select a tuning strategy

| Strategy | Configuration | Trainable scope | Typical use |
|---|---|---|---|
| PEFT | `tuning/tuning_config/peft_tuning_config.yml` | Mamba-LoRA plus selected decoder/head components | Limited labels or rapid adaptation |
| Half | `tuning/tuning_config/half_tuning_config.yml` | Mamba-LoRA plus the complete decoder and prediction head | Intermediate compute and stronger decoder adaptation |
| Full | `tuning/tuning_config/full_tuning_config.yml` | All model parameters | Larger high-quality datasets and sufficient GPU memory |

### 3. Train

Single-GPU training:

```bash
python -m tuning.train \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --tuning_config tuning/tuning_config/peft_tuning_config.yml \
  --train_dataset /data/training_datasets \
  --valid_dataset /data/heldout_training_datasets \
  --output_dir /data/tuning_output
```

Multi-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes 4 \
  -m tuning.train \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --tuning_config tuning/tuning_config/peft_tuning_config.yml \
  --train_dataset /data/training_datasets \
  --valid_dataset /data/heldout_training_datasets \
  --output_dir /data/tuning_output
```

Use a biologically independent validation dataset whenever possible. A held-out species, chromosome, or non-overlapping set of contigs is preferable to a random window split because overlapping genomic windows can cause information leakage.

The tuning CLI accepts either `--valid_dataset` or `--valid_fraction`; these options are mutually exclusive and must not be supplied together.

Each tuning YAML exposes one method-level optimizer policy. PEFT, half, and
full default to learning rate `1e-4` and weight decay `0.01`; choose either
`cosine` or `linear` through `training.scheduler`. Bias and normalization
parameters use the standard AdamW no-decay groups. PEFT and half use
`training.warmup_ratio: 0.05`, while full tuning uses `0.025`; all three
configurations default to BF16, per-process train/evaluation batches of `1`/`8`,
gradient accumulation `16`, and `20` epochs.

Training reports masked segmentation metrics including macro F1, nucleotide accuracy, CDS F1, intron F1, boundary F1, and per-strand/per-class F1. The metric configured by `training.metric_for_best_model` controls early stopping and best-checkpoint selection.

### 4. Annotate with the tuned model

Always use the deployable model under `best/model/`:

```bash
python run_annotator.py \
  --genome_file target_species.fa \
  --model_path /data/tuning_output/best/model \
  --output_file target_species.PlantGeneAnn.gff3
```

For PEFT and half tuning, `best/peft/` is an archival trainable-weight bundle, whereas `best/model/` contains merged, inference-ready Hugging Face weights. Detailed configuration, checkpoint, loss, metric, and export documentation is available in [the fine-tuning guide](scripts/finetuning.md).

## 📚 Documentation

| Document | Contents |
|---|---|
| [Inference guide](scripts/inference.md) | One-step and two-step annotation, complete CLI reference, multi-GPU inference, HDF5 schema, SMM decoding, and decoder parameters |
| [Preprocessing guide](scripts/preprocessing.md) | FASTA/GFF input requirements, representative-transcript selection, biological QC, loss masks, labels, windows, and Dataset outputs |
| [Fine-tuning guide](scripts/finetuning.md) | PEFT/half/full strategies, configuration, single- and multi-GPU training, validation, metrics, checkpoints, export, and tuned-model inference |
| [Training species table](scripts/species.md) | Training species and assembly versions |

## 📖 Citation

If you use PlantGeneAnn in your research, please cite our preprint:

> Zhang, Q., Zhang, Z., Lin, K., Wang, J., Deng, K., Xiang, X., Xu, W., & Hu, X. (2026). PlantGeneAnn: a strand-specific genome foundation model for ab initio gene structure annotation of plant genomes. *bioRxiv*. https://doi.org/10.64898/2026.06.25.733695

```bibtex
@article{zhang2026plantgeneann,
  title={PlantGeneAnn: a strand-specific genome foundation model for ab initio gene structure annotation of plant genomes},
  author={Zhang, Qizhe and Zhang, Zhengyang and Lin, Kepeng and Wang, Jing and Deng, Kaixuan and Xiang, Xianglei and Xu, Wei and Hu, Xuehai},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.06.25.733695},
  url={https://doi.org/10.64898/2026.06.25.733695}
}
```

## 📜 License

PlantGeneAnn is released under the [MIT License](LICENSE). See the `LICENSE` file for details.

## ✉️ Contact

For questions or suggestions regarding the PlantGeneAnn code and pretrained models, contact:

**qzzhang@webmail.hzau.edu.cn**
