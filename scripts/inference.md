# PlantGeneAnn Inference Guide

This document provides the detailed inference reference for PlantGeneAnn. For installation, model download, benchmarks, and the complete project overview, see the [main README](../README.md).

## Contents

- [Inference workflows](#inference-workflows)
- [Input genome](#input-genome)
- [Model directory](#model-directory)
- [One-step annotation](#one-step-annotation)
- [Two-step annotation](#two-step-annotation)
- [Multi-GPU inference](#multi-gpu-inference)
- [Complete CLI reference](#complete-cli-reference)
- [Sequence windows and record processing](#sequence-windows-and-record-processing)
- [Prediction schema](#prediction-schema)
- [Chromosome-level HDF5](#chromosome-level-hdf5)
- [Semi-Markov model decoding](#semi-markov-model-decoding)
- [Cache and disk usage](#cache-and-disk-usage)
- [Hugging Face model loading](#hugging-face-model-loading)

## Inference Workflows

PlantGeneAnn provides two equivalent routes from a genome assembly to structural gene annotation.

### One-step workflow

```text
Genome FASTA
  → record selection
  → fixed-length windows
  → tokenization
  → strand-specific model inference
  → chromosome-level probabilities
  → 15-state SMM decoding
  → GFF3
```

Use this workflow for routine annotation. Intermediate predictions are automatically removed after decoding.

### Two-step workflow

```text
Step 1: Genome FASTA → chromosome_predictions.h5
Step 2: Genome FASTA + chromosome_predictions.h5 → GFF3
```

Use this workflow when you need to:

- test several decoding thresholds without repeating GPU inference;
- schedule GPU prediction and CPU decoding separately;
- retain nucleotide-level state probabilities;
- inspect or reuse model predictions in downstream analyses.

## Input Genome

Accepted input suffixes include:

- `.fa`
- `.fna`
- `.fasta`
- gzip-compressed variants such as `.fa.gz`, `.fna.gz`, and `.fasta.gz`

Each FASTA record identifier must be unique and non-empty. PlantGeneAnn uses the first whitespace-delimited FASTA identifier as the genomic record key. The same identifiers and sequence lengths must be preserved when a prediction HDF5 is decoded later.

For gzip input, PlantGeneAnn materializes a temporary uncompressed FASTA in the selected cache and builds its random-access index there. Runtime FASTA materializations and indexes are removed by the owning pipeline.

## Model Directory

`--model_path` must identify a complete local Hugging Face model directory. It must contain:

```text
config.json
configuration_caduceus_ph.py
modeling_caduceus_moe.py
modeling_segment_caduceus_v2.py
tokenization_caduceus.py
tokenizer_config.json
special_tokens_map.json
pytorch_model.bin             # or supported Safetensors/sharded weights
```

Download one of the released clade-specific checkpoints with:

```bash
python fetch_model_weights.py
```

The command downloads the Angiospermae checkpoint to
`models/PlantGeneAnn-v2-Angiospermae` by default. Use
`--model_type bryophyta` or `--model_type chlorophyta` for those clades, or
`--model_type all` to download all three checkpoints into separate directories.
Pass the specific clade directory to `--model_path`; `all` is an output mode,
not a model directory. The inference worker loads the selected directory
locally with `AutoModel.from_pretrained(..., trust_remote_code=True,
local_files_only=True)`.
See the [pretrained model table](../README.md#-download-the-pretrained-model)
for the repository and plant-group mapping.

## One-step Annotation

### Minimal command

```bash
python run_annotator.py \
  --genome_file genome.fa.gz \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --output_file annotation.gff3
```

### Resource-controlled command

```bash
python run_annotator.py \
  --genome_file genome.fa.gz \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --output_file annotation.gff3 \
  --cache_path work_cache \
  --chunk_size 3200 \
  --batch_size 16 \
  --num_processes 1 \
  --min_chromosome_size 40960 \
  --min_intron_length 20 \
  --min_cds_length 60 \
  --min_mean_gene_log_odds 0.50 \
  -c 18
```

When `--cache_path auto` is used, the pipeline creates a temporary directory under the repository's `tmp/` directory. Prediction intermediates are deleted after success or failure. With an explicit cache path, pipeline-owned intermediates are also cleaned while the user-provided directory itself is retained.

## Two-step Annotation

### Step 1: Model prediction

```bash
python run_model_prediction.py \
  --genome_file genome.fa.gz \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --cache_path prediction_cache \
  --output_h5 prediction_cache/chromosome_predictions.h5 \
  --chunk_size 3200 \
  --batch_size 16 \
  --num_processes 1 \
  --min_chromosome_size 40960 \
  -c 18
```

If `--output_h5` is omitted, output defaults to:

```text
<cache_path>/chromosome_predictions.h5
```

The output path is replaced atomically after a completed inference run. A stale file at the same output path is removed before prediction begins.

### Step 2: SMM decoding

```bash
python run_prediction_decoding.py \
  --genome_file genome.fa.gz \
  --chromosome_h5 prediction_cache/chromosome_predictions.h5 \
  --output_file annotation.gff3 \
  --min_intron_length 20 \
  --min_cds_length 60 \
  --min_mean_gene_log_odds 0.50 \
  -c 18
```

You may repeat this step using different thresholds and output paths without loading the neural network.

The decoder verifies that:

- the HDF5 uses the current completed 15-state schema;
- each HDF5 record exists in the FASTA;
- FASTA and HDF5 record lengths agree;
- probability datasets have the required shape and dtype;
- prediction coverage is complete.

Legacy 5-state caches cannot be converted to the current format and must be regenerated.

## Multi-GPU Inference

The public commands invoke Accelerate automatically. One process is normally used per visible GPU.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_annotator.py \
  --genome_file genome.fa.gz \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --output_file annotation.gff3 \
  --num_processes 4 \
  --batch_size 16 \
  -c 36
```

The same options apply to `run_model_prediction.py`.

Resource behavior:

- `--num_processes` controls the number of Accelerate inference workers;
- omitting it selects the visible GPU count;
- `--batch_size` is applied per process/GPU;
- `-c` / `--num_cpu_threads` is a total CPU budget, not a per-rank budget;
- DataLoader workers are derived from the CPU budget and process count;
- model inference uses BF16 mixed precision;
- global rank zero restores deterministic window order and writes the HDF5.

Multi-GPU acceleration applies primarily to model inference. FASTA processing, tokenization, storage I/O, candidate scanning, and SMM decoding can limit end-to-end scaling.

## Complete CLI Reference

### `run_annotator.py`

| Option | Type | Default | Description |
|---|---|---:|---|
| `-i`, `--genome_file` | path | required | Genome FASTA, optionally gzip-compressed |
| `-m`, `--model_path` | path | required | Local model directory |
| `-o`, `--output_file` | path | required | Output `.gff` or `.gff3` |
| `-c`, `--num_cpu_threads` | integer | process CPU affinity | Total CPU-thread budget |
| `--chunk_size` | integer | 3,200 | Number of windows per inference chunk |
| `--batch_size` | integer | 16 | Per-process inference batch size |
| `--num_processes` | integer | visible GPU count | Accelerate worker count |
| `--cache_path` | path or `auto` | `auto` | Runtime cache directory |
| `--min_chromosome_size` | integer | 40,960 | Minimum retained FASTA-record length |
| `--min_intron_length` | integer | 20 | Minimum decoded intron length |
| `--min_cds_length` | integer | 60 | Minimum complete decoded CDS length |
| `--min_mean_gene_log_odds` | float | 0.50 | Minimum span-normalized gene score |
| `-v`, `--verbose` | flag | off | Enable debug logging |

### `run_model_prediction.py`

| Option | Type | Default | Description |
|---|---|---:|---|
| `-i`, `--genome_file` | path | required | Genome FASTA, optionally gzip-compressed |
| `-m`, `--model_path` | path | required | Local model directory |
| `--cache_path` | path | required | Dedicated persistent working directory |
| `--output_h5` | path | `<cache_path>/chromosome_predictions.h5` | Completed chromosome probability HDF5 |
| `-c`, `--num_cpu_threads` | integer | process CPU affinity | Tokenization/DataLoader CPU budget |
| `--chunk_size` | integer | 3,200 | Number of windows per inference chunk |
| `--batch_size` | integer | 16 | Per-process inference batch size |
| `--num_processes` | integer | visible GPU count | Accelerate worker count |
| `--min_chromosome_size` | integer | 40,960 | Minimum retained record length |
| `-v`, `--verbose` | flag | off | Enable debug logging |

### `run_prediction_decoding.py`

| Option | Type | Default | Description |
|---|---|---:|---|
| `-i`, `--genome_file` | path | required | Genome FASTA used for prediction |
| `--chromosome_h5` | path | required | Completed 15-state probability HDF5 |
| `-o`, `--output_file` | path | required | Output `.gff` or `.gff3` |
| `--cache_path` | path | HDF5 directory | Optional decoder/FASTA runtime cache |
| `--min_intron_length` | integer | 20 | Minimum decoded intron length |
| `--min_cds_length` | integer | 60 | Minimum complete decoded CDS length |
| `--min_mean_gene_log_odds` | float | 0.50 | Minimum span-normalized gene score |
| `-c`, `--num_cpu_threads` | integer | process CPU affinity | Candidate-scanning and decoding CPU budget |
| `-v`, `--verbose` | flag | off | Enable debug logging |

Candidate scanning uses at most 20 worker processes. Region-level SMM decoding may use the remaining available CPU budget.

## Sequence Windows and Record Processing

The public inference pipeline uses:

- input sequence length: 40,960 bp;
- context flank configured by the inference pipeline: 5,120 bp per side;
- genomic center interval written per regular window: 30,720 bp;
- sequence padding: `N` at genomic-record boundaries;
- chunk size: 3,200 windows by default.

Every FASTA record at least `--min_chromosome_size` bases long is processed, regardless of its identifier. Chromosome, scaffold, contig, alternate, haplotype, unplaced, and random records are not filtered by name. The center intervals tile each processed record continuously. A terminal window is aligned to the end of the record, and duplicate terminal overlap is removed during direct HDF5 writing. Every processed genomic base receives one prediction with no gaps.

## Prediction Schema

For every position and strand, PlantGeneAnn stores a complete 15-class softmax distribution.

| ID | State name | Meaning |
|---:|---|---|
| 0 | `background` | Intergenic/background |
| 1 | `intron_phase_0` | Intron state 0 |
| 2 | `intron_phase_1` | Intron state 1 |
| 3 | `intron_phase_2` | Intron state 2 |
| 4 | `cds_frame_0` | CDS frame state 0 |
| 5 | `cds_frame_1` | CDS frame state 1 |
| 6 | `cds_frame_2` | CDS frame state 2 |
| 7 | `start` | First base of the start codon |
| 8 | `donor_phase_0` | Donor state 0 |
| 9 | `donor_phase_1` | Donor state 1 |
| 10 | `donor_phase_2` | Donor state 2 |
| 11 | `acceptor_phase_0` | Acceptor state 0 |
| 12 | `acceptor_phase_1` | Acceptor state 1 |
| 13 | `acceptor_phase_2` | Acceptor state 2 |
| 14 | `stop` | Last base of the stop codon |

The 30 model channels are ordered as:

```text
channels 0–14:  positive-strand logits
channels 15–29: negative-strand logits
```

Softmax is applied independently to each 15-channel strand block. Negative-strand labels are defined in transcript direction.

## Chromosome-level HDF5

The current prediction file contract is:

```text
file_format: plantgeneann_chromosome_level_predictions
file_format_version: 2
prediction_schema: plantgeneann_15state_transcript_v2
status: complete
coordinate_system: 0-based half-open genomic coordinates
probability_dtype: float16
probability_normalization: per_strand_softmax
strand_axis: 0=positive;1=negative
```

Conceptual layout:

```text
/
├── chromosome_index/
│   ├── chrom_id
│   ├── chrom_group
│   ├── chrom_length
│   ├── chrom_index
│   └── num_windows
└── chromosomes/
    └── <encoded_record_id>/
        └── full_probabilities   # shape: (2, record_length, 15)
```

The prediction stage writes to `<output>.tmp`, verifies completion metadata, and atomically promotes it to the requested output path. An interrupted temporary file is never accepted as a completed cache.

The output is uncompressed in the public CLI. Its primary payload is approximately 60 bytes per retained genomic base:

```text
2 strands × 15 probabilities × 2 bytes/float16
```

## Semi-Markov Model Decoding

PlantGeneAnn uses strict 15-state SMM decoding to transform nucleotide probabilities into complete gene models. The decoder:

1. validates the HDF5 and FASTA manifest;
2. scans probability tracks for candidate genic regions;
3. expands and merges supported candidate intervals;
4. reads candidate sequences and probabilities in bounded regions;
5. performs segmental dynamic programming under the 15-state gene grammar;
6. enforces intron and complete-CDS minimum durations;
7. retains complete genes meeting the mean gene log-odds threshold;
8. calculates transcript-direction CDS phases and writes GFF3.

### Decoder parameters

#### `--min_intron_length`

Hard minimum intron duration accepted by the SMM. Default: 20 bp.

Increase this value only when biologically justified for the target lineage. An excessively large value removes genuine short introns; an excessively small value can permit implausible gene paths.

#### `--min_cds_length`

Hard minimum total CDS length for a complete decoded gene. Default: 60 bp.

This constraint is enforced inside decoding rather than as a post hoc GFF filter.

#### `--min_mean_gene_log_odds`

Minimum additive SMM gene score divided by complete genomic gene span. Default: 0.50.

Higher values generally produce more conservative annotations. Lower values may recover weak predictions while increasing false positives. Calibrate this parameter using independent annotated chromosomes, contigs, or species whenever available.

## Cache and Disk Usage

### One-step annotation

With `--cache_path auto`, temporary data are created under `tmp/` and removed after the run. With an explicit path, the directory remains but pipeline-owned extraction, tokenization, FASTA-runtime, and prediction artifacts are cleaned.

### Two-step prediction

The final HDF5 is caller-owned and retained. Temporary sequence TSVs, tokenized Dataset chunks, Hugging Face runtime caches, and materialized/indexed FASTA runtime files are removed after prediction. Use a dedicated `--cache_path` to avoid mixing unrelated files with pipeline-managed artifacts.

### Capacity planning

Reserve space for:

- a temporary uncompressed FASTA when input is gzip-compressed;
- tokenized chunks during model preparation;
- the temporary HDF5 written during inference;
- the final HDF5 after atomic promotion.

The implementation removes an old final output before creating the new temporary HDF5, avoiding simultaneous storage of two completed probability files.

## Hugging Face Model Loading

PlantGeneAnn checkpoints, including fine-tuned `best/model/` exports, can be loaded directly:

```python
from transformers import AutoModel, AutoTokenizer

model_dir = "models/PlantGeneAnn-v2-Angiospermae"

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

The public inference pipeline remains the recommended interface because it implements the required sequence geometry, metadata preservation, multi-GPU gathering, chromosome-level probability assembly, and SMM decoding.
