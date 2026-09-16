# PlantGeneAnn Training-data Preprocessing Guide

This document describes how to convert genome FASTA and structural GFF/GFF3 annotations into Hugging Face Datasets accepted by the PlantGeneAnn fine-tuning pipeline. For project installation and the end-to-end overview, see the [main README](../README.md). For training instructions, see the [fine-tuning guide](finetuning.md).

## Contents

- [Purpose](#purpose)
- [Input directory layout](#input-directory-layout)
- [Input requirements](#input-requirements)
- [Running preprocessing](#running-preprocessing)
- [CLI reference](#cli-reference)
- [Default configuration](#default-configuration)
- [Representative transcript selection](#representative-transcript-selection)
- [Biological quality control](#biological-quality-control)
- [Loss weights and overlap handling](#loss-weights-and-overlap-handling)
- [15-state label construction](#15-state-label-construction)
- [Window construction](#window-construction)
- [Dataset schema](#dataset-schema)
- [Output files](#output-files)
- [Multi-species behavior](#multi-species-behavior)
- [Data-quality recommendations](#data-quality-recommendations)

## Purpose

The `preprocess` package transforms one or more species-level FASTA/GFF pairs into fixed-length, strand-specific training examples. It performs:

1. strict species-input discovery;
2. genome and GFF parsing;
3. representative-transcript selection;
4. ORF and gene-structure QC;
5. strand-specific 15-state labeling;
6. loss-mask assignment;
7. fixed-length window construction;
8. tokenization with the selected PlantGeneAnn model;
9. atomic Hugging Face Dataset writing;
10. QC and provenance reporting.

Preprocessing is intended for high-quality protein-coding annotations whose coordinates refer to exactly the same assembly version as the supplied FASTA.

## Input Directory Layout

The input root must contain one first-level directory per species. Every species directory must contain **exactly one** supported FASTA and **exactly one** supported GFF/GFF3 file.

```text
input_root/
├── Species_A/
│   ├── genome.fa
│   └── annotation.gff3
├── Species_B/
│   ├── assembly.fna.gz
│   └── genes.gff.gz
└── Species_C/
    ├── reference.fasta
    └── reference.gff3.gz
```

Hidden first-level directories are ignored. Files nested below a species directory are not recursively discovered.

Supported FASTA suffixes:

```text
.fa
.fna
.fasta
.fa.gz
.fna.gz
.fasta.gz
```

Supported annotation suffixes:

```text
.gff
.gff3
.gff.gz
.gff3.gz
```

Discovery fails before species processing if any visible species directory has zero or multiple files of either type.

## Input Requirements

### FASTA

- Record identifiers must be unique.
- Sequences should represent the same assembly release used by the annotation.
- Sequence content is normalized to uppercase.
- Windows containing too many non-`A/T/C/G` bases are excluded according to configuration.

### GFF/GFF3

The parser recognizes gene, transcript/mRNA, exon, CDS, and intron relationships through feature IDs and `Parent` attributes. For the most reliable results:

- provide `gene → mRNA/transcript → exon/CDS` hierarchy;
- ensure GFF sequence IDs exactly match FASTA record IDs;
- provide `+` or `-` strands for protein-coding genes and transcripts;
- provide valid CDS phases (`0`, `1`, or `2`);
- use 1-based inclusive GFF coordinates;
- retain gene/transcript biotype attributes where available;
- ensure coordinates lie within the corresponding FASTA record.

If a transcript has CDS features but biotype attributes are absent, the pipeline can infer protein-coding evidence. This fallback is intended for otherwise valid filtered annotations and should not replace correct source metadata when it is available.

## Running Preprocessing

Run from the PlantGeneAnn repository root:

```bash
python -m preprocess.cli \
  --input_dir /data/species_inputs \
  --output_dir /data/training_datasets \
  --model_path models/PlantGeneAnn-v2-Angiospermae
```

Use a custom preprocessing configuration with:

```bash
python -m preprocess.cli \
  --input_dir /data/species_inputs \
  --output_dir /data/training_datasets \
  --model_path models/PlantGeneAnn-v2-Angiospermae \
  --config /path/to/preprocess_config.yml \
  --overwrite \
  --verbose
```

The model directory is required because preprocessing tokenizes genomic windows with the same Hugging Face tokenizer used by the base model. The examples use the Angiospermae checkpoint; select the matching `PlantGeneAnn-v2-Bryophyta` or `PlantGeneAnn-v2-Chlorophyta` directory for those plant groups. See the [pretrained model table](../README.md#-download-the-pretrained-model) for the downloader options.

## CLI Reference

| Option | Required | Description |
|---|---|---|
| `--input_dir` | yes | Root containing one first-level directory per species |
| `--output_dir` | yes | Root for per-species Datasets, QC, and manifests |
| `--model_path` | yes | Local PlantGeneAnn model/tokenizer directory |
| `--config` | no | Custom YAML configuration; defaults to `preprocess/default_preprocess_config.yml` |
| `--overwrite` | no | Replace an existing species Dataset output |
| `--stop_on_error` | no | Stop the species loop after the first processing failure |
| `--verbose` | no | Enable debug logging |

Without `--stop_on_error`, a failed species is reported and the next discovered species is processed. The command returns a non-zero exit code if any species fails.

## Default Configuration

The distributed default is `preprocess/default_preprocess_config.yml`:

```yaml
format_version: 1

window:
  length: 40960
  overlap: 8192
  flank_length: 4096
  min_record_length: 32768
  max_non_atcg: 327

qc:
  gene_biotype_key: gene_biotype
  protein_coding_values: [protein_coding, protein-coding]
  multi_transcript: longest_cds
  min_cds_length: 60
  hard_min_intron_length: 4
  soft_min_intron_length: 20
  start_codons: [ATG]
  stop_codons: [TAA, TAG, TGA]
  intron_source: exon
  allow_partial: false
  allow_annotation_exceptions: false
  noncanonical_splice_policy: hard
  low_quality_policy: half

labels:
  reset_mask0_labels_to_background: true
  overlap_conflict_policy: error

output:
  shard_rows: 1000
  write_qc_beds: true
  write_qc_tsv: true

workers:
  qc: 8
  chromosomes: 4
  tokenization: 16
```

Worker counts are reduced automatically if they exceed the CPUs available to the current process.

### Window options

| Key | Default | Meaning |
|---|---:|---|
| `window.length` | 40,960 | Genomic bases before adding tokenizer special tokens |
| `window.overlap` | 8,192 | Overlap between neighboring training windows |
| `window.flank_length` | 4,096 | Context bases removed from each side before loss; must match the model configuration |
| `window.min_record_length` | 32,768 | Minimum original FASTA record length eligible for window generation |
| `window.max_non_atcg` | 327 | Maximum non-ATCG bases accepted in the model center/loss interval |

### QC options

| Key | Default | Meaning |
|---|---|---|
| `gene_biotype_key` | `gene_biotype` | Preferred gene biotype attribute |
| `protein_coding_values` | two common spellings | Values recognized as protein coding |
| `multi_transcript` | `longest_cds` | Representative-transcript policy |
| `min_cds_length` | 60 | Minimum total CDS length |
| `hard_min_intron_length` | 4 | Shorter introns produce loss weight 0.0 |
| `soft_min_intron_length` | 20 | Introns from hard minimum to this value produce weight 0.5 |
| `start_codons` | `ATG` | Accepted start codons |
| `stop_codons` | `TAA`, `TAG`, `TGA` | Accepted terminal stop codons |
| `intron_source` | `exon` | Features used to infer intron intervals |
| `allow_partial` | false | Whether partial markers are permitted |
| `allow_annotation_exceptions` | false | Whether annotation exceptions are permitted |
| `noncanonical_splice_policy` | `hard` | Reject splice pairs outside the strict SMM decoder contract |
| `low_quality_policy` | `half` | Assign low-quality proteins weight 0.5 |

## Representative Transcript Selection

PlantGeneAnn uses a **select-before-QC** policy.

For each protein-coding gene:

1. collect transcripts with CDS features; if none have CDS, use the available transcript list for deterministic failure reporting;
2. select one representative transcript according to `qc.multi_transcript`;
3. perform biological QC only on that selected transcript;
4. use its structure and QC result for training labels and loss weights.

With the default `longest_cds` policy, the selected transcript maximizes, in order:

1. total CDS length;
2. number of CDS blocks;
3. earlier genomic start;
4. transcript identifier as a deterministic final tie-breaker.

Supported policies in the implementation include:

- `longest_cds`: deterministic longest-CDS selection;
- `first`: first transcript by genomic coordinates and identifier;
- `phytozome_longest`: prefer a uniquely marked `longest=1` transcript, otherwise fall back to longest CDS;
- `fail`: mark a gene with multiple CDS-bearing transcripts as unusable.

The pipeline does **not** QC all isoforms and then select among passing transcripts. This distinction is important when interpreting mask summaries.

## Biological Quality Control

QC checks the selected transcript and its relationship to the genome assembly.

### Metadata and hierarchy checks

Potential zero-weight reasons include:

- partial gene or CDS markers when partial structures are disallowed;
- annotation exceptions or `transl_except` when exceptions are disallowed;
- pseudo markers;
- missing CDS;
- transcript/gene sequence-ID mismatch;
- invalid strand;
- invalid, out-of-bounds, overlapping, or touching features.

Low-quality protein markers receive weight 0.5 by default.

### CDS and ORF checks

QC evaluates:

- minimum total CDS length;
- CDS length divisible by three;
- valid `A/C/G/T` CDS sequence;
- accepted start codon;
- accepted terminal stop codon;
- absence of internal in-frame stop codons;
- CDS coordinates within the genome;
- valid CDS phase fields;
- internal CDS blocks longer than two bases.

CDS phase inconsistencies are recorded as repaired notes. Label generation follows the transcript structure and model-state grammar rather than blindly copying inconsistent source phases.

### Intron checks

With `intron_source: exon`, introns are inferred from consecutive exons. Other implemented modes can use explicit introns, CDS blocks, or automatic source selection.

QC examines:

- positive intron length;
- hard and soft minimum intron thresholds;
- coordinates within the genome;
- ambiguous intronic bases;
- donor and acceptor availability;
- decoder-supported `GT-AG`, `GC-AG`, and `AT-AC` splice pairs in transcript direction.

Only the three splice pairs accepted by the strict SMM decoder are valid for training. Any other unambiguous donor/acceptor combination is classified as `splice_pair_not_supported_by_decoder`; the selected transcript and its gene receive loss weight `0.0`. Ambiguous splice boundaries and structurally invalid introns are also hard failures.

## Loss Weights and Overlap Handling

Each selected protein-coding gene receives one of three supervision weights:

| Weight | Interpretation |
|---:|---|
| `1.0` | Passes hard and soft QC criteria |
| `0.5` | Usable but contains a configured soft-quality condition |
| `0.0` | Fails a hard structural or ORF criterion |

The loss mask is strand specific. At positions covered by multiple mask rules, the minimum applicable weight is retained.

With `reset_mask0_labels_to_background: true`:

- a selected transcript with weight 0.0 is not painted as a gene structure;
- its affected strand interval is reset to background labels;
- its interval receives zero loss weight and contributes no training objective.

### Same-strand coding overlaps

One categorical state path cannot represent two simultaneous protein-coding structures on the same strand. After historical transcript QC is recorded, preprocessing detects selected genes whose non-background label states overlap on the same FASTA record and strand.

Every gene involved in such a conflict is removed from label painting and masked from supervision. The original QC report remains available, with conflict handling represented in the label-safe selection state. Opposite-strand overlaps remain valid because positive and negative strands occupy independent channels.

A window is discarded if either strand is entirely masked across its center/loss
interval, because training computes a separate objective for each strand.

## 15-state Label Construction

Each training window contains a label matrix with shape:

```text
(2, 40960)
```

where channel 0 is the positive strand and channel 1 is the negative strand.

| ID | Label | Meaning |
|---:|---|---|
| 0 | background | Intergenic/background |
| 1–3 | intron phase states | Intron positions indexed by transition state |
| 4–6 | CDS frame states | Coding positions indexed by codon frame |
| 7 | start | First base of translation start codon |
| 8–10 | donor states | Exon-to-intron splice boundary states |
| 11–13 | acceptor states | Intron-to-exon splice boundary states |
| 14 | stop | Last base of translation stop codon |

Negative-strand labels are generated in transcript direction while stored at forward genomic coordinates. State suffixes follow the model transition grammar and are not identical to GFF3 phase values.

Label painting rejects unresolved same-strand overlaps. The separate pre-paint conflict pass normally masks these genes before window construction.

## Window Construction

Training windows use the model's center-output geometry and the same
start/terminal boundary strategy as inference.

Default preprocessing geometry:

- window length: 40,960 bp;
- overlap: 20,480 bp;
- flank length: 4,096 bp on each side;
- center/loss interval: 32,768 bp;
- regular input step: 20,480 bp (capped at the center length to prevent gaps);
- maximum non-ATCG positions in the center/loss interval: 408;
- labels: `(2, 40960)`;
- attention/loss mask: `(2, 40960)`.

For each FASTA record, the first input window starts at `-4,096`, so its
center interval begins at genomic coordinate `0`. Regular windows advance by
the configured step over center intervals. A terminal window is right-aligned
to the record: its center interval ends at the record length, and the final
4,096 input positions are synthetic `N` context. Records shorter than one
center interval can be padded by the window builder, but are excluded by the
default `window.min_record_length: 32768` filter. Increase or lower this
threshold explicitly when that behavior is desired. Synthetic padding is
assigned loss weight `0.0` and is excluded from the non-ATCG filter for records
that pass the minimum-length check.
Consequently, every real base lies in the supervised center interval of at
least one retained window (subject to the ordinary annotation-mask and quality
filters).

Each window preserves:

- species name;
- FASTA record identifier;
- 0-based half-open window start and end;
- deterministic window identifier;
- tokenized input sequence;
- labels and loss masks.

## Dataset Schema

Each saved Dataset contains the columns required by tuning plus provenance columns.

| Column | Shape/type | Description |
|---|---|---|
| `input_ids` | `[L+2]` integer IDs | Tokenized sequence with CLS and SEP; default length 40,962 |
| `labels` | `[2, L]` int8 | Positive- and negative-strand 15-state labels |
| `attention_mask` | `[2, L]` float | Per-position loss weights of 0.0, 0.5, or 1.0 |
| `species` | string | Source species directory name |
| `chrom_id` | string | FASTA record identifier |
| `window_start` | integer | Input interval start in record coordinates; may be negative for the initial padded window |
| `window_end` | integer | Input interval end in record coordinates; may exceed the record length for terminal padding |
| `window_id` | string | Deterministic provenance identifier |

Despite its name, `attention_mask` is the training loss-weight mask, not the conventional one-dimensional tokenizer padding mask.

The tuning loader performs lightweight first-row geometry and required-field
checks, casts supervised labels as needed, and removes provenance columns before
forwarding batches to the model. It does not scan the complete Dataset or inspect
loss-mask values; preprocessing is responsible for those data-quality decisions.

## Output Files

Output layout:

```text
output_root/
├── Species_A/
│   ├── dataset/                         # Dataset.save_to_disk output
│   ├── qc/
│   │   ├── transcript_loss_mask.tsv
│   │   ├── selected_transcripts.tsv
│   │   ├── genes.mask0.bed
│   │   ├── genes.mask05.bed
│   │   └── summary.json
│   ├── resolved_preprocess_config.yml
│   └── preprocess_manifest.json
├── Species_B/
│   └── ...
├── preprocess_batch_manifest.json
└── preprocess_failures.tsv
```

### QC files

- `transcript_loss_mask.tsv`: selected-transcript QC measurements, reasons, notes, and weights;
- `selected_transcripts.tsv`: one representative-transcript decision per gene;
- `genes.mask0.bed`: zero-weight gene intervals;
- `genes.mask05.bed`: half-weight gene intervals;
- `summary.json`: counts of protein-coding genes/transcripts and weights.

BED coordinates are 0-based and half-open.

### Provenance files

- `resolved_preprocess_config.yml`: complete resolved configuration after worker-count adjustment;
- `preprocess_manifest.json`: species input paths, row count, FASTA-record count, and parsed-gene count;
- `preprocess_batch_manifest.json`: all successful and failed species;
- `preprocess_failures.tsv`: concise species-level failures.

Review `summary.json`, the reason columns in `transcript_loss_mask.tsv`, and the mask BED files before training. Unexpectedly high zero-weight or half-weight fractions usually indicate assembly/annotation mismatch, unsupported annotation conventions, or low source-annotation quality.

## Multi-species Behavior

Species are processed serially in stable directory-name order. This design limits memory pressure when assemblies are large.

Within one species:

- gene QC can use multiple Linux fork workers;
- chromosome/window construction can use multiple workers;
- tokenization can use multiple workers;
- output shards are written and combined into one Dataset.

The implementation relies on Linux `fork` behavior to share read-only genome data efficiently during QC and window construction. The documented environment is Linux.

The complete preprocessing output root can be passed directly to training. The tuning loader recursively discovers nested, complete `Dataset.save_to_disk()` directories and concatenates them in stable path order.

## Data-quality Recommendations

For reliable adaptation:

1. use FASTA and GFF files from exactly the same assembly release;
2. prefer chromosome- or scaffold-level assemblies with low contamination;
3. use manually curated or otherwise high-confidence protein-coding annotations;
4. retain valid gene/transcript/CDS hierarchy and phases;
5. avoid annotations dominated by partial genes, pseudogenes, or automated low-quality predictions;
6. inspect mask reasons before accepting a Dataset;
7. separate training and validation by species, chromosome, or non-overlapping contigs;
8. do not create biological validation claims from random overlapping-window splits;
9. avoid using the final benchmark chromosomes during preprocessing for training;
10. record the source assembly and annotation versions alongside generated manifests.

The training sampler is window-proportional after datasets are concatenated. A species with many windows therefore contributes more updates unless the user constructs balanced inputs externally.
