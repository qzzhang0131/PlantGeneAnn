"""Strict configuration models for PlantGeneAnn training-data preprocessing."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Type, TypeVar
import os
import yaml

T = TypeVar("T")


def _strict(cls: Type[T], value: Mapping[str, Any], name: str) -> T:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {name}: {unknown}")
    return cls(**dict(value))


@dataclass(frozen=True)
class WindowConfig:
    length: int = 40960
    overlap: int = 20480
    max_non_atcg: int = 408
    # The model consumes the full input window but computes supervision only
    # on the center interval after removing this context from both sides.
    flank_length: int = 4096
    # Records shorter than one center interval are not useful training sources
    # by default, even though the window builder can pad them to model length.
    min_record_length: int = 32768

    def __post_init__(self):
        if self.length <= 0 or self.length % 16:
            raise ValueError("window.length must be positive and divisible by 16")
        if not 0 <= self.overlap < self.length:
            raise ValueError("window.overlap must be in [0, length)")
        if self.flank_length < 0 or 2 * self.flank_length >= self.length:
            raise ValueError(
                "window.flank_length must be non-negative and leave a non-empty "
                "center interval"
            )
        if self.min_record_length <= 0:
            raise ValueError("window.min_record_length must be positive")
        if not 0 <= self.max_non_atcg <= self.length:
            raise ValueError("window.max_non_atcg must be in [0, length]")

    @property
    def step(self) -> int:
        return self.length - self.overlap

    @property
    def center_length(self) -> int:
        return self.length - 2 * self.flank_length

    @property
    def tiling_step(self) -> int:
        """Return a step that cannot leave gaps in center-output coverage."""
        return min(self.step, self.center_length)


@dataclass(frozen=True)
class QcConfig:
    """QC options matching the historical ``qc_gene_loss_mask.py`` CLI."""

    gene_biotype_key: str = "gene_biotype"
    protein_coding_values: tuple[str, ...] = ("protein_coding", "protein-coding")
    multi_transcript: str = "longest_cds"
    min_cds_length: int = 60
    hard_min_intron_length: int = 4
    soft_min_intron_length: int = 20
    start_codons: tuple[str, ...] = ("ATG",)
    stop_codons: tuple[str, ...] = ("TAA", "TAG", "TGA")
    intron_source: str = "exon"
    allow_partial: bool = False
    allow_annotation_exceptions: bool = False
    noncanonical_splice_policy: str = "hard"
    low_quality_policy: str = "half"

    def __post_init__(self):
        object.__setattr__(self, "protein_coding_values", tuple(self.protein_coding_values))
        object.__setattr__(self, "start_codons", tuple(x.upper() for x in self.start_codons))
        object.__setattr__(self, "stop_codons", tuple(x.upper() for x in self.stop_codons))
        if self.multi_transcript not in {"longest_cds", "phytozome_longest", "first", "fail"}:
            raise ValueError("qc.multi_transcript must be longest_cds, phytozome_longest, first, or fail")
        if not self.protein_coding_values:
            raise ValueError("qc.protein_coding_values cannot be empty")
        if self.min_cds_length <= 0:
            raise ValueError("qc.min_cds_length must be positive")
        if self.hard_min_intron_length < 1 or self.soft_min_intron_length < self.hard_min_intron_length:
            raise ValueError("QC intron thresholds are inconsistent")
        if self.intron_source not in {"exon", "gff", "cds", "auto"}:
            raise ValueError("qc.intron_source must be exon, gff, cds, or auto")
        if self.noncanonical_splice_policy != "hard":
            raise ValueError(
                "qc.noncanonical_splice_policy must be hard so preprocessing "
                "matches the strict SMM splice-pair contract"
            )
        if self.low_quality_policy not in {"half", "hard", "note"}:
            raise ValueError("qc.low_quality_policy must be half, hard, or note")


@dataclass(frozen=True)
class LabelsConfig:
    reset_mask0_labels_to_background: bool = True
    overlap_conflict_policy: str = "error"

    def __post_init__(self):
        if self.overlap_conflict_policy != "error":
            raise ValueError("Only labels.overlap_conflict_policy=error is supported")


@dataclass(frozen=True)
class OutputConfig:
    shard_rows: int = 1000
    write_qc_beds: bool = True
    write_qc_tsv: bool = True

    def __post_init__(self):
        if self.shard_rows <= 0:
            raise ValueError("output.shard_rows must be positive")


@dataclass(frozen=True)
class WorkersConfig:
    qc: int = 8
    chromosomes: int = 4
    tokenization: int = 16

    def __post_init__(self):
        if min(self.qc, self.chromosomes, self.tokenization) <= 0:
            raise ValueError("All workers values must be positive")

    def resolved(self) -> "WorkersConfig":
        try:
            available = max(1, len(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            available = max(1, os.cpu_count() or 1)
        return WorkersConfig(**{f.name: min(getattr(self, f.name), available) for f in fields(self)})


@dataclass(frozen=True)
class PreprocessConfig:
    format_version: int = 1
    window: WindowConfig = WindowConfig()
    qc: QcConfig = QcConfig()
    labels: LabelsConfig = LabelsConfig()
    output: OutputConfig = OutputConfig()
    workers: WorkersConfig = WorkersConfig()

    def __post_init__(self):
        for name, cls in (("window", WindowConfig), ("qc", QcConfig), ("labels", LabelsConfig), ("output", OutputConfig), ("workers", WorkersConfig)):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, _strict(cls, value, name))
        if self.format_version != 1:
            raise ValueError("Unsupported preprocess format_version")


def load_config(path: str | None = None) -> PreprocessConfig:
    config_path = Path(path) if path else Path(__file__).with_name("default_preprocess_config.yml")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise TypeError("Preprocess YAML root must be a mapping")
    return _strict(PreprocessConfig, data, "root")
