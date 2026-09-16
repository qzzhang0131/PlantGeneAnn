"""Chromosome-level candidate scanning, parallel decoding, and GFF3 output."""

from __future__ import annotations

import atexit
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple
from urllib.parse import quote

import h5py
import numpy as np
from tqdm import tqdm

from .chromosome_writer import (
    get_chromosome_probability_dataset,
    load_chromosome_h5_manifest,
    validate_chromosome_h5_header,
)
from .configuration import (
    CANDIDATE_SCAN_MAX_WORKERS,
    DEFAULT_DECODER_MIN_CDS_LENGTH,
    DEFAULT_DECODER_MIN_INTRON_LENGTH,
    DEFAULT_DECODER_MIN_MEAN_GENE_LOG_ODDS,
)
from .constants import (
    CODING_LABELS,
    INTRON_LABELS,
    PREDICTION_NUM_CLASSES,
    PREDICTION_NUM_STRANDS,
)
from .runtime import (
    cleanup_cached_fasta_index,
    cleanup_materialized_genome_fasta,
    configure_single_threaded_libraries,
    prepare_cached_fasta_index_path,
    prepare_genome_fasta,
)
from .segmental_core import (
    DEFAULT_EMISSION_CALIBRATION,
    NUMBA_AVAILABLE,
    EmissionCalibration,
    GenePrediction,
    _validate_calibration,
)
from .segmental_core import (
    decode_region_predictions as _decode_region_predictions_core,
)

logger = logging.getLogger("PlantGeneAnn.src.segmental_decoder")

CANDIDATE_BIN_SIZE = 50
CANDIDATE_SEED_MAX_CODING = 0.50
CANDIDATE_SEED_MEAN_CODING = 0.20
CANDIDATE_GROW_MEAN_GENIC = 0.05
CANDIDATE_GROW_MAX_CODING = 0.25
CANDIDATE_GAP_TOLERANCE_BP = 500
DEFAULT_CANDIDATE_REGION_BUFFER = 2000
CANDIDATE_BACKGROUND_BARRIER_BP = 1000
CANDIDATE_MAX_TERMINAL_EXTENSION_BP = 10000
CANDIDATE_SAMPLE_FRACTION_NUMERATOR = 1
CANDIDATE_SAMPLE_FRACTION_DENOMINATOR = 100
CANDIDATE_SAMPLE_MIN_BP = 512_000
CANDIDATE_SAMPLE_MAX_BP = 2_000_000
CANDIDATE_SAMPLE_TARGET_STRATUM_BP = 5_000_000
CANDIDATE_SAMPLE_MIN_STRATA = 16
CANDIDATE_SAMPLE_MAX_STRATA = 64
CANDIDATE_SAMPLE_WHOLE_RECORD_NUMERATOR = 4
CANDIDATE_SAMPLE_WHOLE_RECORD_DENOMINATOR = 5
CANDIDATE_SAMPLE_READ_BLOCK_BP = 250_000
# Each parallel candidate task processes this many of the exact bin-aligned
# blocks used by the serial scanner. Keeping the inner block boundaries intact
# preserves every floating-point reduction and threshold comparison.
CANDIDATE_BLOCKS_PER_TASK = 2

_DNA_COMPLEMENT_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")
SOURCE = "PlantGeneAnn"


@dataclass(frozen=True)
class _CandidateSamplingPlan:
    """Deterministic, chromosome-wide sampling geometry for one record."""

    sequence_length: int
    intervals: Tuple[Tuple[int, int], ...]

    @property
    def sampled_bp(self) -> int:
        return sum(end - start for start, end in self.intervals)


@dataclass(frozen=True)
class _CandidateThresholdStatistics:
    """Coding-distribution statistics and derived thresholds for one strand."""

    thresholds: Tuple[float, float, float]


def _thresholds_from_coding_statistics(
    coding_mean: float,
    coding_std: float,
) -> Tuple[float, float, float]:
    """Apply the existing adaptive-threshold formula to supplied moments."""

    coding_mean = float(coding_mean)
    coding_std = float(coding_std)
    if not np.isfinite(coding_mean) or not np.isfinite(coding_std):
        raise ValueError("Candidate coding statistics must be finite.")
    if coding_std < 0.0:
        raise ValueError("Candidate coding standard deviation cannot be negative.")
    return (
        max(CANDIDATE_SEED_MAX_CODING, coding_mean + 1.5 * coding_std),
        max(CANDIDATE_SEED_MEAN_CODING, coding_mean + 0.5 * coding_std),
        max(CANDIDATE_GROW_MEAN_GENIC, coding_mean - 0.5 * coding_std),
    )


def _build_candidate_sampling_plan(sequence_length: int) -> _CandidateSamplingPlan:
    """Build deterministic, non-overlapping sample windows across a record.

    The target is 1% of the record, bounded to 512 kb--2 Mb.  Records for which
    that target covers at least 80% are sampled in full.  Otherwise 16--64 equal
    strata contribute one centered, 50-bp-aligned window apiece.
    """

    length = int(sequence_length)
    if length < 0:
        raise ValueError("Candidate sampling length cannot be negative.")
    if length == 0:
        return _CandidateSamplingPlan(0, ())

    fractional_bp = (
        length * CANDIDATE_SAMPLE_FRACTION_NUMERATOR
        + CANDIDATE_SAMPLE_FRACTION_DENOMINATOR
        - 1
    ) // CANDIDATE_SAMPLE_FRACTION_DENOMINATOR
    target_bp = min(
        length,
        max(
            CANDIDATE_SAMPLE_MIN_BP,
            min(fractional_bp, CANDIDATE_SAMPLE_MAX_BP),
        ),
    )
    if (
        target_bp * CANDIDATE_SAMPLE_WHOLE_RECORD_DENOMINATOR
        >= length * CANDIDATE_SAMPLE_WHOLE_RECORD_NUMERATOR
    ):
        return _CandidateSamplingPlan(length, ((0, length),))

    num_strata = min(
        CANDIDATE_SAMPLE_MAX_STRATA,
        max(
            CANDIDATE_SAMPLE_MIN_STRATA,
            (
                length
                + CANDIDATE_SAMPLE_TARGET_STRATUM_BP
                - 1
            ) // CANDIDATE_SAMPLE_TARGET_STRATUM_BP,
        ),
    )
    window_bp = (
        target_bp // num_strata // CANDIDATE_BIN_SIZE
    ) * CANDIDATE_BIN_SIZE
    if window_bp < CANDIDATE_BIN_SIZE:
        raise RuntimeError(
            "Candidate sampling produced a window shorter than one bin."
        )

    intervals: List[Tuple[int, int]] = []
    for stratum_index in range(num_strata):
        stratum_start = stratum_index * length // num_strata
        stratum_end = (stratum_index + 1) * length // num_strata
        max_start = stratum_end - window_bp
        if max_start < stratum_start:
            raise RuntimeError("Candidate sample window exceeds its stratum.")

        center = (stratum_start + stratum_end) // 2
        centered_start = center - window_bp // 2
        minimum_aligned_start = (
            (stratum_start + CANDIDATE_BIN_SIZE - 1)
            // CANDIDATE_BIN_SIZE
            * CANDIDATE_BIN_SIZE
        )
        maximum_aligned_start = (
            max_start // CANDIDATE_BIN_SIZE * CANDIDATE_BIN_SIZE
        )
        if minimum_aligned_start > maximum_aligned_start:
            raise RuntimeError(
                "Candidate sampling stratum has no bin-aligned window position."
            )
        # Choose the nearest globally aligned position (integer half-up
        # rounding), then clamp within the stratum's legal aligned range.
        # This avoids Python's banker rounding and cannot shift a window across
        # a stratum boundary.
        start = (
            (centered_start + CANDIDATE_BIN_SIZE // 2)
            // CANDIDATE_BIN_SIZE
            * CANDIDATE_BIN_SIZE
        )
        start = max(minimum_aligned_start, min(start, maximum_aligned_start))
        intervals.append((start, start + window_bp))

    return _CandidateSamplingPlan(length, tuple(intervals))


def _compute_stratified_threshold_statistics(
    dataset,
    plan: Optional[_CandidateSamplingPlan] = None,
) -> Tuple[_CandidateThresholdStatistics, ...]:
    """Stream one shared sampling plan and return statistics for every strand."""

    if len(dataset.shape) != 3 or dataset.shape[2] != PREDICTION_NUM_CLASSES:
        raise ValueError("Candidate threshold dataset must have shape (S, L, 15).")
    sequence_length = int(dataset.shape[1])
    if plan is None:
        plan = _build_candidate_sampling_plan(sequence_length)
    if plan.sequence_length != sequence_length:
        raise ValueError("Candidate sampling plan and dataset lengths differ.")
    if not plan.intervals:
        raise ValueError("Candidate threshold sampling requires a non-empty record.")

    previous_end = 0
    for interval_index, interval in enumerate(plan.intervals):
        if len(interval) != 2:
            raise ValueError("Candidate sampling intervals must be (start, end) pairs.")
        start, end = interval
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Candidate sampling interval coordinates must be integers.")
        if start < 0 or start >= end or end > sequence_length:
            raise ValueError("Candidate sampling interval is outside the record.")
        if interval_index and start < previous_end:
            raise ValueError("Candidate sampling intervals must not overlap.")
        previous_end = end
    if CANDIDATE_SAMPLE_READ_BLOCK_BP <= 0:
        raise ValueError("Candidate sampling read block must be positive.")

    num_strands = int(dataset.shape[0])
    counts = np.zeros(num_strands, dtype=np.int64)
    means = np.zeros(num_strands, dtype=np.float64)
    m2 = np.zeros(num_strands, dtype=np.float64)

    for start, end in plan.intervals:
        # Full-record plans are also streamed; no sampling geometry may trigger
        # a large ``dataset[:, :, :]`` materialization in the parent process.
        for block_start in range(start, end, CANDIDATE_SAMPLE_READ_BLOCK_BP):
            block_end = min(end, block_start + CANDIDATE_SAMPLE_READ_BLOCK_BP)
            block = np.asarray(dataset[:, block_start:block_end, :])
            expected_shape = (
                num_strands,
                block_end - block_start,
                PREDICTION_NUM_CLASSES,
            )
            if block.shape != expected_shape:
                raise RuntimeError(
                    "Candidate threshold sample returned shape {}, expected {}."
                    .format(block.shape, expected_shape)
                )
            for strand_index in range(num_strands):
                coding = np.zeros(block_end - block_start, dtype=np.float32)
                for label in CODING_LABELS:
                    coding += block[strand_index, :, label]
                if not np.all(np.isfinite(coding)):
                    raise ValueError(
                        "Candidate threshold sample contains non-finite values."
                    )

                block_count = int(len(coding))
                block_mean = float(np.mean(coding, dtype=np.float64))
                block_variance = float(np.var(coding, dtype=np.float64))
                previous_count = int(counts[strand_index])
                combined_count = previous_count + block_count
                delta = block_mean - float(means[strand_index])
                means[strand_index] += delta * block_count / combined_count
                m2[strand_index] += (
                    block_variance * block_count
                    + delta * delta * previous_count * block_count
                    / combined_count
                )
                counts[strand_index] = combined_count

    statistics: List[_CandidateThresholdStatistics] = []
    for strand_index in range(num_strands):
        sample_count = int(counts[strand_index])
        if sample_count != plan.sampled_bp:
            raise RuntimeError(
                "Candidate threshold sampling consumed an unexpected number of bases."
            )
        coding_mean = float(means[strand_index])
        coding_std = float(np.sqrt(max(float(m2[strand_index]) / sample_count, 0.0)))
        statistics.append(
            _CandidateThresholdStatistics(
                thresholds=_thresholds_from_coding_statistics(
                    coding_mean,
                    coding_std,
                ),
            )
        )
    return tuple(statistics)


def decode_region_predictions(
    sequence: str,
    predictions: np.ndarray,
    min_intron_length: int = DEFAULT_DECODER_MIN_INTRON_LENGTH,
    min_cds_length: int = DEFAULT_DECODER_MIN_CDS_LENGTH,
    calibration: EmissionCalibration = DEFAULT_EMISSION_CALIBRATION,
    chrom_id: str = "region",
    strand: str = "+",
    region_offset: int = 0,
    chrom_length: Optional[int] = None,
    use_numba: Optional[bool] = None,
) -> List[GenePrediction]:
    """Compatibility wrapper around the pure region-level decoder.

    Keeping the dispatcher in this driver preserves the historical behavior in
    which callers can temporarily disable the compiled path by replacing this
    module's ``NUMBA_AVAILABLE`` binding.
    """

    if use_numba is None and not NUMBA_AVAILABLE:
        use_numba = False
    return _decode_region_predictions_core(
        sequence=sequence,
        predictions=predictions,
        min_intron_length=min_intron_length,
        min_cds_length=min_cds_length,
        calibration=calibration,
        chrom_id=chrom_id,
        strand=strand,
        region_offset=region_offset,
        chrom_length=chrom_length,
        use_numba=use_numba,
    )


@dataclass(frozen=True)
class _DecoderContext:
    min_intron_length: int
    min_cds_length: int
    class_scale: Tuple[float, ...]
    class_bias: Tuple[float, ...]
    gene_intercept: float
    epsilon: float
    calibration_version: str


@dataclass(frozen=True)
class _CandidateTrack:
    """Immutable scan geometry for one genomic record and one strand."""

    track_index: int
    chrom_id: str
    group_name: str
    strand: str
    strand_index: int
    chrom_length: int
    block_size: int
    thresholds: Tuple[float, float, float]

    @property
    def num_bins(self) -> int:
        return (
            self.chrom_length + CANDIDATE_BIN_SIZE - 1
        ) // CANDIDATE_BIN_SIZE

    @property
    def shard_size(self) -> int:
        return self.block_size * CANDIDATE_BLOCKS_PER_TASK


@dataclass(frozen=True)
class _CandidateScanTask:
    """Small worker request for consecutive, whole serial scan blocks."""

    track_index: int
    chrom_id: str
    group_name: str
    strand: str
    strand_index: int
    chrom_length: int
    forward_start: int
    forward_end: int
    block_size: int
    thresholds: Tuple[float, float, float]


@dataclass(frozen=True)
class _CandidateScanResult:
    """Bin flags returned for one non-overlapping candidate scan shard."""

    track_index: int
    forward_start: int
    seed: np.ndarray
    support: np.ndarray


@dataclass(frozen=True)
class _RegionTask:
    """Small, pickle-friendly description of a region held in HDF5/FASTA."""

    chrom_id: str
    group_name: str
    strand: str
    strand_index: int
    forward_start: int
    forward_end: int
    chrom_length: int

    @property
    def region_start(self) -> int:
        if self.strand == "+":
            return self.forward_start
        return self.chrom_length - self.forward_end

    @property
    def region_end(self) -> int:
        if self.strand == "+":
            return self.forward_end
        return self.chrom_length - self.forward_start

    @property
    def length(self) -> int:
        return self.forward_end - self.forward_start


@dataclass(frozen=True)
class _RegionResult:
    chrom_id: str
    strand: str
    region_start: int
    genes: Tuple[GenePrediction, ...]


_WORKER_CONTEXT: Optional[_DecoderContext] = None
_WORKER_H5 = None
_WORKER_FASTA = None


def _reverse_complement_sequence(sequence: str) -> str:
    return sequence.translate(_DNA_COMPLEMENT_TABLE)[::-1].upper()


def _escape_gff3_attribute(value: object) -> str:
    """URL-escape one GFF3 attribute value."""

    return quote(str(value), safe=".:^*$@!+_?-|")


def _format_attributes(attributes: Mapping[str, object]) -> str:
    """Format attributes as a semicolon-separated GFF3 key-value field."""

    return ";".join(
        f"{key}={_escape_gff3_attribute(value)}"
        for key, value in attributes.items()
    )


def gff3_line(
    *,
    seqid: str,
    feature_type: str,
    start0: int,
    end0: int,
    strand: str,
    phase: str,
    attributes: Mapping[str, object],
    source: str = SOURCE,
    score: str = ".",
) -> str:
    """Create one strict 9-column GFF3 line from a half-open interval."""

    if start0 < 0:
        raise ValueError(f"GFF3 feature start cannot be negative: {start0}")
    if end0 <= start0:
        raise ValueError(
            f"GFF3 feature interval must be non-empty: {start0}-{end0}"
        )
    return "\t".join(
        [
            seqid,
            source,
            feature_type,
            str(start0 + 1),
            str(end0),
            score,
            strand,
            phase,
            _format_attributes(attributes),
        ]
    )

def _summarize_candidate_bins(
    predictions: np.ndarray,
    seed_max_threshold: float = CANDIDATE_SEED_MAX_CODING,
    seed_mean_threshold: float = CANDIDATE_SEED_MEAN_CODING,
    support_genic_threshold: float = CANDIDATE_GROW_MEAN_GENIC,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized bin summary with configurable thresholds.

    Optimization 6: Supports adaptive thresholds based on global statistics.
    """

    probabilities = np.asarray(predictions)
    if probabilities.ndim != 2 or probabilities.shape[1] != PREDICTION_NUM_CLASSES:
        raise ValueError("Candidate detection requires an (L, 15) array.")
    length = int(probabilities.shape[0])
    if length == 0:
        return np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.bool_)

    # Do not use a channel reduction: sequential float32 additions preserve the
    # exact old threshold semantics for float16 source probabilities.
    coding = np.zeros(length, dtype=np.float32)
    intron = np.zeros(length, dtype=np.float32)
    for label in CODING_LABELS:
        coding += probabilities[:, label]
    for label in INTRON_LABELS:
        intron += probabilities[:, label]
    genic = np.clip(coding + intron, 0.0, 1.0)

    full_bins = length // CANDIDATE_BIN_SIZE
    num_bins = (length + CANDIDATE_BIN_SIZE - 1) // CANDIDATE_BIN_SIZE
    mean_coding = np.empty(num_bins, dtype=np.float32)
    max_coding = np.empty(num_bins, dtype=np.float32)
    mean_genic = np.empty(num_bins, dtype=np.float32)
    if full_bins:
        shape = (full_bins, CANDIDATE_BIN_SIZE)
        mean_coding[:full_bins] = coding[:full_bins * CANDIDATE_BIN_SIZE].reshape(shape).mean(axis=1)
        max_coding[:full_bins] = coding[:full_bins * CANDIDATE_BIN_SIZE].reshape(shape).max(axis=1)
        mean_genic[:full_bins] = genic[:full_bins * CANDIDATE_BIN_SIZE].reshape(shape).mean(axis=1)
    if full_bins < num_bins:
        start = full_bins * CANDIDATE_BIN_SIZE
        mean_coding[full_bins] = coding[start:].mean()
        max_coding[full_bins] = coding[start:].max()
        mean_genic[full_bins] = genic[start:].mean()

    # Use adaptive thresholds (defaults to original constants if not provided)
    seed = (max_coding >= seed_max_threshold) | (
        mean_coding >= seed_mean_threshold
    )
    support = seed | (mean_genic >= support_genic_threshold) | (
        max_coding >= CANDIDATE_GROW_MAX_CODING
    )
    return seed, support


def _regions_from_candidate_bins(
    seed: np.ndarray, support: np.ndarray, length: int
) -> List[Tuple[int, int]]:
    """Apply gap filling and region boundaries to precomputed bin flags."""

    seed = np.asarray(seed, dtype=np.bool_)
    support = np.asarray(support, dtype=np.bool_).copy()
    unsupported = np.flatnonzero(~support)
    if unsupported.size:
        index = 0
        while index < len(unsupported):
            run_start = int(unsupported[index])
            run_end_index = index + 1
            while (
                run_end_index < len(unsupported)
                and unsupported[run_end_index] == unsupported[run_end_index - 1] + 1
            ):
                run_end_index += 1
            run_end = int(unsupported[run_end_index - 1]) + 1
            if (
                run_start > 0
                and run_end < len(support)
                and (run_end - run_start) * CANDIDATE_BIN_SIZE
                <= CANDIDATE_GAP_TOLERANCE_BP
            ):
                support[run_start:run_end] = True
            index = run_end_index

    regions: List[Tuple[int, int]] = []
    index = 0
    while index < len(support):
        if not support[index]:
            index += 1
            continue
        start_bin = index
        while index < len(support) and support[index]:
            index += 1
        end_bin = index
        if np.any(seed[start_bin:end_bin]):
            regions.append(
                (
                    start_bin * CANDIDATE_BIN_SIZE,
                    min(length, end_bin * CANDIDATE_BIN_SIZE),
                )
            )
    return regions


def _resolve_candidate_block_size(dataset, block_size: Optional[int] = None) -> int:
    """Resolve the exact bin-aligned block geometry used by all scan paths."""

    if block_size is None:
        chunk_bp = int(dataset.chunks[1]) if dataset.chunks else 262144
        block_size = int(np.lcm(chunk_bp, CANDIDATE_BIN_SIZE))
    block_size = max(CANDIDATE_BIN_SIZE, int(block_size))
    block_size -= block_size % CANDIDATE_BIN_SIZE
    return block_size


def _stream_candidate_bins(
    dataset,
    strand_index: int,
    block_size: Optional[int] = None,
    seed_max_threshold: float = CANDIDATE_SEED_MAX_CODING,
    seed_mean_threshold: float = CANDIDATE_SEED_MEAN_CODING,
    support_genic_threshold: float = CANDIDATE_GROW_MEAN_GENIC,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read one HDF5 strand in bin-aligned blocks with adaptive thresholds.

    Optimization 6: Supports adaptive thresholds for candidate filtering.
    """

    length = int(dataset.shape[1])
    block_size = _resolve_candidate_block_size(dataset, block_size)
    seeds: List[np.ndarray] = []
    supports: List[np.ndarray] = []
    for start in range(0, length, block_size):
        end = min(length, start + block_size)
        block = dataset[strand_index, start:end, :]
        seed, support = _summarize_candidate_bins(
            block,
            seed_max_threshold=seed_max_threshold,
            seed_mean_threshold=seed_mean_threshold,
            support_genic_threshold=support_genic_threshold
        )
        seeds.append(seed)
        supports.append(support)
    if not seeds:
        return np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.bool_)
    return np.concatenate(seeds), np.concatenate(supports)


def _adaptive_left_region_boundary(
    support: np.ndarray,
    raw_start: int,
    min_start: int,
    hard_start: int,
    background_barrier: int,
) -> int:
    """Find a safe left boundary using the forward-coordinate support bins."""

    cursor = raw_start
    background_run = 0
    while cursor > hard_start:
        bin_index = (cursor - 1) // CANDIDATE_BIN_SIZE
        segment_start = max(hard_start, bin_index * CANDIDATE_BIN_SIZE)
        segment_bp = cursor - segment_start
        if not support[bin_index]:
            previous_run = background_run
            background_run += segment_bp
            if previous_run >= background_barrier:
                barrier_boundary = cursor
            else:
                barrier_boundary = cursor - (
                    background_barrier - previous_run
                )
            eligible_boundary = min(min_start, barrier_boundary)
            if eligible_boundary >= segment_start:
                return eligible_boundary
        else:
            background_run = 0
        cursor = segment_start
    return hard_start


def _adaptive_right_region_boundary(
    support: np.ndarray,
    raw_end: int,
    min_end: int,
    hard_end: int,
    background_barrier: int,
) -> int:
    """Find a safe right boundary using the forward-coordinate support bins."""

    cursor = raw_end
    background_run = 0
    while cursor < hard_end:
        bin_index = cursor // CANDIDATE_BIN_SIZE
        segment_end = min(hard_end, (bin_index + 1) * CANDIDATE_BIN_SIZE)
        segment_bp = segment_end - cursor
        if not support[bin_index]:
            previous_run = background_run
            background_run += segment_bp
            if previous_run >= background_barrier:
                barrier_boundary = cursor
            else:
                barrier_boundary = cursor + (
                    background_barrier - previous_run
                )
            eligible_boundary = max(min_end, barrier_boundary)
            if eligible_boundary <= segment_end:
                return eligible_boundary
        else:
            background_run = 0
        cursor = segment_end
    return hard_end


def _expand_and_merge_regions(
    regions: Iterable[Tuple[int, int]],
    sequence_length: int,
    support: np.ndarray,
    *,
    min_buffer: int = DEFAULT_CANDIDATE_REGION_BUFFER,
    background_barrier: int = CANDIDATE_BACKGROUND_BARRIER_BP,
    max_terminal_extension: int = CANDIDATE_MAX_TERMINAL_EXTENSION_BP,
) -> List[Tuple[int, int]]:
    """Adaptively extend raw regions to a terminal background barrier.

    A bin with ``support == False`` is the operational strong-background
    definition: it has neither a seed nor sufficient genic/coding support.
    Each terminal keeps at least ``min_buffer`` bases when the chromosome allows
    it, then stops only when the immediately preceding ``background_barrier``
    bases are continuously background.  If no such barrier is found, extension
    is capped at ``max_terminal_extension`` bases from the raw terminal.
    """

    length = int(sequence_length)
    min_buffer = int(min_buffer)
    background_barrier = int(background_barrier)
    max_terminal_extension = int(max_terminal_extension)
    if length < 0:
        raise ValueError("Candidate region sequence length cannot be negative.")
    if min_buffer < 0:
        raise ValueError("Candidate minimum buffer cannot be negative.")
    if background_barrier <= 0:
        raise ValueError("Candidate background barrier must be positive.")
    if max_terminal_extension < min_buffer:
        raise ValueError(
            "Candidate maximum terminal extension cannot be smaller than "
            "the minimum buffer."
        )

    support_bins = np.asarray(support)
    expected_bins = (length + CANDIDATE_BIN_SIZE - 1) // CANDIDATE_BIN_SIZE
    if support_bins.ndim != 1 or support_bins.shape != (expected_bins,):
        raise ValueError(
            "Candidate support must have shape ({},), got {}.".format(
                expected_bins,
                support_bins.shape,
            )
        )
    if support_bins.dtype != np.bool_:
        raise ValueError("Candidate support bins must have boolean dtype.")

    expanded: List[Tuple[int, int]] = []
    for raw_start, raw_end in regions:
        start = int(raw_start)
        end = int(raw_end)
        if start >= end:
            continue
        if start < 0 or end > length:
            raise ValueError(
                "Raw candidate region [{}, {}) is outside sequence length {}."
                .format(start, end, length)
            )

        min_start = max(0, start - min_buffer)
        hard_start = max(0, start - max_terminal_extension)
        min_end = min(length, end + min_buffer)
        hard_end = min(length, end + max_terminal_extension)
        expanded.append(
            (
                _adaptive_left_region_boundary(
                    support_bins,
                    start,
                    min_start,
                    hard_start,
                    background_barrier,
                ),
                _adaptive_right_region_boundary(
                    support_bins,
                    end,
                    min_end,
                    hard_end,
                    background_barrier,
                ),
            )
        )

    expanded.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in expanded:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _close_worker_resources() -> None:
    global _WORKER_H5, _WORKER_FASTA
    if _WORKER_H5 is not None:
        _WORKER_H5.close()
        _WORKER_H5 = None
    if _WORKER_FASTA is not None:
        _WORKER_FASTA.close()
        _WORKER_FASTA = None


def _initialize_worker(
    context: _DecoderContext,
    h5_path: str,
    genome_fasta: str,
    fasta_index_path: str,
) -> None:
    """Open process-local read-only data handles after worker creation."""

    configure_single_threaded_libraries()
    import pyfaidx

    global _WORKER_CONTEXT, _WORKER_H5, _WORKER_FASTA
    _WORKER_CONTEXT = context
    _WORKER_H5 = h5py.File(h5_path, "r")
    _WORKER_FASTA = pyfaidx.Fasta(
        genome_fasta,
        indexname=fasta_index_path,
        build_index=False,
        rebuild=False,
        one_based_attributes=False,
    )
    atexit.register(_close_worker_resources)


def _iter_candidate_scan_tasks(
    tracks: Iterable[_CandidateTrack],
) -> Iterator[_CandidateScanTask]:
    """Yield deterministic, non-overlapping shards in track/genomic order."""

    for track in tracks:
        if track.block_size <= 0 or track.block_size % CANDIDATE_BIN_SIZE:
            raise ValueError(
                f"Candidate track {track.track_index} has an invalid block size "
                f"{track.block_size}."
            )
        for forward_start in range(0, track.chrom_length, track.shard_size):
            forward_end = min(
                track.chrom_length,
                forward_start + track.shard_size,
            )
            yield _CandidateScanTask(
                track_index=track.track_index,
                chrom_id=track.chrom_id,
                group_name=track.group_name,
                strand=track.strand,
                strand_index=track.strand_index,
                chrom_length=track.chrom_length,
                forward_start=forward_start,
                forward_end=forward_end,
                block_size=track.block_size,
                thresholds=track.thresholds,
            )


def _candidate_scan_task_count(tracks: Iterable[_CandidateTrack]) -> int:
    """Return the number of shards without materializing their task objects."""

    count = 0
    for track in tracks:
        count += (
            track.chrom_length + track.shard_size - 1
        ) // track.shard_size
    return count


def _process_candidate_scan_worker(
    task: _CandidateScanTask,
) -> _CandidateScanResult:
    """Summarize whole serial blocks from a process-local read-only HDF5."""

    if _WORKER_H5 is None:
        raise RuntimeError("Candidate scan worker HDF5 was not initialized.")
    if (
        task.forward_start < 0
        or task.forward_end <= task.forward_start
        or task.forward_end > task.chrom_length
    ):
        raise ValueError(
            f"Invalid candidate shard for {task.chrom_id} {task.strand}: "
            f"[{task.forward_start}, {task.forward_end})."
        )
    if task.forward_start % CANDIDATE_BIN_SIZE:
        raise ValueError("Candidate shard start is not bin-aligned.")
    if (
        task.forward_end < task.chrom_length
        and task.forward_end % CANDIDATE_BIN_SIZE
    ):
        raise ValueError("Non-terminal candidate shard end is not bin-aligned.")
    if task.block_size <= 0 or task.block_size % CANDIDATE_BIN_SIZE:
        raise ValueError("Candidate scan block size is not bin-aligned.")

    dataset = _WORKER_H5["chromosomes"][task.group_name]["full_probabilities"]
    if dataset.shape != (
        PREDICTION_NUM_STRANDS,
        task.chrom_length,
        PREDICTION_NUM_CLASSES,
    ):
        raise ValueError(
            f"Invalid probability shape for {task.chrom_id!r}: {dataset.shape}."
        )

    seeds: List[np.ndarray] = []
    supports: List[np.ndarray] = []
    seed_max, seed_mean, support_genic = task.thresholds
    for block_start in range(
        task.forward_start,
        task.forward_end,
        task.block_size,
    ):
        block_end = min(task.forward_end, block_start + task.block_size)
        block = dataset[task.strand_index, block_start:block_end, :]
        seed, support = _summarize_candidate_bins(
            block,
            seed_max_threshold=seed_max,
            seed_mean_threshold=seed_mean,
            support_genic_threshold=support_genic,
        )
        seeds.append(seed)
        supports.append(support)

    if not seeds:
        raise RuntimeError("Candidate scan worker produced no block summaries.")
    return _CandidateScanResult(
        track_index=task.track_index,
        forward_start=task.forward_start,
        seed=np.concatenate(seeds),
        support=np.concatenate(supports),
    )


def _process_region_worker(task: _RegionTask) -> _RegionResult:
    if _WORKER_CONTEXT is None or _WORKER_H5 is None or _WORKER_FASTA is None:
        raise RuntimeError("Segmental decoder worker context was not initialized.")
    context = _WORKER_CONTEXT
    if task.region_end <= task.region_start:
        raise ValueError("Worker received an empty candidate region.")
    dataset = _WORKER_H5["chromosomes"][task.group_name]["full_probabilities"]
    predictions = dataset[
        task.strand_index, task.forward_start:task.forward_end, :
    ]
    sequence = str(
        _WORKER_FASTA[task.chrom_id][task.forward_start:task.forward_end]
    ).upper()
    if task.strand == "-":
        predictions = predictions[::-1].copy()
        sequence = _reverse_complement_sequence(sequence)
    if predictions.shape != (len(sequence), PREDICTION_NUM_CLASSES):
        raise ValueError("Worker HDF5/FASTA region lengths differ.")
    calibration = EmissionCalibration(
        class_scale=context.class_scale,
        class_bias=context.class_bias,
        gene_intercept=context.gene_intercept,
        epsilon=context.epsilon,
        version=context.calibration_version,
    )
    genes = decode_region_predictions(
        sequence=sequence,
        predictions=predictions,
        min_intron_length=context.min_intron_length,
        min_cds_length=context.min_cds_length,
        calibration=calibration,
        chrom_id=task.chrom_id,
        strand=task.strand,
        region_offset=task.region_start,
        chrom_length=task.chrom_length,
    )
    return _RegionResult(
        chrom_id=task.chrom_id,
        strand=task.strand,
        region_start=task.region_start,
        genes=tuple(genes),
    )


def _scan_candidate_tracks_serial(
    h5_path: str,
    tracks: Iterable[_CandidateTrack],
) -> Dict[
    int,
    Tuple[
        np.ndarray,
        np.ndarray,
        List[Tuple[int, int]],
        List[Tuple[int, int]],
    ],
]:
    """Run the block scanner and shared global region logic serially."""

    output = {}
    with h5py.File(h5_path, "r") as h5_file:
        chromosomes_root = h5_file["chromosomes"]
        for track in tracks:
            dataset = chromosomes_root[track.group_name]["full_probabilities"]
            if dataset.shape != (
                PREDICTION_NUM_STRANDS,
                track.chrom_length,
                PREDICTION_NUM_CLASSES,
            ):
                raise ValueError(
                    f"Invalid probability shape for {track.chrom_id!r}: "
                    f"{dataset.shape}."
                )
            seed_max, seed_mean, support_genic = track.thresholds
            seed, support = _stream_candidate_bins(
                dataset,
                track.strand_index,
                block_size=track.block_size,
                seed_max_threshold=seed_max,
                seed_mean_threshold=seed_mean,
                support_genic_threshold=support_genic,
            )
            raw_regions = _regions_from_candidate_bins(
                seed,
                support,
                track.chrom_length,
            )
            expanded_regions = _expand_and_merge_regions(
                raw_regions,
                track.chrom_length,
                support,
            )
            output[track.track_index] = (
                seed,
                support,
                raw_regions,
                expanded_regions,
            )
    return output


def _bounded_candidate_scan_results(
    executor: ProcessPoolExecutor,
    tasks: Iterable[_CandidateScanTask],
    max_in_flight: int,
) -> Iterator[Tuple[_CandidateScanTask, _CandidateScanResult]]:
    """Submit candidate shards lazily while bounding futures and IPC state."""

    task_iterator = iter(tasks)
    in_flight = {}
    for _ in range(max(1, max_in_flight)):
        try:
            task = next(task_iterator)
        except StopIteration:
            break
        in_flight[executor.submit(_process_candidate_scan_worker, task)] = task
    while in_flight:
        completed, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
        for future in completed:
            task = in_flight.pop(future)
            try:
                yield task, future.result()
            except Exception:
                logger.exception(
                    "Candidate scanning failed for %s %s shard %d-%d",
                    task.chrom_id,
                    task.strand,
                    task.forward_start,
                    task.forward_end,
                )
                raise
            try:
                next_task = next(task_iterator)
            except StopIteration:
                continue
            in_flight[executor.submit(_process_candidate_scan_worker, next_task)] = (
                next_task
            )


def _scan_candidate_tracks_parallel(
    executor: ProcessPoolExecutor,
    tracks: Iterable[_CandidateTrack],
    max_in_flight: int,
    progress_callback: Optional[Callable[[], None]] = None,
) -> Dict[
    int,
    Tuple[
        np.ndarray,
        np.ndarray,
        List[Tuple[int, int]],
        List[Tuple[int, int]],
    ],
]:
    """Parallelize bin summaries, then run shared global region operations."""

    track_list = list(tracks)
    if not track_list:
        return {}
    track_by_index = {track.track_index: track for track in track_list}
    if len(track_by_index) != len(track_list):
        raise ValueError("Candidate track indices must be unique.")

    buffers = {
        track.track_index: (
            np.empty(track.num_bins, dtype=np.bool_),
            np.empty(track.num_bins, dtype=np.bool_),
        )
        for track in track_list
    }
    seen_shards = {track.track_index: set() for track in track_list}

    for task, result in _bounded_candidate_scan_results(
        executor,
        _iter_candidate_scan_tasks(track_list),
        max_in_flight=max_in_flight,
    ):
        if (
            result.track_index != task.track_index
            or result.forward_start != task.forward_start
        ):
            raise RuntimeError("Candidate worker returned a mismatched shard key.")
        if task.track_index not in track_by_index:
            raise RuntimeError(
                f"Candidate worker returned unknown track {task.track_index}."
            )
        if task.forward_start in seen_shards[task.track_index]:
            raise RuntimeError(
                f"Candidate shard {task.track_index}:{task.forward_start} "
                "was returned more than once."
            )

        expected_bins = (
            task.forward_end
            - task.forward_start
            + CANDIDATE_BIN_SIZE
            - 1
        ) // CANDIDATE_BIN_SIZE
        if result.seed.shape != (expected_bins,) or result.support.shape != (
            expected_bins,
        ):
            raise RuntimeError(
                f"Candidate shard {task.track_index}:{task.forward_start} "
                f"returned shapes {result.seed.shape}/{result.support.shape}; "
                f"expected ({expected_bins},)."
            )
        if result.seed.dtype != np.bool_ or result.support.dtype != np.bool_:
            raise RuntimeError("Candidate worker returned non-boolean bin flags.")

        bin_start = task.forward_start // CANDIDATE_BIN_SIZE
        bin_end = bin_start + expected_bins
        track = track_by_index[task.track_index]
        if bin_end > track.num_bins:
            raise RuntimeError("Candidate shard extends beyond its track bin array.")
        seed_buffer, support_buffer = buffers[task.track_index]
        seed_buffer[bin_start:bin_end] = result.seed
        support_buffer[bin_start:bin_end] = result.support
        seen_shards[task.track_index].add(task.forward_start)
        if progress_callback is not None:
            progress_callback()

    output = {}
    for track in track_list:
        expected_shards = (
            track.chrom_length + track.shard_size - 1
        ) // track.shard_size
        if len(seen_shards[track.track_index]) != expected_shards:
            raise RuntimeError(
                f"Candidate track {track.track_index} returned "
                f"{len(seen_shards[track.track_index])} of "
                f"{expected_shards} expected shards."
            )
        seed, support = buffers[track.track_index]
        raw_regions = _regions_from_candidate_bins(
            seed,
            support,
            track.chrom_length,
        )
        expanded_regions = _expand_and_merge_regions(
            raw_regions,
            track.chrom_length,
            support,
        )
        output[track.track_index] = (
            seed,
            support,
            raw_regions,
            expanded_regions,
        )
    return output


def _bounded_region_results(
    executor: ProcessPoolExecutor,
    tasks: Iterable[_RegionTask],
    max_in_flight: int,
) -> Iterator[Tuple[_RegionTask, _RegionResult]]:
    """Submit lazily while retaining at most ``max_in_flight`` futures."""

    task_iterator = iter(tasks)
    in_flight = {}
    for _ in range(max(1, max_in_flight)):
        try:
            task = next(task_iterator)
        except StopIteration:
            break
        in_flight[executor.submit(_process_region_worker, task)] = task
    while in_flight:
        completed, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
        for future in completed:
            task = in_flight.pop(future)
            try:
                yield task, future.result()
            except Exception:
                logger.exception(
                    "Strict segmental decoding failed for %s %s region %d-%d",
                    task.chrom_id,
                    task.strand,
                    task.region_start,
                    task.region_end,
                )
                raise
            try:
                next_task = next(task_iterator)
            except StopIteration:
                continue
            in_flight[executor.submit(_process_region_worker, next_task)] = next_task


class SegmentalDecoder:
    """Chromosome-HDF5 driver for the unique strict 15-state segmental DAG."""

    def __init__(
        self,
        cache_path: str,
        genome_fasta: str,
        output_gff: str,
        chromosome_h5_path: Optional[str] = None,
        min_intron_length: int = DEFAULT_DECODER_MIN_INTRON_LENGTH,
        min_cds_length: int = DEFAULT_DECODER_MIN_CDS_LENGTH,
        num_cpu_threads: int = 1,
        calibration: EmissionCalibration = DEFAULT_EMISSION_CALIBRATION,
        min_mean_gene_log_odds: float = DEFAULT_DECODER_MIN_MEAN_GENE_LOG_ODDS,
    ) -> None:
        self.cache_path = os.path.abspath(cache_path)
        self.h5_path = os.path.abspath(
            chromosome_h5_path
            if chromosome_h5_path
            else os.path.join(cache_path, "chromosome_predictions.h5")
        )
        self.genome_fasta = os.path.abspath(genome_fasta)
        self.output_gff = os.path.abspath(output_gff)
        self.min_intron_length = int(min_intron_length)
        self.min_cds_length = int(min_cds_length)
        self.min_mean_gene_log_odds = float(min_mean_gene_log_odds)
        self.num_cpu_threads = int(num_cpu_threads)
        if self.num_cpu_threads <= 0:
            raise ValueError("num_cpu_threads must be positive.")
        configure_single_threaded_libraries()
        _validate_calibration(calibration)
        if self.min_intron_length < 1:
            raise ValueError("min_intron_length must be at least 1.")
        if self.min_cds_length < 1:
            raise ValueError("min_cds_length must be at least 1.")
        if (
            not np.isfinite(self.min_mean_gene_log_odds)
            or self.min_mean_gene_log_odds < 0.0
        ):
            raise ValueError(
                "min_mean_gene_log_odds must be finite and non-negative."
            )
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(
                "Chromosome-level HDF5 not found: {}".format(self.h5_path)
            )
        if not os.path.isfile(self.genome_fasta):
            raise FileNotFoundError("Genome FASTA not found: {}".format(self.genome_fasta))
        # Fail fast on incompatible/unfinished files without traversing every
        # scaffold group. Manifest and per-record checks are fused with the
        # dataset accesses that _process_impl() must perform anyway.
        validate_chromosome_h5_header(self.h5_path)
        self.worker_context = _DecoderContext(
            min_intron_length=self.min_intron_length,
            min_cds_length=self.min_cds_length,
            class_scale=tuple(float(v) for v in calibration.class_scale),
            class_bias=tuple(float(v) for v in calibration.class_bias),
            gene_intercept=float(calibration.gene_intercept),
            epsilon=float(calibration.epsilon),
            calibration_version=str(calibration.version),
        )

    def _filter_genes_by_confidence(
        self,
        genes: Iterable[GenePrediction],
    ) -> Tuple[GenePrediction, ...]:
        """Retain genes meeting the configured span-normalized DAG threshold."""

        return tuple(
            gene
            for gene in genes
            if gene.mean_gene_log_odds >= self.min_mean_gene_log_odds
        )

    def _write_header(self, handle) -> None:
        handle.write("##gff-version 3\n")

    def _write_gene(
        self,
        handle,
        gene: GenePrediction,
        gene_number: int,
    ) -> None:
        gene_id = f"geneid-{gene_number}"
        mrna_id = gene_id + ".t"
        handle.write(
            gff3_line(
                seqid=gene.chrom_id,
                feature_type="gene",
                start0=gene.start0,
                end0=gene.end0,
                strand=gene.strand,
                phase=".",
                attributes={
                    "ID": gene_id,
                    "Name": gene_id,
                    "biotype": "protein_coding",
                },
            )
            + "\n"
        )
        handle.write(
            gff3_line(
                seqid=gene.chrom_id,
                feature_type="mRNA",
                start0=gene.start0,
                end0=gene.end0,
                strand=gene.strand,
                phase=".",
                attributes={"ID": mrna_id, "Parent": gene_id},
            )
            + "\n"
        )

        cumulative = 0
        transcript_features = []
        for exon in gene.exons:
            phase = str((3 - (cumulative % 3)) % 3)
            transcript_features.append((exon[0], exon[1], phase))
            cumulative += exon[1] - exon[0]
        if cumulative != gene.cds_length or cumulative % 3:
            raise RuntimeError("Cannot emit a non-triplet strict CDS.")

        for feature_number, (start0, end0, phase) in enumerate(
            sorted(transcript_features, key=lambda item: item[0]), start=1
        ):
            exon_id = f"{mrna_id}.e{feature_number}"
            cds_id = f"{mrna_id}.c{feature_number}"
            handle.write(
                gff3_line(
                    seqid=gene.chrom_id,
                    feature_type="exon",
                    start0=start0,
                    end0=end0,
                    strand=gene.strand,
                    phase=".",
                    attributes={"ID": exon_id, "Parent": mrna_id},
                )
                + "\n"
            )
            handle.write(
                gff3_line(
                    seqid=gene.chrom_id,
                    feature_type="CDS",
                    start0=start0,
                    end0=end0,
                    strand=gene.strand,
                    phase=phase,
                    attributes={"ID": cds_id, "Parent": mrna_id},
                )
                + "\n"
            )

    def process(
        self,
        *,
        step_current: int = 4,
        step_total: int = 4,
    ) -> str:
        """Run decoding with a random-access FASTA and clean runtime files."""

        original_fasta = self.genome_fasta
        resolved_fasta = prepare_genome_fasta(original_fasta, self.cache_path)
        self.genome_fasta = resolved_fasta
        try:
            return self._process_impl(
                step_current=step_current,
                step_total=step_total,
            )
        finally:
            cleanup_cached_fasta_index(resolved_fasta, self.cache_path)
            cleanup_materialized_genome_fasta(
                original_fasta,
                resolved_fasta,
                self.cache_path,
            )
            self.genome_fasta = original_fasta

    def _process_impl(
        self,
        *,
        step_current: int = 4,
        step_total: int = 4,
    ) -> str:
        """Scan candidates in parallel, then reuse the pool for strict decoding.

        ``step_current`` and ``step_total`` are retained and validated for
        compatibility with existing Python callers, but progress-step prefixes
        are no longer included in user-facing logs.
        """

        try:
            step_current = int(step_current)
            step_total = int(step_total)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "step_current and step_total must be integers."
            ) from error
        if step_current < 1 or step_total < 1 or step_current > step_total:
            raise ValueError(
                "Decoder step context must satisfy "
                "1 <= step_current <= step_total; got "
                f"step_current={step_current}, step_total={step_total}."
            )

        import pyfaidx

        started = time.monotonic()
        fasta_index_path = prepare_cached_fasta_index_path(
            self.genome_fasta,
            self.cache_path,
        )
        logger.debug("Using cached FASTA index: %s", fasta_index_path)
        output_directory = os.path.dirname(self.output_gff)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        grouped: Dict[str, Dict[str, List[GenePrediction]]] = defaultdict(
            lambda: {"+": [], "-": []}
        )
        chromosome_order: List[Tuple[str, int]] = []
        candidate_tracks: List[_CandidateTrack] = []

        # Prepare immutable scan geometry while validating FASTA/HDF5 agreement.
        # Parent handles are closed before workers are created, so no h5py or
        # pyfaidx state is inherited across process boundaries.
        fasta = pyfaidx.Fasta(
            self.genome_fasta,
            indexname=fasta_index_path,
            one_based_attributes=False,
        )
        try:
            with h5py.File(self.h5_path, "r") as h5_file:
                manifest = load_chromosome_h5_manifest(h5_file)
                chromosomes_root = h5_file["chromosomes"]
                for info in manifest.values():
                    chrom_id = info.chrom_id
                    chrom_length = info.chrom_length
                    if chrom_id not in fasta:
                        raise ValueError(
                            "HDF5 record {!r} is absent from FASTA.".format(chrom_id)
                        )
                    if len(fasta[chrom_id]) != chrom_length:
                        raise ValueError("FASTA/HDF5 chromosome length mismatch.")
                    chromosome_order.append((chrom_id, chrom_length))
                    dataset = get_chromosome_probability_dataset(
                        chromosomes_root,
                        info,
                        require_complete=True,
                    )
                    block_size = _resolve_candidate_block_size(dataset)
                    sampling_plan = _build_candidate_sampling_plan(chrom_length)
                    threshold_statistics = (
                        _compute_stratified_threshold_statistics(
                            dataset,
                            sampling_plan,
                        )
                    )
                    for strand, strand_index in (("+", 0), ("-", 1)):
                        strand_statistics = threshold_statistics[strand_index]
                        candidate_tracks.append(
                            _CandidateTrack(
                                track_index=len(candidate_tracks),
                                chrom_id=chrom_id,
                                group_name=info.group_name,
                                strand=strand,
                                strand_index=strand_index,
                                chrom_length=chrom_length,
                                block_size=block_size,
                                thresholds=strand_statistics.thresholds,
                            )
                        )
        finally:
            fasta.close()

        # Candidate scanning is bounded by the shared CPU budget, the empirical
        # scaling cap, and the actual number of scan shards.
        candidate_task_count = _candidate_scan_task_count(candidate_tracks)
        candidate_workers = max(
            1,
            min(
                self.num_cpu_threads,
                CANDIDATE_SCAN_MAX_WORKERS,
                candidate_task_count,
            ),
        )
        executor: Optional[ProcessPoolExecutor] = None
        candidate_executor: Optional[ProcessPoolExecutor] = None
        tasks: List[_RegionTask] = []
        try:
            candidate_started = time.monotonic()
            if candidate_workers == 1:
                candidate_output = _scan_candidate_tracks_serial(
                    self.h5_path,
                    candidate_tracks,
                )
            elif candidate_workers == self.num_cpu_threads:
                # The candidate pool may be reusable when the number of decode
                # tasks later resolves to the same worker count.
                executor = ProcessPoolExecutor(
                    max_workers=candidate_workers,
                    initializer=_initialize_worker,
                    initargs=(
                        self.worker_context,
                        self.h5_path,
                        self.genome_fasta,
                        fasta_index_path,
                    ),
                )
                candidate_pbar = tqdm(
                    total=candidate_task_count,
                    desc="Scanning candidates",
                    unit="shard",
                    bar_format=(
                        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}, {rate_fmt}]"
                    ),
                )
                try:
                    candidate_output = _scan_candidate_tracks_parallel(
                        executor,
                        candidate_tracks,
                        max_in_flight=max(1, 2 * candidate_workers),
                        progress_callback=lambda: candidate_pbar.update(1),
                    )
                finally:
                    candidate_pbar.close()
            else:
                # Candidate workers < decoder workers: use a dedicated capped
                # pool for candidate scanning, then shut it down and create a
                # larger pool for DAG decoding.
                candidate_executor = ProcessPoolExecutor(
                    max_workers=candidate_workers,
                    initializer=_initialize_worker,
                    initargs=(
                        self.worker_context,
                        self.h5_path,
                        self.genome_fasta,
                        fasta_index_path,
                    ),
                )
                candidate_pbar = tqdm(
                    total=candidate_task_count,
                    desc="Scanning candidates",
                    unit="shard",
                    bar_format=(
                        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}, {rate_fmt}]"
                    ),
                )
                try:
                    candidate_output = _scan_candidate_tracks_parallel(
                        candidate_executor,
                        candidate_tracks,
                        max_in_flight=max(1, 2 * candidate_workers),
                        progress_callback=lambda: candidate_pbar.update(1),
                    )
                finally:
                    candidate_pbar.close()
                    candidate_executor.shutdown(wait=True)
                    candidate_executor = None

            for track in candidate_tracks:
                if track.track_index not in candidate_output:
                    raise RuntimeError(
                        f"Candidate output is missing track {track.track_index}."
                    )
                for forward_start, forward_end in candidate_output[
                    track.track_index
                ][3]:
                    tasks.append(
                        _RegionTask(
                            chrom_id=track.chrom_id,
                            group_name=track.group_name,
                            strand=track.strand,
                            strand_index=track.strand_index,
                            forward_start=forward_start,
                            forward_end=forward_end,
                            chrom_length=track.chrom_length,
                        )
                    )
            del candidate_output
            logger.info(
                "Candidate scanning completed: tracks=%d, regions=%d, elapsed=%.1fs",
                len(candidate_tracks),
                len(tasks),
                time.monotonic() - candidate_started,
            )

            # Longest processing time first improves worker utilization at the
            # tail. Python's sort is stable, preserving all equal-length ties.
            tasks.sort(key=lambda task: task.length, reverse=True)
            task_count = len(tasks)
            decoder_workers = max(
                1,
                min(self.num_cpu_threads, max(1, task_count)),
            )
            total_bp = sum(task.length for task in tasks)
            completed_bp = 0
            decoded_genes = 0
            retained_genes = 0

            if executor is not None and candidate_workers != decoder_workers:
                executor.shutdown(wait=True)
                executor = None
            if executor is None:
                executor = ProcessPoolExecutor(
                    max_workers=decoder_workers,
                    initializer=_initialize_worker,
                    initargs=(
                        self.worker_context,
                        self.h5_path,
                        self.genome_fasta,
                        fasta_index_path,
                    ),
                )

            pbar = tqdm(
                total=task_count,
                desc="Decoding regions",
                unit="region",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
            )
            try:
                for task, result in _bounded_region_results(
                    executor,
                    iter(tasks),
                    max_in_flight=max(1, 2 * decoder_workers),
                ):
                    accepted_genes = self._filter_genes_by_confidence(
                        result.genes
                    )
                    grouped[result.chrom_id][result.strand].extend(accepted_genes)
                    completed_bp += task.length
                    decoded_genes += len(result.genes)
                    retained_genes += len(accepted_genes)

                    pbar.update(1)
                    elapsed = max(time.monotonic() - started, 1.0e-9)
                    throughput = completed_bp / elapsed
                    pbar.set_postfix_str(
                        f"genes={retained_genes}, bp={completed_bp}/{total_bp}, "
                        f"{throughput:.0f} bp/s"
                    )
            finally:
                pbar.close()
        finally:
            if candidate_executor is not None:
                candidate_executor.shutdown(wait=True)
            if executor is not None:
                executor.shutdown(wait=True)

        filtered_genes = decoded_genes - retained_genes
        logger.info(
            "Gene confidence filtering: threshold=%.3f, decoded=%d, "
            "retained=%d, filtered=%d",
            self.min_mean_gene_log_odds,
            decoded_genes,
            retained_genes,
            filtered_genes,
        )

        total_genes = 0
        with open(self.output_gff, "w", encoding="utf-8") as handle:
            self._write_header(handle)
            for chrom_id, chrom_length in chromosome_order:
                handle.write("##sequence-region {} 1 {}\n".format(chrom_id, chrom_length))
                for strand in ("+", "-"):
                    genes = grouped[chrom_id][strand]
                    genes.sort(
                        key=lambda gene: (
                            gene.start0 if strand == "+" else -gene.end0,
                            gene.end0,
                        )
                    )
                    for gene in genes:
                        if total_genes:
                            handle.write("###\n")
                        self._write_gene(handle, gene, total_genes + 1)
                        total_genes += 1

        logger.info(
            "Gene decoding completed: genes=%d, elapsed=%.1fs",
            total_genes,
            time.monotonic() - started,
        )
        return self.output_gff
