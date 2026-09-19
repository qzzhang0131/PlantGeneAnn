"""Pure strict 15-state segmental SMM decoding algorithms.

This module contains no chromosome HDF5 traversal, FASTA I/O, process-pool
orchestration, or GFF3 serialization. It decodes one transcript-oriented
candidate region from an in-memory sequence and probability matrix.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised by import-isolation tests.
    njit = None

NUMBA_AVAILABLE = njit is not None

from .configuration import (
    DEFAULT_DECODER_MIN_CDS_LENGTH,
    DEFAULT_DECODER_MIN_INTRON_LENGTH,
)
from .constants import (
    ALLOWED_LABEL_TRANSITIONS,
    ALLOWED_SPLICE_PAIRS,
    LABEL_ACCEPTOR_0,
    LABEL_ACCEPTOR_1,
    LABEL_ACCEPTOR_2,
    LABEL_BACKGROUND,
    LABEL_CDS_FRAME_0,
    LABEL_CDS_FRAME_1,
    LABEL_CDS_FRAME_2,
    LABEL_DONOR_0,
    LABEL_DONOR_1,
    LABEL_DONOR_2,
    LABEL_INTRON_0,
    LABEL_INTRON_1,
    LABEL_INTRON_2,
    LABEL_START,
    LABEL_STOP,
    PREDICTION_NUM_CLASSES,
)

DECODER_CONTRACT_VERSION = "segmental_15state_strict_v1"
DECODER_CALIBRATION_VERSION = "label_logodds_identity_v1"
DEFAULT_EMISSION_EPSILON = 1.0e-7
DEFAULT_CLASS_SCALE = (1.0,) * PREDICTION_NUM_CLASSES
DEFAULT_CLASS_BIAS = (0.0,) * PREDICTION_NUM_CLASSES
DEFAULT_GENE_INTERCEPT = 0.0

START_CODON = "ATG"
STOP_CODONS = frozenset(("TAA", "TAG", "TGA"))
SPLICE_PAIRS = frozenset(ALLOWED_SPLICE_PAIRS)
DONOR_MOTIFS = ("GT", "GC", "AT")

_DONOR_BY_PHASE = (LABEL_DONOR_0, LABEL_DONOR_1, LABEL_DONOR_2)
_ACCEPTOR_BY_INTRON_PHASE = (
    LABEL_ACCEPTOR_1,
    LABEL_ACCEPTOR_2,
    LABEL_ACCEPTOR_0,
)
_INTRON_BY_PHASE = (LABEL_INTRON_0, LABEL_INTRON_1, LABEL_INTRON_2)
_DONOR_PREDECESSOR = {
    LABEL_DONOR_0: LABEL_CDS_FRAME_2,
    LABEL_DONOR_1: LABEL_CDS_FRAME_0,
    LABEL_DONOR_2: LABEL_CDS_FRAME_1,
}


@dataclass(frozen=True)
class EmissionCalibration:
    """Immutable calibration applied to 15 label-vs-background log odds."""

    class_scale: Tuple[float, ...] = DEFAULT_CLASS_SCALE
    class_bias: Tuple[float, ...] = DEFAULT_CLASS_BIAS
    gene_intercept: float = DEFAULT_GENE_INTERCEPT
    epsilon: float = DEFAULT_EMISSION_EPSILON
    version: str = DECODER_CALIBRATION_VERSION


DEFAULT_EMISSION_CALIBRATION = EmissionCalibration()


@dataclass(frozen=True)
class GenePrediction:
    """One complete strict ORF, with exons stored in transcript order.

    Coordinates are genomic, 0-based, and half-open.  Consequently negative-
    strand exon intervals occur in descending genomic order in ``exons``.
    ``score`` is an additive calibrated DAG score, not a posterior probability.
    """

    chrom_id: str
    strand: str
    exons: Tuple[Tuple[int, int], ...]
    score: float
    contract_version: str = DECODER_CONTRACT_VERSION
    calibration_version: str = DECODER_CALIBRATION_VERSION

    @property
    def start0(self) -> int:
        return min(exon[0] for exon in self.exons)

    @property
    def end0(self) -> int:
        return max(exon[1] for exon in self.exons)

    @property
    def span_length(self) -> int:
        """Return the complete genomic gene span, including introns."""

        span = self.end0 - self.start0
        if span <= 0:
            raise ValueError("GenePrediction must have a positive genomic span.")
        return span

    @property
    def mean_gene_log_odds(self) -> float:
        """Return the additive DAG score normalized by complete gene span."""

        return float(self.score) / self.span_length

    @property
    def cds_length(self) -> int:
        return sum(end - start for start, end in self.exons)


@dataclass(frozen=True)
class _OutputNode:
    previous: Optional["_OutputNode"]
    gene: GenePrediction


@dataclass(frozen=True)
class _GeneTrace:
    completed_exons: Tuple[Tuple[int, int], ...]
    current_exon_start: int
    output: Optional[_OutputNode]
    base_score: float


@dataclass(frozen=True)
class _DonorCandidate:
    adjusted_score: float
    donor_position: int
    trace: _GeneTrace
    coding_count: int
    phase: int
    motif: str
    split_context: str

def _validate_contract_graph() -> None:
    """Fail at import/use time if the shared schema no longer matches this DAG."""

    required_edges = (
        (LABEL_BACKGROUND, LABEL_START),
        (LABEL_START, LABEL_CDS_FRAME_1),
        (LABEL_CDS_FRAME_0, LABEL_CDS_FRAME_1),
        (LABEL_CDS_FRAME_1, LABEL_CDS_FRAME_2),
        (LABEL_CDS_FRAME_2, LABEL_CDS_FRAME_0),
        (LABEL_CDS_FRAME_0, LABEL_DONOR_1),
        (LABEL_CDS_FRAME_1, LABEL_DONOR_2),
        (LABEL_CDS_FRAME_2, LABEL_DONOR_0),
        (LABEL_DONOR_0, LABEL_INTRON_0),
        (LABEL_DONOR_1, LABEL_INTRON_1),
        (LABEL_DONOR_2, LABEL_INTRON_2),
        (LABEL_INTRON_0, LABEL_ACCEPTOR_1),
        (LABEL_INTRON_1, LABEL_ACCEPTOR_2),
        (LABEL_INTRON_2, LABEL_ACCEPTOR_0),
        (LABEL_ACCEPTOR_0, LABEL_CDS_FRAME_1),
        (LABEL_ACCEPTOR_1, LABEL_CDS_FRAME_2),
        (LABEL_ACCEPTOR_2, LABEL_CDS_FRAME_0),
        (LABEL_CDS_FRAME_1, LABEL_STOP),
        (LABEL_STOP, LABEL_BACKGROUND),
    )
    for source, target in required_edges:
        if target not in ALLOWED_LABEL_TRANSITIONS.get(source, ()):
            raise RuntimeError(
                "The strict segmental decoder is incompatible with "
                "constants.ALLOWED_LABEL_TRANSITIONS: missing edge "
                "{} -> {}.".format(source, target)
            )


_validate_contract_graph()


def _validate_calibration(calibration: EmissionCalibration) -> None:
    if len(calibration.class_scale) != PREDICTION_NUM_CLASSES:
        raise ValueError("class_scale must contain exactly 15 values.")
    if len(calibration.class_bias) != PREDICTION_NUM_CLASSES:
        raise ValueError("class_bias must contain exactly 15 values.")
    values = np.asarray(
        calibration.class_scale + calibration.class_bias, dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Calibration scale and bias values must be finite.")
    if np.any(np.asarray(calibration.class_scale, dtype=np.float64) < 0.0):
        raise ValueError("Calibration class scales must be non-negative.")
    if not np.isfinite(calibration.gene_intercept):
        raise ValueError("Calibration gene_intercept must be finite.")
    if not np.isfinite(calibration.epsilon) or calibration.epsilon <= 0.0:
        raise ValueError("Calibration epsilon must be finite and positive.")
    if not calibration.version:
        raise ValueError("Calibration version must be non-empty.")


def build_label_log_odds(
    predictions: np.ndarray,
    calibration: EmissionCalibration = DEFAULT_EMISSION_CALIBRATION,
) -> np.ndarray:
    """Build float16-safe calibrated 15-way label-vs-background emissions."""

    _validate_calibration(calibration)
    probabilities = np.asarray(predictions)
    if probabilities.ndim != 2 or probabilities.shape[1] != PREDICTION_NUM_CLASSES:
        raise ValueError(
            "Strict decoder predictions must have shape (length, {}), got {}."
            .format(PREDICTION_NUM_CLASSES, probabilities.shape)
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predictions contain non-finite values.")
    if np.any(probabilities < 0.0):
        raise ValueError("Predictions contain negative probabilities.")

    safe = np.maximum(
        probabilities.astype(np.float64, copy=False), float(calibration.epsilon)
    )
    background_log = np.log(safe[:, LABEL_BACKGROUND])[:, None]
    log_odds = np.log(safe) - background_log
    log_odds *= np.asarray(calibration.class_scale, dtype=np.float64)[None, :]
    log_odds += np.asarray(calibration.class_bias, dtype=np.float64)[None, :]
    return log_odds


def _context_from_donor(sequence: str, donor_position: int, phase: int) -> str:
    if phase == 0:
        return sequence[donor_position]
    if phase == 1:
        return sequence[donor_position - 1:donor_position + 1]
    return ""


def _map_exons_to_genome(
    local_exons: Sequence[Tuple[int, int]],
    strand: str,
    region_offset: int,
    chrom_length: int,
) -> Tuple[Tuple[int, int], ...]:
    if strand == "+":
        return tuple(
            (region_offset + start, region_offset + end)
            for start, end in local_exons
        )
    return tuple(
        (
            chrom_length - (region_offset + end),
            chrom_length - (region_offset + start),
        )
        for start, end in local_exons
    )


def _update_state(
    states: Dict[Tuple[int, int, str], Tuple[float, _GeneTrace]],
    key: Tuple[int, int, str],
    score: float,
    trace: _GeneTrace,
) -> None:
    previous = states.get(key)
    if previous is None or score > previous[0]:
        states[key] = (score, trace)


def _advance_count(count: int, minimum: int) -> int:
    return min(minimum, count + 1)


def _collect_output(node: Optional[_OutputNode]) -> List[GenePrediction]:
    genes: List[GenePrediction] = []
    while node is not None:
        genes.append(node.gene)
        node = node.previous
    genes.reverse()
    return genes


def decode_region_predictions_reference(
    sequence: str,
    predictions: np.ndarray,
    min_intron_length: int = DEFAULT_DECODER_MIN_INTRON_LENGTH,
    min_cds_length: int = DEFAULT_DECODER_MIN_CDS_LENGTH,
    calibration: EmissionCalibration = DEFAULT_EMISSION_CALIBRATION,
    chrom_id: str = "region",
    strand: str = "+",
    region_offset: int = 0,
    chrom_length: Optional[int] = None,
) -> List[GenePrediction]:
    """Decode one transcript-oriented candidate region with one strict DAG DP.

    The null path and every OUTSIDE base score exactly zero.  A path may contain
    zero, one, or multiple complete, non-overlapping genes.  A single OUTSIDE
    base is consumed after STOP before a subsequent START, matching the shared
    label graph's STOP -> BACKGROUND -> START transitions.
    """

    if strand not in ("+", "-"):
        raise ValueError("strand must be '+' or '-'.")
    if min_intron_length < 1:
        raise ValueError("min_intron_length must be at least 1.")
    if min_cds_length < 1:
        raise ValueError("min_cds_length must be at least 1.")
    sequence = str(sequence).upper()
    if len(sequence) != int(np.asarray(predictions).shape[0]):
        raise ValueError("Sequence and prediction lengths differ.")
    if chrom_length is None:
        chrom_length = region_offset + len(sequence)
    if region_offset < 0 or chrom_length < region_offset + len(sequence):
        raise ValueError("Region coordinates fall outside the chromosome.")
    if not sequence:
        return []

    emissions = build_label_log_odds(predictions, calibration)
    length = len(sequence)
    minimum = int(min_cds_length)

    intron_prefix = np.zeros((3, length + 1), dtype=np.float64)
    for phase, label in enumerate(_INTRON_BY_PHASE):
        intron_prefix[phase, 1:] = np.cumsum(emissions[:, label], dtype=np.float64)

    # Exact-label coding states at the preceding position.  The key includes a
    # capped CDS length and phase-0 split-codon prefix where required.
    previous_states: Dict[Tuple[int, int, str], Tuple[float, _GeneTrace]] = {}

    # Segment-edge sweep.  Donors wait only until their minimum-length boundary,
    # then compete in running-best buckets by phase, motif, split-codon context,
    # and capped CDS length.  Thus no donor x acceptor enumeration is performed.
    ring_size = min_intron_length + 1
    pending_donors: List[List[_DonorCandidate]] = [
        [] for _ in range(ring_size)
    ]
    eligible: Dict[
        Tuple[int, str, str], Dict[int, _DonorCandidate]
    ] = defaultdict(dict)

    ready_score = 0.0
    ready_output: Optional[_OutputNode] = None
    blocked_score = -np.inf
    blocked_output: Optional[_OutputNode] = None

    for position in range(length):
        # A donor at d becomes eligible at a=d+min_intron_length+1 because the
        # intron is [d+1,a), whose length is a-d-1.
        slot = position % ring_size
        activating = pending_donors[slot]
        pending_donors[slot] = []
        for candidate in activating:
            group_key = (candidate.phase, candidate.motif, candidate.split_context)
            old = eligible[group_key].get(candidate.coding_count)
            if old is None or candidate.adjusted_score > old.adjusted_score:
                eligible[group_key][candidate.coding_count] = candidate

        current_states: Dict[
            Tuple[int, int, str], Tuple[float, _GeneTrace]
        ] = {}

        # Ordinary one-base label edges from the exact preceding label.
        for (previous_label, count, pending_context), (score, trace) in previous_states.items():
            if previous_label in (
                LABEL_START,
                LABEL_CDS_FRAME_0,
                LABEL_ACCEPTOR_0,
            ):
                next_label = LABEL_CDS_FRAME_1
            elif previous_label in (LABEL_CDS_FRAME_1, LABEL_ACCEPTOR_1):
                next_label = LABEL_CDS_FRAME_2
            elif previous_label in (LABEL_CDS_FRAME_2, LABEL_ACCEPTOR_2):
                next_label = LABEL_CDS_FRAME_0
            else:
                continue
            if next_label not in ALLOWED_LABEL_TRANSITIONS[previous_label]:
                continue

            next_context = ""
            if next_label == LABEL_CDS_FRAME_2:
                if previous_label == LABEL_ACCEPTOR_1:
                    if len(pending_context) != 1:
                        raise RuntimeError("Missing phase-0 split-codon context.")
                    codon = pending_context + sequence[position - 1:position + 1]
                else:
                    codon = sequence[position - 2:position + 1]
                if codon in STOP_CODONS:
                    continue
            next_count = _advance_count(count, minimum)
            _update_state(
                current_states,
                (next_label, next_count, next_context),
                score + emissions[position, next_label],
                trace,
            )

        # A complete gene can begin only at the first base of an exact ATG.
        if sequence[position:position + 3] == START_CODON:
            start_trace = _GeneTrace(
                completed_exons=(),
                current_exon_start=position,
                output=ready_output,
                base_score=ready_score,
            )
            _update_state(
                current_states,
                (LABEL_START, min(minimum, 1), ""),
                ready_score
                + float(calibration.gene_intercept)
                + emissions[position, LABEL_START],
                start_trace,
            )

        # Segment edges ending at an acceptor label at this position.
        acceptor_motif = sequence[position - 2:position] if position >= 2 else ""
        compatible_donors: Tuple[str, ...]
        if acceptor_motif == "AG":
            compatible_donors = ("GT", "GC")
        elif acceptor_motif == "AC":
            compatible_donors = ("AT",)
        else:
            compatible_donors = ()

        if compatible_donors:
            for phase in range(3):
                acceptor_label = _ACCEPTOR_BY_INTRON_PHASE[phase]
                for (group_phase, donor_motif, split_context), by_count in eligible.items():
                    if group_phase != phase or donor_motif not in compatible_donors:
                        continue
                    if (donor_motif, acceptor_motif) not in SPLICE_PAIRS:
                        continue
                    for count, candidate in by_count.items():
                        if phase == 1:
                            if len(split_context) != 2:
                                raise RuntimeError("Missing phase-1 split-codon context.")
                            if split_context + sequence[position] in STOP_CODONS:
                                continue
                        pending_context = split_context if phase == 0 else ""
                        next_count = _advance_count(count, minimum)
                        score = (
                            candidate.adjusted_score
                            + intron_prefix[phase, position]
                            + emissions[position, acceptor_label]
                        )
                        trace = _GeneTrace(
                            completed_exons=(
                                candidate.trace.completed_exons
                                + ((candidate.trace.current_exon_start,
                                    candidate.donor_position + 1),)
                            ),
                            current_exon_start=position,
                            output=candidate.trace.output,
                            base_score=candidate.trace.base_score,
                        )
                        _update_state(
                            current_states,
                            (acceptor_label, next_count, pending_context),
                            score,
                            trace,
                        )

        # Exact STOP is a terminal segment edge back to OUTSIDE.  It is allowed
        # only from the schema's frame-1 label and only after the hard CDS count.
        best_stop_score = -np.inf
        best_stop_output: Optional[_OutputNode] = None
        if sequence[position - 2:position + 1] in STOP_CODONS:
            for (previous_label, count, pending_context), (score, trace) in previous_states.items():
                if previous_label != LABEL_CDS_FRAME_1 or pending_context:
                    continue
                if LABEL_STOP not in ALLOWED_LABEL_TRANSITIONS[previous_label]:
                    continue
                final_count = _advance_count(count, minimum)
                if final_count < minimum:
                    continue
                final_score = score + emissions[position, LABEL_STOP]
                local_exons = trace.completed_exons + (
                    (trace.current_exon_start, position + 1),
                )
                mapped_exons = _map_exons_to_genome(
                    local_exons, strand, region_offset, int(chrom_length)
                )
                gene = GenePrediction(
                    chrom_id=chrom_id,
                    strand=strand,
                    exons=mapped_exons,
                    score=final_score - trace.base_score,
                    calibration_version=calibration.version,
                )
                if gene.cds_length < minimum or gene.cds_length % 3 != 0:
                    raise RuntimeError("Internal strict-DAG CDS length invariant failed.")
                if final_score > best_stop_score:
                    best_stop_score = final_score
                    best_stop_output = _OutputNode(trace.output, gene)

        # Generate donor-labelled coding endpoints from exact frame labels.
        generated_donors: List[_DonorCandidate] = []
        donor_motif = sequence[position + 1:position + 3]
        if donor_motif in DONOR_MOTIFS:
            for phase, donor_label in enumerate(_DONOR_BY_PHASE):
                predecessor = _DONOR_PREDECESSOR[donor_label]
                for (previous_label, count, pending_context), (score, trace) in previous_states.items():
                    if previous_label != predecessor or pending_context:
                        continue
                    if donor_label not in ALLOWED_LABEL_TRANSITIONS[previous_label]:
                        continue
                    if phase == 2 and sequence[position - 2:position + 1] in STOP_CODONS:
                        continue
                    next_count = _advance_count(count, minimum)
                    raw_score = score + emissions[position, donor_label]
                    split_context = _context_from_donor(sequence, position, phase)
                    generated_donors.append(
                        _DonorCandidate(
                            adjusted_score=(
                                raw_score - intron_prefix[phase, position + 1]
                            ),
                            donor_position=position,
                            trace=trace,
                            coding_count=next_count,
                            phase=phase,
                            motif=donor_motif,
                            split_context=split_context,
                        )
                    )
        pending_donors[slot] = generated_donors

        # STOP must consume one zero-score BACKGROUND/OUTSIDE base before its
        # path is eligible to START another gene.  Terminal STOP remains valid.
        next_ready_score = ready_score
        next_ready_output = ready_output
        if blocked_score > next_ready_score:
            next_ready_score = blocked_score
            next_ready_output = blocked_output
        ready_score = next_ready_score
        ready_output = next_ready_output
        blocked_score = best_stop_score
        blocked_output = best_stop_output
        previous_states = current_states

    if blocked_score > ready_score:
        return _collect_output(blocked_output)
    return _collect_output(ready_output)


def _encode_sequence(sequence: str) -> np.ndarray:
    """Encode DNA as A/C/G/T/N = 0/1/2/3/4 for the numerical kernel."""

    encoded = np.full(len(sequence), 4, dtype=np.int8)
    raw = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    encoded[raw == ord("A")] = 0
    encoded[raw == ord("C")] = 1
    encoded[raw == ord("G")] = 2
    encoded[raw == ord("T")] = 3
    return encoded


def _is_stop_numeric(sequence: np.ndarray, first: int, second: int, third: int) -> bool:
    return (
        sequence[first] == 3
        and (
            (sequence[second] == 0 and sequence[third] in (0, 2))
            or (sequence[second] == 2 and sequence[third] == 0)
        )
    )


if NUMBA_AVAILABLE:
    _is_stop_numeric = njit(inline="always")(_is_stop_numeric)


def _decode_numeric_dp_impl(
    sequence: np.ndarray,
    emissions: np.ndarray,
    intron_prefix: np.ndarray,
    min_intron_length: int,
    minimum: int,
    gene_intercept: float,
):
    """Array-only strict DP.  This function is compiled with Numba when present."""

    length = sequence.shape[0]
    contexts = 6
    counts = minimum + 1
    state_size = PREDICTION_NUM_CLASSES * counts * contexts
    negative_infinity = -np.inf

    previous_score = np.full(state_size, negative_infinity, dtype=np.float64)
    current_score = np.full(state_size, negative_infinity, dtype=np.float64)
    previous_trace = np.full(state_size, -1, dtype=np.int64)
    current_trace = np.full(state_size, -1, dtype=np.int64)
    previous_active = np.empty(state_size, dtype=np.int64)
    current_active = np.empty(state_size, dtype=np.int64)
    previous_count = 0

    # Trace nodes hold only integer links/coordinates plus the score at START.
    # Optimization 3: Better initial capacity estimation to reduce reallocation
    # Estimate based on typical gene density (~1 exon per 100bp in coding regions)
    estimated_traces = max(1024, length // 100 + 1000)
    trace_capacity = estimated_traces
    trace_parent = np.empty(trace_capacity, dtype=np.int64)
    trace_donor_end = np.empty(trace_capacity, dtype=np.int64)
    trace_exon_start = np.empty(trace_capacity, dtype=np.int64)
    trace_output = np.empty(trace_capacity, dtype=np.int64)
    trace_base_score = np.empty(trace_capacity, dtype=np.float64)
    trace_count = 0

    # Better output capacity estimation
    output_capacity = max(128, length // 1000 + 100)
    output_previous = np.empty(output_capacity, dtype=np.int64)
    output_trace = np.empty(output_capacity, dtype=np.int64)
    output_stop_end = np.empty(output_capacity, dtype=np.int64)
    output_gene_score = np.empty(output_capacity, dtype=np.float64)
    output_count = 0

    ring_size = min_intron_length + 1
    ring_score = np.full(
        (ring_size, 3, counts), negative_infinity, dtype=np.float64
    )
    ring_trace = np.full((ring_size, 3, counts), -1, dtype=np.int64)
    ring_position = np.full((ring_size, 3, counts), -1, dtype=np.int64)
    ring_order = np.empty((ring_size, 3, counts), dtype=np.int64)
    ring_order_count = np.zeros((ring_size, 3), dtype=np.int64)

    # group = phase x donor-motif x split-context.  Explicit insertion-order
    # arrays reproduce dict iteration and therefore strict > tie behaviour.
    group_total = 3 * 3 * 25
    eligible_score = np.full(
        (group_total, counts), negative_infinity, dtype=np.float64
    )
    eligible_trace = np.full((group_total, counts), -1, dtype=np.int64)
    eligible_position = np.full((group_total, counts), -1, dtype=np.int64)
    group_seen = np.zeros(group_total, dtype=np.uint8)
    group_order = np.empty(group_total, dtype=np.int64)
    group_order_count = 0
    eligible_count_seen = np.zeros((group_total, counts), dtype=np.uint8)
    eligible_count_order = np.empty((group_total, counts), dtype=np.int64)
    eligible_count_order_count = np.zeros(group_total, dtype=np.int64)

    ready_score = 0.0
    ready_output = -1
    blocked_score = negative_infinity
    blocked_output = -1

    for position in range(length):
        slot = position % ring_size

        # Donors generated exactly ring_size positions earlier become eligible.
        for phase in range(3):
            pending_n = ring_order_count[slot, phase]
            for pending_index in range(pending_n):
                coding_count = ring_order[slot, phase, pending_index]
                candidate_trace = ring_trace[slot, phase, coding_count]
                if candidate_trace < 0:
                    continue
                donor_position = ring_position[slot, phase, coding_count]
                first = sequence[donor_position + 1]
                second = sequence[donor_position + 2]
                if first == 2 and second == 3:
                    motif = 0  # GT
                elif first == 2 and second == 1:
                    motif = 1  # GC
                else:
                    motif = 2  # AT
                if phase == 0:
                    split_context = int(sequence[donor_position])
                elif phase == 1:
                    split_context = (
                        int(sequence[donor_position - 1]) * 5
                        + int(sequence[donor_position])
                    )
                else:
                    split_context = 0
                group = (phase * 3 + motif) * 25 + split_context
                if group_seen[group] == 0:
                    group_seen[group] = 1
                    group_order[group_order_count] = group
                    group_order_count += 1
                if eligible_count_seen[group, coding_count] == 0:
                    eligible_count_seen[group, coding_count] = 1
                    order_n = eligible_count_order_count[group]
                    eligible_count_order[group, order_n] = coding_count
                    eligible_count_order_count[group] = order_n + 1
                adjusted = ring_score[slot, phase, coding_count]
                if adjusted > eligible_score[group, coding_count]:
                    eligible_score[group, coding_count] = adjusted
                    eligible_trace[group, coding_count] = candidate_trace
                    eligible_position[group, coding_count] = donor_position
                ring_trace[slot, phase, coding_count] = -1
                ring_score[slot, phase, coding_count] = negative_infinity
            ring_order_count[slot, phase] = 0

        current_count = 0

        # Ordinary exact-label one-base transitions, in prior insertion order.
        for active_index in range(previous_count):
            state = previous_active[active_index]
            context = state % contexts
            packed = state // contexts
            coding_count = packed % counts
            previous_label = packed // counts
            if (
                previous_label == LABEL_START
                or previous_label == LABEL_CDS_FRAME_0
                or previous_label == LABEL_ACCEPTOR_0
            ):
                next_label = LABEL_CDS_FRAME_1
            elif (
                previous_label == LABEL_CDS_FRAME_1
                or previous_label == LABEL_ACCEPTOR_1
            ):
                next_label = LABEL_CDS_FRAME_2
            elif (
                previous_label == LABEL_CDS_FRAME_2
                or previous_label == LABEL_ACCEPTOR_2
            ):
                next_label = LABEL_CDS_FRAME_0
            else:
                continue

            if next_label == LABEL_CDS_FRAME_2:
                if previous_label == LABEL_ACCEPTOR_1:
                    base = context - 1
                    if (
                        base == 3
                        and position >= 1
                        and (
                            (sequence[position - 1] == 0 and sequence[position] in (0, 2))
                            or (sequence[position - 1] == 2 and sequence[position] == 0)
                        )
                    ):
                        continue
                elif position >= 2 and _is_stop_numeric(
                    sequence, position - 2, position - 1, position
                ):
                    continue
            next_count = coding_count + 1
            if next_count > minimum:
                next_count = minimum
            next_state = (next_label * counts + next_count) * contexts
            score = previous_score[state] + emissions[position, next_label]
            if current_score[next_state] == negative_infinity:
                current_active[current_count] = next_state
                current_count += 1
                current_score[next_state] = score
                current_trace[next_state] = previous_trace[state]
            elif score > current_score[next_state]:
                current_score[next_state] = score
                current_trace[next_state] = previous_trace[state]

        # Exact ATG START edge from the best ready OUTSIDE path.
        if (
            position + 2 < length
            and sequence[position] == 0
            and sequence[position + 1] == 3
            and sequence[position + 2] == 2
        ):
            if trace_count == trace_capacity:
                new_capacity = trace_capacity * 2
                new_i = np.empty(new_capacity, dtype=np.int64)
                new_i[:trace_capacity] = trace_parent
                trace_parent = new_i
                new_i = np.empty(new_capacity, dtype=np.int64)
                new_i[:trace_capacity] = trace_donor_end
                trace_donor_end = new_i
                new_i = np.empty(new_capacity, dtype=np.int64)
                new_i[:trace_capacity] = trace_exon_start
                trace_exon_start = new_i
                new_i = np.empty(new_capacity, dtype=np.int64)
                new_i[:trace_capacity] = trace_output
                trace_output = new_i
                new_f = np.empty(new_capacity, dtype=np.float64)
                new_f[:trace_capacity] = trace_base_score
                trace_base_score = new_f
                trace_capacity = new_capacity
            start_trace = trace_count
            trace_count += 1
            trace_parent[start_trace] = -1
            trace_donor_end[start_trace] = -1
            trace_exon_start[start_trace] = position
            trace_output[start_trace] = ready_output
            trace_base_score[start_trace] = ready_score
            start_count = 1
            if start_count > minimum:
                start_count = minimum
            start_state = (LABEL_START * counts + start_count) * contexts
            start_score = (
                ready_score + gene_intercept + emissions[position, LABEL_START]
            )
            if current_score[start_state] == negative_infinity:
                current_active[current_count] = start_state
                current_count += 1
                current_score[start_state] = start_score
                current_trace[start_state] = start_trace
            elif start_score > current_score[start_state]:
                current_score[start_state] = start_score
                current_trace[start_state] = start_trace

        # Segment acceptor edges.  Iterate persistent group/count insertion order.
        acceptor_kind = -1
        if position >= 2:
            if sequence[position - 2] == 0 and sequence[position - 1] == 2:
                acceptor_kind = 0  # AG accepts GT/GC
            elif sequence[position - 2] == 0 and sequence[position - 1] == 1:
                acceptor_kind = 1  # AC accepts AT
        if acceptor_kind >= 0:
            for phase in range(3):
                if phase == 0:
                    acceptor_label = LABEL_ACCEPTOR_1
                elif phase == 1:
                    acceptor_label = LABEL_ACCEPTOR_2
                else:
                    acceptor_label = LABEL_ACCEPTOR_0
                for group_index in range(group_order_count):
                    group = group_order[group_index]
                    group_phase = group // 75
                    remainder = group - group_phase * 75
                    motif = remainder // 25
                    split_context = remainder - motif * 25
                    if group_phase != phase:
                        continue
                    if acceptor_kind == 0:
                        if motif == 2:
                            continue
                    elif motif != 2:
                        continue
                    count_n = eligible_count_order_count[group]
                    for count_index in range(count_n):
                        coding_count = eligible_count_order[group, count_index]
                        candidate_trace = eligible_trace[group, coding_count]
                        if candidate_trace < 0:
                            continue
                        if phase == 1:
                            first = split_context // 5
                            second = split_context - first * 5
                            if (
                                first == 3
                                and (
                                    (second == 0 and sequence[position] in (0, 2))
                                    or (second == 2 and sequence[position] == 0)
                                )
                            ):
                                continue
                        state_context = split_context + 1 if phase == 0 else 0
                        next_count = coding_count + 1
                        if next_count > minimum:
                            next_count = minimum
                        score = (
                            eligible_score[group, coding_count]
                            + intron_prefix[phase, position]
                            + emissions[position, acceptor_label]
                        )
                        next_state = (
                            (acceptor_label * counts + next_count) * contexts
                            + state_context
                        )
                        wins = current_score[next_state] == negative_infinity
                        if not wins and score > current_score[next_state]:
                            wins = True
                        if wins:
                            if trace_count == trace_capacity:
                                new_capacity = trace_capacity * 2
                                new_i = np.empty(new_capacity, dtype=np.int64)
                                new_i[:trace_capacity] = trace_parent
                                trace_parent = new_i
                                new_i = np.empty(new_capacity, dtype=np.int64)
                                new_i[:trace_capacity] = trace_donor_end
                                trace_donor_end = new_i
                                new_i = np.empty(new_capacity, dtype=np.int64)
                                new_i[:trace_capacity] = trace_exon_start
                                trace_exon_start = new_i
                                new_i = np.empty(new_capacity, dtype=np.int64)
                                new_i[:trace_capacity] = trace_output
                                trace_output = new_i
                                new_f = np.empty(new_capacity, dtype=np.float64)
                                new_f[:trace_capacity] = trace_base_score
                                trace_base_score = new_f
                                trace_capacity = new_capacity
                            new_trace = trace_count
                            trace_count += 1
                            trace_parent[new_trace] = candidate_trace
                            trace_donor_end[new_trace] = (
                                eligible_position[group, coding_count] + 1
                            )
                            trace_exon_start[new_trace] = position
                            trace_output[new_trace] = trace_output[candidate_trace]
                            trace_base_score[new_trace] = trace_base_score[candidate_trace]
                            if current_score[next_state] == negative_infinity:
                                current_active[current_count] = next_state
                                current_count += 1
                            current_score[next_state] = score
                            current_trace[next_state] = new_trace

        # Terminal STOP edge from exact frame-1 states.
        best_stop_score = negative_infinity
        best_stop_trace = -1
        if position >= 2 and _is_stop_numeric(
            sequence, position - 2, position - 1, position
        ):
            for active_index in range(previous_count):
                state = previous_active[active_index]
                context = state % contexts
                packed = state // contexts
                coding_count = packed % counts
                previous_label = packed // counts
                if previous_label != LABEL_CDS_FRAME_1 or context != 0:
                    continue
                final_count = coding_count + 1
                if final_count > minimum:
                    final_count = minimum
                if final_count < minimum:
                    continue
                final_score = previous_score[state] + emissions[position, LABEL_STOP]
                if final_score > best_stop_score:
                    best_stop_score = final_score
                    best_stop_trace = previous_trace[state]
        best_stop_output = -1
        if best_stop_trace >= 0:
            if output_count == output_capacity:
                new_capacity = output_capacity * 2
                new_i = np.empty(new_capacity, dtype=np.int64)
                new_i[:output_capacity] = output_previous
                output_previous = new_i
                new_i = np.empty(new_capacity, dtype=np.int64)
                new_i[:output_capacity] = output_trace
                output_trace = new_i
                new_i = np.empty(new_capacity, dtype=np.int64)
                new_i[:output_capacity] = output_stop_end
                output_stop_end = new_i
                new_f = np.empty(new_capacity, dtype=np.float64)
                new_f[:output_capacity] = output_gene_score
                output_gene_score = new_f
                output_capacity = new_capacity
            best_stop_output = output_count
            output_count += 1
            output_previous[best_stop_output] = trace_output[best_stop_trace]
            output_trace[best_stop_output] = best_stop_trace
            output_stop_end[best_stop_output] = position + 1
            output_gene_score[best_stop_output] = (
                best_stop_score - trace_base_score[best_stop_trace]
            )

        # Generate donor endpoints into the fixed pending ring slot.
        motif_valid = False
        if position + 2 < length:
            first = sequence[position + 1]
            second = sequence[position + 2]
            motif_valid = (
                (first == 2 and (second == 3 or second == 1))
                or (first == 0 and second == 3)
            )
        if motif_valid:
            for phase in range(3):
                if phase == 0:
                    donor_label = LABEL_DONOR_0
                    predecessor = LABEL_CDS_FRAME_2
                elif phase == 1:
                    donor_label = LABEL_DONOR_1
                    predecessor = LABEL_CDS_FRAME_0
                else:
                    donor_label = LABEL_DONOR_2
                    predecessor = LABEL_CDS_FRAME_1
                for active_index in range(previous_count):
                    state = previous_active[active_index]
                    context = state % contexts
                    packed = state // contexts
                    coding_count = packed % counts
                    previous_label = packed // counts
                    if previous_label != predecessor or context != 0:
                        continue
                    if (
                        phase == 2
                        and position >= 2
                        and _is_stop_numeric(
                            sequence, position - 2, position - 1, position
                        )
                    ):
                        continue
                    next_count = coding_count + 1
                    if next_count > minimum:
                        next_count = minimum
                    raw_score = previous_score[state] + emissions[position, donor_label]
                    adjusted_score = (
                        raw_score - intron_prefix[phase, position + 1]
                    )
                    # min-1 and min both saturate to ``minimum``.  Match the
                    # reference eligible-dict semantics: preserve insertion
                    # order and replace only on strict score improvement.
                    if ring_trace[slot, phase, next_count] < 0:
                        order_n = ring_order_count[slot, phase]
                        ring_order[slot, phase, order_n] = next_count
                        ring_order_count[slot, phase] = order_n + 1
                        ring_score[slot, phase, next_count] = adjusted_score
                        ring_trace[slot, phase, next_count] = previous_trace[state]
                        ring_position[slot, phase, next_count] = position
                    elif adjusted_score > ring_score[slot, phase, next_count]:
                        ring_score[slot, phase, next_count] = adjusted_score
                        ring_trace[slot, phase, next_count] = previous_trace[state]
                        ring_position[slot, phase, next_count] = position

        next_ready_score = ready_score
        next_ready_output = ready_output
        if blocked_score > next_ready_score:
            next_ready_score = blocked_score
            next_ready_output = blocked_output
        ready_score = next_ready_score
        ready_output = next_ready_output
        blocked_score = best_stop_score
        blocked_output = best_stop_output

        # Reuse arrays without O(number-of-states) clearing per base.
        for active_index in range(previous_count):
            old_state = previous_active[active_index]
            previous_score[old_state] = negative_infinity
            previous_trace[old_state] = -1
        swap_score = previous_score
        previous_score = current_score
        current_score = swap_score
        swap_trace = previous_trace
        previous_trace = current_trace
        current_trace = swap_trace
        swap_active = previous_active
        previous_active = current_active
        current_active = swap_active
        previous_count = current_count

    final_output = ready_output
    if blocked_score > ready_score:
        final_output = blocked_output
    return (
        final_output,
        output_previous[:output_count],
        output_trace[:output_count],
        output_stop_end[:output_count],
        output_gene_score[:output_count],
        trace_parent[:trace_count],
        trace_donor_end[:trace_count],
        trace_exon_start[:trace_count],
    )


if NUMBA_AVAILABLE:
    _decode_numeric_dp_numba = njit(cache=True)(_decode_numeric_dp_impl)
else:
    _decode_numeric_dp_numba = None


def _rebuild_numeric_output(
    numeric_output,
    chrom_id: str,
    strand: str,
    region_offset: int,
    chrom_length: int,
    calibration_version: str,
) -> List[GenePrediction]:
    (
        final_output,
        output_previous,
        output_trace,
        output_stop_end,
        output_gene_score,
        trace_parent,
        trace_donor_end,
        trace_exon_start,
    ) = numeric_output
    output_ids: List[int] = []
    node = int(final_output)
    while node >= 0:
        output_ids.append(node)
        node = int(output_previous[node])
    output_ids.reverse()

    genes: List[GenePrediction] = []
    for node in output_ids:
        trace = int(output_trace[node])
        exons_reversed = [
            (int(trace_exon_start[trace]), int(output_stop_end[node]))
        ]
        while int(trace_parent[trace]) >= 0:
            parent = int(trace_parent[trace])
            exons_reversed.append(
                (int(trace_exon_start[parent]), int(trace_donor_end[trace]))
            )
            trace = parent
        local_exons = tuple(reversed(exons_reversed))
        genes.append(
            GenePrediction(
                chrom_id=chrom_id,
                strand=strand,
                exons=_map_exons_to_genome(
                    local_exons, strand, region_offset, chrom_length
                ),
                score=float(output_gene_score[node]),
                calibration_version=calibration_version,
            )
        )
    return genes


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
    """Decode through the compiled array DP, or the exact reference fallback."""

    if use_numba is False or not NUMBA_AVAILABLE:
        return decode_region_predictions_reference(
            sequence=sequence,
            predictions=predictions,
            min_intron_length=min_intron_length,
            min_cds_length=min_cds_length,
            calibration=calibration,
            chrom_id=chrom_id,
            strand=strand,
            region_offset=region_offset,
            chrom_length=chrom_length,
        )
    if strand not in ("+", "-"):
        raise ValueError("strand must be '+' or '-'.")
    if min_intron_length < 1:
        raise ValueError("min_intron_length must be at least 1.")
    if min_cds_length < 1:
        raise ValueError("min_cds_length must be at least 1.")
    sequence = str(sequence).upper()
    probabilities = np.asarray(predictions)
    if len(sequence) != int(probabilities.shape[0]):
        raise ValueError("Sequence and prediction lengths differ.")
    if chrom_length is None:
        chrom_length = region_offset + len(sequence)
    if region_offset < 0 or chrom_length < region_offset + len(sequence):
        raise ValueError("Region coordinates fall outside the chromosome.")
    if not sequence:
        return []
    emissions = build_label_log_odds(probabilities, calibration)
    intron_prefix = np.zeros((3, len(sequence) + 1), dtype=np.float64)
    for phase, label in enumerate(_INTRON_BY_PHASE):
        intron_prefix[phase, 1:] = np.cumsum(emissions[:, label], dtype=np.float64)
    numeric_output = _decode_numeric_dp_numba(
        _encode_sequence(sequence),
        emissions,
        intron_prefix,
        int(min_intron_length),
        int(min_cds_length),
        float(calibration.gene_intercept),
    )
    return _rebuild_numeric_output(
        numeric_output,
        chrom_id,
        strand,
        region_offset,
        int(chrom_length),
        calibration.version,
    )
