"""Minimal phase-aware HMM decoder for PlantGeneAnn.

This module intentionally keeps the HMM small and emission-driven. ATG starts,
transcript-oriented donor/acceptor motifs, stop codons, and readthrough events
contribute soft transition priors; non-canonical boundaries remain possible
with finite log penalties. The decoder does not use a rigid full gene grammar,
which avoids forcing sequence motifs to override strong model evidence.

The retained constraints are the core structural ones:

- intergenic vs CDS vs intron states
- internal codon-position states cycle CDS0 -> CDS1 -> CDS2 -> CDS0
- genes can only end after internal CDS2, so total CDS length is a multiple of three
- introns remember the next internal codon position and satisfy a minimum length
- opening an intron carries a configurable splice-event penalty
- donor/acceptor motifs and their pair compatibility softly score boundaries
- stop codons softly score CDS2-to-intergenic and terminal-CDS2 boundaries
- in-frame stop codons softly penalise continued coding from contiguous CDS2

Here CDS0/CDS1/CDS2 mean first/second/third codon position; they are not the
model's GFF3 phase labels. At CDS feature boundaries, these positions map to
GFF3 phases 0/2/1. The deep-learning probabilities remain the main evidence
source; the HMM only smooths them into a phase-consistent gene structure.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Mapping, Optional, Tuple

import h5py
import numba
import numpy as np
import pyfaidx

from .constants import _decode_h5_string
from .configuration import (
    DEFAULT_HMM_ACCEPTOR_MOTIF_WEIGHTS,
    DEFAULT_HMM_DONOR_MOTIF_WEIGHTS,
    DEFAULT_HMM_MIN_CDS_LENGTH,
    DEFAULT_HMM_MIN_GENE_LENGTH,
    DEFAULT_HMM_MIN_GENE_SCORE,
    DEFAULT_HMM_MIN_INTRON_LENGTH,
    DEFAULT_HMM_READTHROUGH_CODON_WEIGHTS,
    DEFAULT_HMM_READTHROUGH_PRIOR_STRENGTH,
    DEFAULT_HMM_SPLICE_MOTIF_PRIOR_STRENGTH,
    DEFAULT_HMM_SPLICE_PAIR_PRIOR_STRENGTH,
    DEFAULT_HMM_SPLICE_PAIR_WEIGHTS,
    DEFAULT_HMM_SPLICE_EVENT_PROB,
    DEFAULT_HMM_STOP_CODON_PRIOR_STRENGTH,
    DEFAULT_HMM_STOP_CODON_WEIGHTS,
)
from .gff_utils import gff3_line

logger = logging.getLogger("PlantGeneAnn.src.hmm_decoder")

CANDIDATE_BIN_SIZE = 50
CANDIDATE_SEED_MAX_CDS = 0.50
CANDIDATE_SEED_MEAN_CDS = 0.20
CANDIDATE_GROW_MEAN_GENIC = 0.05
CANDIDATE_GROW_MAX_CDS = 0.25
CANDIDATE_GAP_TOLERANCE_BP = 500
DEFAULT_CANDIDATE_REGION_BUFFER = 2000
CANDIDATE_SCAN_BLOCK_BP = 2_000_000
CANDIDATE_SCAN_MAX_WORKERS = 8
DEFAULT_START_CODON = "ATG"
DEFAULT_NON_ATG_START_PROB = 1e-6
DEFAULT_NON_ATG_START_LOG_PENALTY = float(np.log(DEFAULT_NON_ATG_START_PROB))

TRANSITION_PRIOR_NONE = 0
TRANSITION_PRIOR_GENE_START = 1
TRANSITION_PRIOR_DONOR = 2
TRANSITION_PRIOR_ACCEPTOR_FIXED = 3
TRANSITION_PRIOR_ACCEPTOR = TRANSITION_PRIOR_ACCEPTOR_FIXED
TRANSITION_PRIOR_STOP = 4
TRANSITION_PRIOR_LONG_DONOR_ROUTE = 5
TRANSITION_PRIOR_ACCEPTOR_LONG = 6
TRANSITION_PRIOR_READTHROUGH = 7
TRANSITION_PRIOR_DONOR_READTHROUGH = 8

EMISSION_CLASS_INTERGENIC = 0
EMISSION_CLASS_CDS = 1
EMISSION_CLASS_INTRON = 2
NUM_EMISSION_CLASSES = 3

DONOR_CLASS_GT = 0
DONOR_CLASS_GC = 1
DONOR_CLASS_AT = 2
DONOR_CLASS_OTHER = 3
NUM_DONOR_CLASSES = 4

ACCEPTOR_CLASS_AG = 0
ACCEPTOR_CLASS_AC = 1
ACCEPTOR_CLASS_OTHER = 2
NUM_ACCEPTOR_CLASSES = 3

NO_DONOR_CLASS = -1

DONOR_CLASS_BY_MOTIF = {
    "GT": DONOR_CLASS_GT,
    "GC": DONOR_CLASS_GC,
    "AT": DONOR_CLASS_AT,
}
DONOR_MOTIF_BY_CLASS = ("GT", "GC", "AT", "OTHER")

ACCEPTOR_CLASS_BY_MOTIF = {
    "AG": ACCEPTOR_CLASS_AG,
    "AC": ACCEPTOR_CLASS_AC,
}
ACCEPTOR_MOTIF_BY_CLASS = ("AG", "AC", "OTHER")

_DNA_COMPLEMENT_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _build_progress_milestones(total: int, percentage_step: int = 10) -> Dict[int, int]:
    """Map completed-item counts to percentage milestones.

    The mapping emits at most one record per completed item. For small inputs,
    multiple percentage thresholds can resolve to the same item; in that case
    only the highest reached threshold is retained. For example, five genomic
    records produce milestones at 20%, 40%, 60%, 80%, and 100% rather than
    duplicate messages for the same record.
    """

    if total <= 0:
        return {}
    if percentage_step <= 0 or percentage_step > 100:
        raise ValueError(
            f"percentage_step must be in [1, 100], got {percentage_step}."
        )

    percentages = list(range(percentage_step, 100, percentage_step)) + [100]
    milestones: Dict[int, int] = {}
    for percentage in percentages:
        completed_items = (total * percentage + 99) // 100
        milestones[completed_items] = percentage
    return milestones


def _reverse_complement_sequence(sequence: str) -> str:
    """Return the reverse-complement DNA sequence in transcript orientation."""

    return sequence.translate(_DNA_COMPLEMENT_TABLE)[::-1].upper()


def _build_atg_start_mask(transcript_sequence: str) -> np.ndarray:
    """Mark transcript-oriented positions where a CDS can start with ATG.

    The mask is used as a soft biological prior on the HMM start transition:
    ``intergenic -> CDS0``. Positions without an in-frame ATG candidate are
    still allowed, but receive a large fixed log penalty.
    """

    sequence = transcript_sequence.upper()
    start_mask = np.zeros(len(sequence), dtype=np.bool_)

    start_index = sequence.find(DEFAULT_START_CODON)
    while start_index != -1:
        start_mask[start_index] = True
        start_index = sequence.find(DEFAULT_START_CODON, start_index + 1)

    return start_mask


def _validate_motif_weights(
    motif_weights: Mapping[str, float],
    *,
    name: str,
) -> None:
    """Validate relative motif weights used to construct finite log priors."""

    if "OTHER" not in motif_weights:
        raise ValueError(f"{name} must define an 'OTHER' fallback weight.")

    for motif, weight in motif_weights.items():
        if not np.isfinite(weight) or not 0.0 < float(weight) <= 1.0:
            raise ValueError(
                f"{name}[{motif!r}] must be finite and in (0, 1], got {weight}."
            )


def _build_sequence_prior_scores(
    transcript_sequence: str,
    splice_prior_strength: float,
    stop_prior_strength: float,
    donor_motif_weights: Mapping[str, float] = DEFAULT_HMM_DONOR_MOTIF_WEIGHTS,
    acceptor_motif_weights: Mapping[str, float] = DEFAULT_HMM_ACCEPTOR_MOTIF_WEIGHTS,
    stop_codon_weights: Mapping[str, float] = DEFAULT_HMM_STOP_CODON_WEIGHTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build donor, acceptor, and stop-boundary log-prior arrays.

    The supplied sequence must be in transcript orientation. For a transition
    at boundary ``t``:

    - ``donor_scores[t]`` evaluates the first two intron bases ``seq[t:t+2]``.
    - ``acceptor_scores[t]`` evaluates the final two intron bases ``seq[t-2:t]``.
    - ``stop_end_scores[t]`` evaluates the final CDS codon ``seq[t-3:t]``.

    ``stop_end_scores`` has length ``L + 1`` so a CDS2 terminal state at the
    final sequence base can evaluate the boundary immediately after the region.
    """

    if not np.isfinite(splice_prior_strength) or splice_prior_strength < 0.0:
        raise ValueError(
            "splice_prior_strength must be finite and non-negative, "
            f"got {splice_prior_strength}."
        )
    if not np.isfinite(stop_prior_strength) or stop_prior_strength < 0.0:
        raise ValueError(
            "stop_prior_strength must be finite and non-negative, "
            f"got {stop_prior_strength}."
        )

    _validate_motif_weights(donor_motif_weights, name="donor_motif_weights")
    _validate_motif_weights(acceptor_motif_weights, name="acceptor_motif_weights")
    _validate_motif_weights(stop_codon_weights, name="stop_codon_weights")

    sequence = str(transcript_sequence).upper()
    seq_length = len(sequence)

    donor_other_score = splice_prior_strength * np.log(
        float(donor_motif_weights["OTHER"])
    )
    acceptor_other_score = splice_prior_strength * np.log(
        float(acceptor_motif_weights["OTHER"])
    )
    stop_other_score = stop_prior_strength * np.log(
        float(stop_codon_weights["OTHER"])
    )

    donor_scores = np.full(seq_length, donor_other_score, dtype=np.float64)
    acceptor_scores = np.full(seq_length, acceptor_other_score, dtype=np.float64)
    stop_end_scores = np.full(seq_length + 1, stop_other_score, dtype=np.float64)

    for motif, weight in donor_motif_weights.items():
        if motif == "OTHER":
            continue
        motif_score = splice_prior_strength * np.log(float(weight))
        motif_start = sequence.find(motif)
        while motif_start != -1:
            donor_scores[motif_start] = motif_score
            motif_start = sequence.find(motif, motif_start + 1)

    for motif, weight in acceptor_motif_weights.items():
        if motif == "OTHER":
            continue
        motif_score = splice_prior_strength * np.log(float(weight))
        motif_start = sequence.find(motif)
        while motif_start != -1:
            first_exon_base = motif_start + len(motif)
            if first_exon_base < seq_length:
                acceptor_scores[first_exon_base] = motif_score
            motif_start = sequence.find(motif, motif_start + 1)

    for codon, weight in stop_codon_weights.items():
        if codon == "OTHER":
            continue
        codon_score = stop_prior_strength * np.log(float(weight))
        codon_start = sequence.find(codon)
        while codon_start != -1:
            cds_end = codon_start + len(codon)
            if cds_end <= seq_length:
                stop_end_scores[cds_end] = codon_score
            codon_start = sequence.find(codon, codon_start + 1)

    return donor_scores, acceptor_scores, stop_end_scores


def _build_readthrough_prior_scores(
    transcript_sequence: str,
    prior_strength: float,
    codon_weights: Mapping[str, float] = DEFAULT_HMM_READTHROUGH_CODON_WEIGHTS,
) -> np.ndarray:
    """Build log priors for continuing CDS after a complete codon.

    ``readthrough_scores[t]`` evaluates the contiguous transcript-oriented
    codon ``seq[t-3:t]`` when the HMM continues coding across boundary ``t``.
    Standard stop codons receive a finite penalty; all other codons are
    neutral by default. The array has length ``L + 1`` so its boundary
    coordinates match ``stop_end_scores`` exactly.

    Split codons are deliberately not evaluated with this local array because
    ``seq[t-3:t]`` would include intronic rather than spliced transcript bases.
    Their transitions remain neutral unless the state topology is extended to
    retain the required pre-intron nucleotide context.
    """

    if not np.isfinite(prior_strength) or prior_strength < 0.0:
        raise ValueError(
            "readthrough prior strength must be finite and non-negative, "
            f"got {prior_strength}."
        )
    _validate_motif_weights(codon_weights, name="readthrough_codon_weights")

    sequence = str(transcript_sequence).upper()
    seq_length = len(sequence)
    fallback_score = prior_strength * np.log(float(codon_weights["OTHER"]))
    readthrough_scores = np.full(
        seq_length + 1,
        fallback_score,
        dtype=np.float64,
    )

    for codon, weight in codon_weights.items():
        if codon == "OTHER":
            continue
        codon_score = prior_strength * np.log(float(weight))
        codon_start = sequence.find(codon)
        while codon_start != -1:
            codon_end = codon_start + len(codon)
            if codon_end <= seq_length:
                readthrough_scores[codon_end] = codon_score
            codon_start = sequence.find(codon, codon_start + 1)

    return readthrough_scores


def _build_splice_motif_classes(
    transcript_sequence: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify donor and acceptor motifs at transcript-oriented boundaries."""

    sequence = str(transcript_sequence).upper()
    seq_length = len(sequence)
    donor_classes = np.full(
        seq_length,
        DONOR_CLASS_OTHER,
        dtype=np.int8,
    )
    acceptor_classes = np.full(
        seq_length,
        ACCEPTOR_CLASS_OTHER,
        dtype=np.int8,
    )

    for motif, donor_class in DONOR_CLASS_BY_MOTIF.items():
        motif_start = sequence.find(motif)
        while motif_start != -1:
            donor_classes[motif_start] = donor_class
            motif_start = sequence.find(motif, motif_start + 1)

    for motif, acceptor_class in ACCEPTOR_CLASS_BY_MOTIF.items():
        motif_start = sequence.find(motif)
        while motif_start != -1:
            first_exon_base = motif_start + len(motif)
            if first_exon_base < seq_length:
                acceptor_classes[first_exon_base] = acceptor_class
            motif_start = sequence.find(motif, motif_start + 1)

    return donor_classes, acceptor_classes


def _validate_splice_pair_weights(
    pair_weights: Mapping[object, float],
) -> None:
    """Validate recognised splice pairs and the non-standard fallback weight."""

    if "OTHER" not in pair_weights:
        raise ValueError("splice pair weights must define an 'OTHER' fallback.")

    for pair, weight in pair_weights.items():
        if not np.isfinite(weight) or not 0.0 < float(weight) <= 1.0:
            raise ValueError(
                f"splice pair weight {pair!r} must be finite and in (0, 1], "
                f"got {weight}."
            )
        if pair == "OTHER":
            continue
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                "splice pair keys must be (donor, acceptor) tuples or 'OTHER', "
                f"got {pair!r}."
            )
        donor_motif, acceptor_motif = pair
        if donor_motif not in DONOR_CLASS_BY_MOTIF:
            raise ValueError(f"Unknown donor motif in splice pair: {pair!r}.")
        if acceptor_motif not in ACCEPTOR_CLASS_BY_MOTIF:
            raise ValueError(f"Unknown acceptor motif in splice pair: {pair!r}.")


def _build_splice_pair_score_matrix(
    pair_prior_strength: float,
    pair_weights: Mapping[object, float] = DEFAULT_HMM_SPLICE_PAIR_WEIGHTS,
) -> np.ndarray:
    """Build donor-by-acceptor log scores for splice-pair compatibility."""

    if not np.isfinite(pair_prior_strength) or pair_prior_strength < 0.0:
        raise ValueError(
            "pair_prior_strength must be finite and non-negative, "
            f"got {pair_prior_strength}."
        )
    _validate_splice_pair_weights(pair_weights)

    fallback_score = pair_prior_strength * np.log(float(pair_weights["OTHER"]))
    pair_scores = np.full(
        (NUM_DONOR_CLASSES, NUM_ACCEPTOR_CLASSES),
        fallback_score,
        dtype=np.float64,
    )

    for pair, weight in pair_weights.items():
        if pair == "OTHER":
            continue
        donor_motif, acceptor_motif = pair
        donor_class = DONOR_CLASS_BY_MOTIF[donor_motif]
        acceptor_class = ACCEPTOR_CLASS_BY_MOTIF[acceptor_motif]
        pair_scores[donor_class, acceptor_class] = (
            pair_prior_strength * np.log(float(weight))
        )

    return pair_scores


def _define_states(
    min_intron_length: int,
) -> Tuple[Dict[str, int], int, List[int], List[int], List[int], List[int]]:
    """Define the minimal phase-aware state topology.

    ``CDS0``, ``CDS1``, and ``CDS2`` are internal codon-position states for
    the first, second, and third base of a codon. They must not be confused
    with the model labels for GFF3 phases 0, 1, and 2.

    Intron states are named by the internal codon position that must follow
    the intron. For example, ``intron_to_CDS1_0`` means the previous CDS base
    was at the first codon position (CDS0), so decoding resumes at the second
    codon position (CDS1), whose GFF3 phase at a new CDS boundary is 2.
    """

    if min_intron_length < 1:
        raise ValueError(f"min_intron_length must be at least 1, got {min_intron_length}.")

    # ``CDS1_split`` and ``CDS2_split`` retain whether the current codon was
    # interrupted by an intron. This allows a local genomic stop-codon prior
    # to be applied only to a truly contiguous CDS0 -> CDS1 -> CDS2 triplet.
    # Split terminal codons remain valid but receive a neutral stop prior.
    states = [
        "intergenic",
        "CDS0",
        "CDS1",
        "CDS1_split",
        "CDS2",
        "CDS2_split",
    ]
    intron_columns: List[str] = []

    for next_phase in range(3):
        for i in range(min_intron_length):
            state = f"intron_to_CDS{next_phase}_{i}"
            states.append(state)
            intron_columns.append(state)
        # Fixed-duration states have a deterministic donor coordinate at each
        # time step, so they do not need donor memory. Only the long state can
        # represent multiple possible intron starts and is split by donor class.
        for donor_motif in DONOR_MOTIF_BY_CLASS:
            long_state = (
                f"intron_to_CDS{next_phase}_long_{donor_motif}"
            )
            states.append(long_state)
            intron_columns.append(long_state)

    states_to_num = {state: i for i, state in enumerate(states)}
    phase_0_indices = [states_to_num["CDS0"]]
    phase_1_indices = [
        states_to_num["CDS1"],
        states_to_num["CDS1_split"],
    ]
    phase_2_indices = [
        states_to_num["CDS2"],
        states_to_num["CDS2_split"],
    ]
    intron_indices = [states_to_num[state] for state in intron_columns]

    return (
        states_to_num,
        len(states_to_num),
        phase_0_indices,
        phase_1_indices,
        phase_2_indices,
        intron_indices,
    )


def _build_minimal_transitions(
    states_to_num: Dict[str, int],
    min_intron_length: int,
    splice_event_log_penalty: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build sparse transitions for the minimal HMM."""

    transitions: List[Tuple[int, int, float, int, int]] = []

    def add(
        src: str,
        dst: str,
        score: float = 0.0,
        prior_type: int = TRANSITION_PRIOR_NONE,
        donor_class: int = NO_DONOR_CLASS,
    ) -> None:
        transitions.append(
            (
                states_to_num[src],
                states_to_num[dst],
                float(score),
                int(prior_type),
                int(donor_class),
            )
        )

    add("intergenic", "intergenic")
    add(
        "intergenic",
        "CDS0",
        prior_type=TRANSITION_PRIOR_GENE_START,
    )

    add("CDS0", "CDS1")
    add("CDS1", "CDS2")
    add("CDS1_split", "CDS2_split")
    add(
        "CDS2",
        "CDS0",
        prior_type=TRANSITION_PRIOR_READTHROUGH,
    )
    add("CDS2_split", "CDS0")
    add(
        "CDS2",
        "intergenic",
        prior_type=TRANSITION_PRIOR_STOP,
    )
    add("CDS2_split", "intergenic")

    for last_state, next_phase in (
        ("CDS0", 1),
        ("CDS1", 2),
        ("CDS1_split", 2),
        ("CDS2", 0),
        ("CDS2_split", 0),
    ):
        donor_prior_type = (
            TRANSITION_PRIOR_DONOR_READTHROUGH
            if last_state == "CDS2"
            else TRANSITION_PRIOR_DONOR
        )
        add(
            last_state,
            f"intron_to_CDS{next_phase}_0",
            splice_event_log_penalty,
            prior_type=donor_prior_type,
        )

    for next_phase in range(3):
        for i in range(min_intron_length - 1):
            add(f"intron_to_CDS{next_phase}_{i}", f"intron_to_CDS{next_phase}_{i + 1}")

        last_required = f"intron_to_CDS{next_phase}_{min_intron_length - 1}"
        resume_state = {
            0: "CDS0",
            1: "CDS1_split",
            2: "CDS2_split",
        }[next_phase]
        add(
            last_required,
            resume_state,
            prior_type=TRANSITION_PRIOR_ACCEPTOR_FIXED,
        )
        for donor_class, donor_motif in enumerate(DONOR_MOTIF_BY_CLASS):
            long_state = (
                f"intron_to_CDS{next_phase}_long_{donor_motif}"
            )
            add(
                last_required,
                long_state,
                prior_type=TRANSITION_PRIOR_LONG_DONOR_ROUTE,
                donor_class=donor_class,
            )
            add(long_state, long_state, donor_class=donor_class)
            add(
                long_state,
                resume_state,
                prior_type=TRANSITION_PRIOR_ACCEPTOR_LONG,
                donor_class=donor_class,
            )

    from_states = np.asarray(
        [src for src, _, _, _, _ in transitions],
        dtype=np.int32,
    )
    to_states = np.asarray(
        [dst for _, dst, _, _, _ in transitions],
        dtype=np.int32,
    )
    transition_scores = np.asarray(
        [score for _, _, score, _, _ in transitions],
        dtype=np.float64,
    )
    transition_prior_types = np.asarray(
        [prior_type for _, _, _, prior_type, _ in transitions],
        dtype=np.int8,
    )
    transition_donor_classes = np.asarray(
        [donor_class for _, _, _, _, donor_class in transitions],
        dtype=np.int8,
    )

    initial_states = np.asarray(
        [states_to_num["intergenic"], states_to_num["CDS0"]],
        dtype=np.int32,
    )
    initial_scores = np.asarray([0.0, 0.0], dtype=np.float64)
    terminal_states = np.asarray(
        [
            states_to_num["intergenic"],
            states_to_num["CDS2"],
            states_to_num["CDS2_split"],
        ],
        dtype=np.int32,
    )

    return (
        from_states,
        to_states,
        transition_scores,
        transition_prior_types,
        transition_donor_classes,
        initial_states,
        initial_scores,
        terminal_states,
    )


def _build_compact_log_emissions(predictions: np.ndarray) -> np.ndarray:
    """Build the three distinct log-emission tracks used by every HMM state.

    The previous implementation materialized one identical emission column per
    state. The state topology has only three emission classes: intergenic,
    aggregate CDS, and intron. Keeping the original float64 conversion,
    flooring, CDS summation, clipping, and logarithm order makes this compact
    representation numerically identical to the former full state matrix.
    """

    epsilon = 1e-64
    # Probabilities are stored as float16 to limit HDF5 size. Convert the
    # candidate-region working copy to float64 before applying the original
    # probability floor; otherwise 1e-64 underflows back to zero in float16.
    predictions = np.asarray(predictions, dtype=np.float64)
    predictions[predictions < epsilon] = epsilon

    cds_prob = predictions[:, 1:4].sum(axis=1, dtype=np.float64)
    cds_prob = np.clip(cds_prob, epsilon, 1.0)

    log_emissions = np.empty(
        (predictions.shape[0], NUM_EMISSION_CLASSES),
        dtype=np.float64,
    )
    log_emissions[:, EMISSION_CLASS_INTERGENIC] = np.log(predictions[:, 0])
    log_emissions[:, EMISSION_CLASS_CDS] = np.log(cds_prob)
    log_emissions[:, EMISSION_CLASS_INTRON] = np.log(predictions[:, 4])
    return log_emissions


def _build_state_emission_classes(
    num_states: int,
    intergenic_state: int,
    phase_0_columns: List[int],
    phase_1_columns: List[int],
    phase_2_columns: List[int],
    intron_columns: List[int],
) -> np.ndarray:
    """Map each HMM state to one of the three compact emission columns."""

    state_emission_classes = np.full(num_states, -1, dtype=np.int8)
    state_emission_classes[intergenic_state] = EMISSION_CLASS_INTERGENIC
    state_emission_classes[phase_0_columns] = EMISSION_CLASS_CDS
    state_emission_classes[phase_1_columns] = EMISSION_CLASS_CDS
    state_emission_classes[phase_2_columns] = EMISSION_CLASS_CDS
    state_emission_classes[intron_columns] = EMISSION_CLASS_INTRON

    unmapped_states = np.flatnonzero(state_emission_classes < 0)
    if unmapped_states.size:
        raise ValueError(
            "Every HMM state must have an emission class; unmapped state "
            f"indices: {unmapped_states.tolist()}."
        )
    return state_emission_classes


def _backpointer_dtype_for_num_states(num_states: int) -> np.dtype:
    """Return the narrowest unsigned dtype that can store every state ID."""

    if num_states < 1:
        raise ValueError(f"num_states must be positive, got {num_states}.")

    max_state_id = num_states - 1
    if max_state_id <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_state_id <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_state_id <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    raise ValueError(
        "HMM state count exceeds the uint32 backpointer capacity: "
        f"{num_states}."
    )


def _allocate_backpointers(seq_length: int, num_states: int) -> np.ndarray:
    """Allocate deterministic, range-safe backpointer storage."""

    if seq_length < 1:
        raise ValueError(f"seq_length must be positive, got {seq_length}.")
    return np.zeros(
        (seq_length, num_states),
        dtype=_backpointer_dtype_for_num_states(num_states),
    )


@numba.njit(cache=True)
def _viterbi_sparse_core(
    log_emissions: np.ndarray,
    state_emission_classes: np.ndarray,
    backpointers: np.ndarray,
    from_states: np.ndarray,
    to_states: np.ndarray,
    transition_scores: np.ndarray,
    transition_prior_types: np.ndarray,
    transition_donor_classes: np.ndarray,
    initial_states: np.ndarray,
    initial_scores: np.ndarray,
    terminal_states: np.ndarray,
    start_allowed_mask: np.ndarray,
    donor_scores: np.ndarray,
    acceptor_scores: np.ndarray,
    stop_end_scores: np.ndarray,
    readthrough_scores: np.ndarray,
    donor_classes: np.ndarray,
    acceptor_classes: np.ndarray,
    splice_pair_scores: np.ndarray,
    min_intron_length: int,
    cds0_state: int,
    cds2_state: int,
    non_atg_start_log_penalty: float,
) -> np.ndarray:
    """Memory-efficient sparse Viterbi for the minimal HMM.

    Only the previous and current DP rows are retained. Full traceback remains
    exact through range-safe unsigned backpointers supplied by the Python
    wrapper. Transition order, strict ``>`` tie-breaking, float64 arithmetic,
    and prior application order intentionally match the former full-matrix
    implementation.
    """

    seq_length = log_emissions.shape[0]
    num_states = state_emission_classes.shape[0]
    previous_dp = np.full(num_states, -np.inf, dtype=np.float64)
    current_dp = np.full(num_states, -np.inf, dtype=np.float64)

    for i in range(initial_states.shape[0]):
        state = initial_states[i]
        score = initial_scores[i]
        if state == cds0_state and not start_allowed_mask[0]:
            score += non_atg_start_log_penalty
        emission_class = state_emission_classes[state]
        previous_dp[state] = score + log_emissions[0, emission_class]
        backpointers[0, state] = state

    for t in range(1, seq_length):
        current_dp.fill(-np.inf)
        for edge_i in range(from_states.shape[0]):
            src = from_states[edge_i]
            dst = to_states[edge_i]
            transition_score = transition_scores[edge_i]
            prior_type = transition_prior_types[edge_i]
            edge_donor_class = transition_donor_classes[edge_i]
            if prior_type == TRANSITION_PRIOR_GENE_START:
                if not start_allowed_mask[t]:
                    transition_score += non_atg_start_log_penalty
            elif prior_type == TRANSITION_PRIOR_DONOR:
                transition_score += donor_scores[t]
            elif prior_type == TRANSITION_PRIOR_DONOR_READTHROUGH:
                transition_score += donor_scores[t]
                transition_score += readthrough_scores[t]
            elif prior_type == TRANSITION_PRIOR_ACCEPTOR_FIXED:
                donor_position = t - min_intron_length
                if donor_position < 0:
                    continue
                donor_class = donor_classes[donor_position]
                acceptor_class = acceptor_classes[t]
                transition_score += acceptor_scores[t]
                transition_score += splice_pair_scores[
                    donor_class,
                    acceptor_class,
                ]
            elif prior_type == TRANSITION_PRIOR_LONG_DONOR_ROUTE:
                donor_position = t - min_intron_length
                if (
                    donor_position < 0
                    or donor_classes[donor_position] != edge_donor_class
                ):
                    continue
            elif prior_type == TRANSITION_PRIOR_ACCEPTOR_LONG:
                acceptor_class = acceptor_classes[t]
                transition_score += acceptor_scores[t]
                transition_score += splice_pair_scores[
                    edge_donor_class,
                    acceptor_class,
                ]
            elif prior_type == TRANSITION_PRIOR_STOP:
                transition_score += stop_end_scores[t]
            elif prior_type == TRANSITION_PRIOR_READTHROUGH:
                transition_score += readthrough_scores[t]
            emission_class = state_emission_classes[dst]
            val = (
                previous_dp[src]
                + transition_score
                + log_emissions[t, emission_class]
            )
            if val > current_dp[dst]:
                current_dp[dst] = val
                backpointers[t, dst] = src

        previous_dp, current_dp = current_dp, previous_dp

    best_final_state = terminal_states[0]
    best_final_val = previous_dp[best_final_state]
    if best_final_state == cds2_state:
        best_final_val += stop_end_scores[seq_length]
    for i in range(1, terminal_states.shape[0]):
        state = terminal_states[i]
        val = previous_dp[state]
        if state == cds2_state:
            val += stop_end_scores[seq_length]
        if val > best_final_val:
            best_final_val = val
            best_final_state = state

    if best_final_val == -np.inf:
        for state in range(num_states):
            val = previous_dp[state]
            if val > best_final_val:
                best_final_val = val
                best_final_state = state

    best_path = np.empty(seq_length, dtype=np.int32)
    best_path[seq_length - 1] = best_final_state
    for t in range(seq_length - 2, -1, -1):
        best_path[t] = backpointers[t + 1, best_path[t + 1]]

    return best_path


def _viterbi_decode(
    predictions: np.ndarray,
    states_to_num: Dict[str, int],
    num_states: int,
    phase_0_columns: List[int],
    phase_1_columns: List[int],
    phase_2_columns: List[int],
    intron_columns: List[int],
    min_intron_length: int,
    splice_event_log_penalty: float,
    start_allowed_mask: np.ndarray,
    donor_scores: np.ndarray,
    acceptor_scores: np.ndarray,
    stop_end_scores: np.ndarray,
    readthrough_scores: np.ndarray,
    donor_classes: np.ndarray,
    acceptor_classes: np.ndarray,
    splice_pair_scores: np.ndarray,
    non_atg_start_log_penalty: float,
) -> List[int]:
    """Run minimal phase-aware Viterbi decoding on one candidate region.

    Internal codon-position states CDS0/CDS1/CDS2 share the aggregate CDS
    probability. Transitions enforce reading-frame continuity independently of
    the model's per-base GFF3 phase labels, which cycle 1 -> 3 -> 2.
    """

    if predictions.shape[0] == 0:
        return []

    if start_allowed_mask.shape[0] != predictions.shape[0]:
        raise ValueError(
            "ATG start mask length must match prediction length: "
            f"{start_allowed_mask.shape[0]} != {predictions.shape[0]}."
        )

    expected_prior_shapes = {
        "donor_scores": (predictions.shape[0],),
        "acceptor_scores": (predictions.shape[0],),
        "stop_end_scores": (predictions.shape[0] + 1,),
        "readthrough_scores": (predictions.shape[0] + 1,),
        "donor_classes": (predictions.shape[0],),
        "acceptor_classes": (predictions.shape[0],),
    }
    observed_prior_arrays = {
        "donor_scores": donor_scores,
        "acceptor_scores": acceptor_scores,
        "stop_end_scores": stop_end_scores,
        "readthrough_scores": readthrough_scores,
        "donor_classes": donor_classes,
        "acceptor_classes": acceptor_classes,
    }
    for prior_name, expected_shape in expected_prior_shapes.items():
        observed_shape = observed_prior_arrays[prior_name].shape
        if observed_shape != expected_shape:
            raise ValueError(
                f"{prior_name} must have shape {expected_shape}, "
                f"got {observed_shape}."
            )
    if splice_pair_scores.shape != (NUM_DONOR_CLASSES, NUM_ACCEPTOR_CLASSES):
        raise ValueError(
            "splice_pair_scores must have shape "
            f"({NUM_DONOR_CLASSES}, {NUM_ACCEPTOR_CLASSES}), "
            f"got {splice_pair_scores.shape}."
        )
    if np.any((donor_classes < 0) | (donor_classes >= NUM_DONOR_CLASSES)):
        raise ValueError("donor_classes contains an invalid class index.")
    if np.any(
        (acceptor_classes < 0) | (acceptor_classes >= NUM_ACCEPTOR_CLASSES)
    ):
        raise ValueError("acceptor_classes contains an invalid class index.")
    if not np.all(np.isfinite(splice_pair_scores)):
        raise ValueError("splice_pair_scores contains NaN or infinite values.")

    seq_length = predictions.shape[0]
    log_emissions = _build_compact_log_emissions(predictions)
    state_emission_classes = _build_state_emission_classes(
        num_states=num_states,
        intergenic_state=states_to_num["intergenic"],
        phase_0_columns=phase_0_columns,
        phase_1_columns=phase_1_columns,
        phase_2_columns=phase_2_columns,
        intron_columns=intron_columns,
    )
    backpointers = _allocate_backpointers(seq_length, num_states)

    (
        from_states,
        to_states,
        transition_scores,
        transition_prior_types,
        transition_donor_classes,
        initial_states,
        initial_scores,
        terminal_states,
    ) = _build_minimal_transitions(
        states_to_num=states_to_num,
        min_intron_length=min_intron_length,
        splice_event_log_penalty=splice_event_log_penalty,
    )

    best_path_array = _viterbi_sparse_core(
        log_emissions=log_emissions,
        state_emission_classes=state_emission_classes,
        backpointers=backpointers,
        from_states=from_states,
        to_states=to_states,
        transition_scores=transition_scores,
        transition_prior_types=transition_prior_types,
        transition_donor_classes=transition_donor_classes,
        initial_states=initial_states,
        initial_scores=initial_scores,
        terminal_states=terminal_states,
        start_allowed_mask=start_allowed_mask,
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        stop_end_scores=stop_end_scores,
        readthrough_scores=readthrough_scores,
        donor_classes=donor_classes,
        acceptor_classes=acceptor_classes,
        splice_pair_scores=splice_pair_scores,
        min_intron_length=min_intron_length,
        cds0_state=states_to_num["CDS0"],
        cds2_state=states_to_num["CDS2"],
        non_atg_start_log_penalty=non_atg_start_log_penalty,
    )

    return best_path_array.tolist()


def _parse_ranges(lst: List[int], targets: List[int]) -> List[Dict[str, List[Tuple[int, int]]]]:
    """Parse state sequence into CDS and intron ranges.

    Returns half-open intervals [start, end).
    """

    sublists = []
    start_idx = 0
    in_zero_sequence = False
    for i, value in enumerate(lst):
        if value == 0:
            if not in_zero_sequence:
                if i != start_idx:
                    sublists.append((lst[start_idx:i], start_idx))
                in_zero_sequence = True
                start_idx = i + 1
        else:
            in_zero_sequence = False

    if start_idx < len(lst) and not all(v == 0 for v in lst[start_idx:]):
        sublists.append((lst[start_idx:], start_idx))

    all_sublist_ranges = []
    for sublist, start_index in sublists:
        sub_ranges = {"CDS": [], "intron": []}
        for target in targets:
            if target == 1:
                key = "CDS"
            elif target == 2:
                key = "intron"
            else:
                continue

            i = 0
            while i < len(sublist):
                if sublist[i] == target:
                    range_start = i
                    while i < len(sublist) and sublist[i] == target:
                        i += 1
                    range_end = i
                    sub_ranges[key].append((range_start + start_index, range_end + start_index))
                else:
                    i += 1

        all_sublist_ranges.append(sub_ranges)

    return all_sublist_ranges


def _summarize_candidate_block(
    block_predictions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return CDS-seed and genic-support masks for one 50 bp-aligned block."""

    block_length = int(block_predictions.shape[0])
    if block_length <= 0:
        return (
            np.empty(0, dtype=np.bool_),
            np.empty(0, dtype=np.bool_),
        )

    # Avoid materializing a float32 copy of all three CDS channels at once.
    # Per-worker temporary memory is bounded by CANDIDATE_SCAN_BLOCK_BP.
    cds_probability = block_predictions[:, 1].astype(np.float32)
    cds_probability += block_predictions[:, 2]
    cds_probability += block_predictions[:, 3]
    genic_probability = cds_probability + block_predictions[:, 4]

    bin_size = CANDIDATE_BIN_SIZE
    num_bins = (block_length + bin_size - 1) // bin_size
    mean_cds = np.empty(num_bins, dtype=np.float32)
    max_cds = np.empty(num_bins, dtype=np.float32)
    mean_genic = np.empty(num_bins, dtype=np.float32)

    num_full_bins = block_length // bin_size
    full_length = num_full_bins * bin_size
    if num_full_bins:
        cds_bins = cds_probability[:full_length].reshape(num_full_bins, bin_size)
        genic_bins = genic_probability[:full_length].reshape(num_full_bins, bin_size)
        mean_cds[:num_full_bins] = cds_bins.mean(axis=1, dtype=np.float32)
        max_cds[:num_full_bins] = cds_bins.max(axis=1)
        mean_genic[:num_full_bins] = genic_bins.mean(axis=1, dtype=np.float32)

    if full_length < block_length:
        tail_cds = cds_probability[full_length:block_length]
        tail_genic = genic_probability[full_length:block_length]
        mean_cds[-1] = tail_cds.mean(dtype=np.float32)
        max_cds[-1] = tail_cds.max()
        mean_genic[-1] = tail_genic.mean(dtype=np.float32)

    seed_mask = (
        (max_cds >= CANDIDATE_SEED_MAX_CDS)
        | (mean_cds >= CANDIDATE_SEED_MEAN_CDS)
    )
    support_mask = (
        (mean_genic >= CANDIDATE_GROW_MEAN_GENIC)
        | (max_cds >= CANDIDATE_GROW_MAX_CDS)
        | seed_mask
    )

    return seed_mask, support_mask


def _candidate_masks_to_regions(
    seed_mask: np.ndarray,
    support_mask: np.ndarray,
    seq_length: int,
) -> List[Tuple[int, int]]:
    """Apply global gap closing and convert bin masks to genomic intervals."""

    if seed_mask.shape != support_mask.shape:
        raise ValueError(
            "Candidate seed/support mask shape mismatch: "
            f"{seed_mask.shape} vs {support_mask.shape}."
        )

    num_bins = int(seed_mask.shape[0])
    bin_size = CANDIDATE_BIN_SIZE

    # Fill only internal unsupported runs whose total genomic span is at most
    # 500 bp. Difference-array filling avoids a Python loop over every bin.
    unsupported = (~support_mask).astype(np.int8, copy=False)
    transitions = np.diff(
        np.concatenate(
            (
                np.zeros(1, dtype=np.int8),
                unsupported,
                np.zeros(1, dtype=np.int8),
            )
        )
    )
    gap_starts = np.flatnonzero(transitions == 1)
    gap_ends = np.flatnonzero(transitions == -1)
    fillable = (
        (gap_starts > 0)
        & (gap_ends < num_bins)
        & ((gap_ends - gap_starts) * bin_size <= CANDIDATE_GAP_TOLERANCE_BP)
    )
    if np.any(fillable):
        fill_delta = np.zeros(num_bins + 1, dtype=np.int32)
        fill_delta[gap_starts[fillable]] += 1
        fill_delta[gap_ends[fillable]] -= 1
        support_mask |= np.cumsum(fill_delta[:-1]) > 0

    # Convert supported components back to genomic intervals, rejecting any
    # component that does not contain a genuine CDS seed.
    support_int = support_mask.astype(np.int8, copy=False)
    support_transitions = np.diff(
        np.concatenate(
            (
                np.zeros(1, dtype=np.int8),
                support_int,
                np.zeros(1, dtype=np.int8),
            )
        )
    )
    region_starts = np.flatnonzero(support_transitions == 1)
    region_ends = np.flatnonzero(support_transitions == -1)

    candidate_regions = []
    for start_bin, end_bin in zip(region_starts, region_ends):
        if not np.any(seed_mask[start_bin:end_bin]):
            continue
        candidate_regions.append(
            (
                int(start_bin * bin_size),
                min(int(end_bin * bin_size), int(seq_length)),
            )
        )

    return candidate_regions


def _detect_gene_location(
    base_predictions: np.ndarray,
    seq_length: int,
    num_workers: int = 1,
) -> List[Tuple[int, int]]:
    """Detect candidate regions with deterministic parallel bin statistics.

    Each worker receives a block aligned to the fixed 50 bp bin boundary and
    performs exactly the same NumPy operations as serial scanning. Results are
    consumed in block order. Gap closing and component extraction remain
    global and serial so candidate intervals are identical for every worker
    count.
    """

    if seq_length <= 0:
        return []
    if base_predictions.ndim != 2 or base_predictions.shape[1] != 5:
        raise ValueError(
            "Candidate detection expects predictions with shape (length, 5), "
            f"got {base_predictions.shape}."
        )
    if base_predictions.shape[0] != seq_length:
        raise ValueError(
            "Candidate detection length mismatch: predictions have "
            f"{base_predictions.shape[0]} bases, expected {seq_length}."
        )

    bin_size = CANDIDATE_BIN_SIZE
    block_bp = (CANDIDATE_SCAN_BLOCK_BP // bin_size) * bin_size
    if block_bp <= 0:
        raise ValueError(
            "CANDIDATE_SCAN_BLOCK_BP must contain at least one complete bin."
        )

    block_ranges = [
        (block_start, min(block_start + block_bp, seq_length))
        for block_start in range(0, seq_length, block_bp)
    ]
    effective_workers = min(
        max(1, int(num_workers)),
        CANDIDATE_SCAN_MAX_WORKERS,
        len(block_ranges),
    )
    block_views = [
        base_predictions[block_start:block_end]
        for block_start, block_end in block_ranges
    ]

    if effective_workers == 1:
        block_masks = [
            _summarize_candidate_block(block_view)
            for block_view in block_views
        ]
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            # executor.map preserves the input block order even when workers
            # finish out of order, keeping the scan deterministic.
            block_masks = list(
                executor.map(_summarize_candidate_block, block_views)
            )

    seed_mask = np.concatenate([masks[0] for masks in block_masks])
    support_mask = np.concatenate([masks[1] for masks in block_masks])
    return _candidate_masks_to_regions(seed_mask, support_mask, seq_length)


def _expand_and_merge_regions(
    regions: List[Tuple[int, int]],
    seq_length: int,
    buffer_size: int = DEFAULT_CANDIDATE_REGION_BUFFER,
) -> List[Tuple[int, int]]:
    """Expand candidate regions by a small buffer and merge overlaps."""

    if not regions:
        return []

    expanded = []
    for start, end in regions:
        s = max(0, int(start) - buffer_size)
        e = min(int(seq_length), int(end) + buffer_size)
        if s < e:
            expanded.append((s, e))

    if not expanded:
        return []

    expanded.sort(key=lambda x: x[0])
    merged = [expanded[0]]
    for s, e in expanded[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def _decode_gene_structure(
    location_start: int,
    predictions: np.ndarray,
    states_to_num: Dict[str, int],
    num_states: int,
    phase_0_columns: List[int],
    phase_1_columns: List[int],
    phase_2_columns: List[int],
    intron_columns: List[int],
    min_intron_length: int,
    splice_event_log_penalty: float,
    start_allowed_mask: np.ndarray,
    donor_scores: np.ndarray,
    acceptor_scores: np.ndarray,
    stop_end_scores: np.ndarray,
    readthrough_scores: np.ndarray,
    donor_classes: np.ndarray,
    acceptor_classes: np.ndarray,
    splice_pair_scores: np.ndarray,
    non_atg_start_log_penalty: float,
    min_cds_length: int,
    min_gene_length: int,
    min_gene_score: float,
) -> List[Tuple]:
    """Decode and filter gene structures in one candidate region."""

    gene_structure_all_states = _viterbi_decode(
        predictions=predictions,
        states_to_num=states_to_num,
        num_states=num_states,
        phase_0_columns=phase_0_columns,
        phase_1_columns=phase_1_columns,
        phase_2_columns=phase_2_columns,
        intron_columns=intron_columns,
        min_intron_length=min_intron_length,
        splice_event_log_penalty=splice_event_log_penalty,
        start_allowed_mask=start_allowed_mask,
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        stop_end_scores=stop_end_scores,
        readthrough_scores=readthrough_scores,
        donor_classes=donor_classes,
        acceptor_classes=acceptor_classes,
        splice_pair_scores=splice_pair_scores,
        non_atg_start_log_penalty=non_atg_start_log_penalty,
    )

    cds_columns = set(phase_0_columns + phase_1_columns + phase_2_columns)
    intron_column_set = set(intron_columns)
    gene_structure_three_states = [
        0 if x == states_to_num["intergenic"] else
        1 if x in cds_columns else
        2 if x in intron_column_set else 0
        for x in gene_structure_all_states
    ]

    gene_list = _parse_ranges(gene_structure_three_states, [1, 2])
    # Any of the model's three CDS-phase channels contributes CDS evidence.
    cds_base_scores = np.clip(
        predictions[:, 1:4].astype(np.float64).sum(axis=1),
        0.0,
        1.0,
    )
    filtered_gene_list = []

    for gene in gene_list:
        cds_list_init = gene["CDS"]
        if not cds_list_init:
            continue

        cds_count = sum(end - start for start, end in cds_list_init)
        if cds_count < min_cds_length:
            continue
        if cds_count % 3 != 0:
            continue

        cds_score = 0.0
        cds_num = 0
        cds_score_list = []
        for cds_start, cds_end in cds_list_init:
            cds_num_single = cds_end - cds_start
            cds_score_single_sum = float(cds_base_scores[cds_start:cds_end].sum())
            cds_score += cds_score_single_sum
            cds_num += cds_num_single
            cds_score_single = cds_score_single_sum / cds_num_single if cds_num_single > 0 else 0.0
            cds_score_list.append(cds_score_single)

        cds_score = cds_score / cds_num if cds_num != 0 else 0.0
        cds_list = [(start + location_start, end + location_start) for start, end in cds_list_init]
        first_cds_position = cds_list[0][0]
        gene_length = cds_list[-1][1] - first_cds_position

        if gene_length < min_gene_length:
            continue
        if cds_score < min_gene_score:
            continue

        filtered_gene_list.append((cds_list, cds_score, cds_score_list, first_cds_position))

    return filtered_gene_list


def _process_region_worker(args: Tuple) -> Tuple:
    """Worker function for parallel HMM decoding on one candidate region."""

    (
        chrom_id,
        strand,
        region_start,
        region_end,
        predictions_slice,
        states_to_num,
        num_states,
        phase_0_columns,
        phase_1_columns,
        phase_2_columns,
        intron_columns,
        min_intron_length,
        splice_event_log_penalty,
        transcript_sequence,
        non_atg_start_log_penalty,
        splice_motif_prior_strength,
        splice_pair_prior_strength,
        stop_codon_prior_strength,
        readthrough_prior_strength,
        min_cds_length,
        min_gene_length,
        min_gene_score,
    ) = args

    if len(transcript_sequence) != predictions_slice.shape[0]:
        raise ValueError(
            "Transcript sequence length must match prediction length: "
            f"{len(transcript_sequence)} != {predictions_slice.shape[0]}."
        )

    start_allowed_mask = _build_atg_start_mask(transcript_sequence)
    donor_scores, acceptor_scores, stop_end_scores = _build_sequence_prior_scores(
        transcript_sequence=transcript_sequence,
        splice_prior_strength=splice_motif_prior_strength,
        stop_prior_strength=stop_codon_prior_strength,
    )
    readthrough_scores = _build_readthrough_prior_scores(
        transcript_sequence=transcript_sequence,
        prior_strength=readthrough_prior_strength,
    )
    donor_classes, acceptor_classes = _build_splice_motif_classes(
        transcript_sequence
    )
    splice_pair_scores = _build_splice_pair_score_matrix(
        splice_pair_prior_strength
    )

    genes = _decode_gene_structure(
        location_start=region_start,
        predictions=predictions_slice,
        states_to_num=states_to_num,
        num_states=num_states,
        phase_0_columns=phase_0_columns,
        phase_1_columns=phase_1_columns,
        phase_2_columns=phase_2_columns,
        intron_columns=intron_columns,
        min_intron_length=min_intron_length,
        splice_event_log_penalty=splice_event_log_penalty,
        start_allowed_mask=start_allowed_mask,
        donor_scores=donor_scores,
        acceptor_scores=acceptor_scores,
        stop_end_scores=stop_end_scores,
        readthrough_scores=readthrough_scores,
        donor_classes=donor_classes,
        acceptor_classes=acceptor_classes,
        splice_pair_scores=splice_pair_scores,
        non_atg_start_log_penalty=non_atg_start_log_penalty,
        min_cds_length=min_cds_length,
        min_gene_length=min_gene_length,
        min_gene_score=min_gene_score,
    )

    return (chrom_id, strand, region_start, genes)


class HMMDecoder:
    """Minimal phase-aware HMM decoder for genomic-record-level PlantGeneAnn HDF5."""

    def __init__(
        self,
        cache_path: str,
        genome_fasta: str,
        output_gff: str,
        chromosome_h5_path: Optional[str] = None,
        min_intron_length: int = DEFAULT_HMM_MIN_INTRON_LENGTH,
        min_cds_length: int = DEFAULT_HMM_MIN_CDS_LENGTH,
        min_gene_length: int = DEFAULT_HMM_MIN_GENE_LENGTH,
        min_gene_score: float = DEFAULT_HMM_MIN_GENE_SCORE,
        splice_event_prob: float = DEFAULT_HMM_SPLICE_EVENT_PROB,
        num_threads: int = 8,
    ):
        self.cache_path = cache_path
        self.h5_path = (
            os.path.abspath(chromosome_h5_path)
            if chromosome_h5_path
            else os.path.join(cache_path, "chromosome_predictions.h5")
        )
        self.genome_fasta = genome_fasta
        self.output_gff = output_gff
        self.min_intron_length = int(min_intron_length)
        self.min_cds_length = int(min_cds_length)
        self.min_gene_length = int(min_gene_length)
        self.min_gene_score = float(min_gene_score)
        if self.min_intron_length < 1:
            raise ValueError(f"min_intron_length must be at least 1, got {min_intron_length}.")
        if self.min_gene_length < 30:
            raise ValueError(
                "min_gene_length must be at least 30 bp (10 codons), "
                f"got {min_gene_length}."
            )
        if not 0 < splice_event_prob <= 1:
            raise ValueError(
                f"splice_event_prob must be in the interval (0, 1], got {splice_event_prob}."
            )
        self.splice_event_prob = float(splice_event_prob)
        self.splice_event_log_penalty = float(np.log(self.splice_event_prob))
        self.non_atg_start_log_penalty = DEFAULT_NON_ATG_START_LOG_PENALTY
        self.splice_motif_prior_strength = float(
            DEFAULT_HMM_SPLICE_MOTIF_PRIOR_STRENGTH
        )
        self.splice_pair_prior_strength = float(
            DEFAULT_HMM_SPLICE_PAIR_PRIOR_STRENGTH
        )
        self.stop_codon_prior_strength = float(
            DEFAULT_HMM_STOP_CODON_PRIOR_STRENGTH
        )
        self.readthrough_prior_strength = float(
            DEFAULT_HMM_READTHROUGH_PRIOR_STRENGTH
        )
        if (
            not np.isfinite(self.splice_motif_prior_strength)
            or self.splice_motif_prior_strength < 0.0
        ):
            raise ValueError(
                "DEFAULT_HMM_SPLICE_MOTIF_PRIOR_STRENGTH must be finite and "
                f"non-negative, got {self.splice_motif_prior_strength}."
            )
        if (
            not np.isfinite(self.splice_pair_prior_strength)
            or self.splice_pair_prior_strength < 0.0
        ):
            raise ValueError(
                "DEFAULT_HMM_SPLICE_PAIR_PRIOR_STRENGTH must be finite and "
                f"non-negative, got {self.splice_pair_prior_strength}."
            )
        if (
            not np.isfinite(self.stop_codon_prior_strength)
            or self.stop_codon_prior_strength < 0.0
        ):
            raise ValueError(
                "DEFAULT_HMM_STOP_CODON_PRIOR_STRENGTH must be finite and "
                f"non-negative, got {self.stop_codon_prior_strength}."
            )
        if (
            not np.isfinite(self.readthrough_prior_strength)
            or self.readthrough_prior_strength < 0.0
        ):
            raise ValueError(
                "DEFAULT_HMM_READTHROUGH_PRIOR_STRENGTH must be finite and "
                f"non-negative, got {self.readthrough_prior_strength}."
            )
        _validate_motif_weights(
            DEFAULT_HMM_DONOR_MOTIF_WEIGHTS,
            name="DEFAULT_HMM_DONOR_MOTIF_WEIGHTS",
        )
        _validate_motif_weights(
            DEFAULT_HMM_ACCEPTOR_MOTIF_WEIGHTS,
            name="DEFAULT_HMM_ACCEPTOR_MOTIF_WEIGHTS",
        )
        _validate_motif_weights(
            DEFAULT_HMM_STOP_CODON_WEIGHTS,
            name="DEFAULT_HMM_STOP_CODON_WEIGHTS",
        )
        _validate_motif_weights(
            DEFAULT_HMM_READTHROUGH_CODON_WEIGHTS,
            name="DEFAULT_HMM_READTHROUGH_CODON_WEIGHTS",
        )
        _validate_splice_pair_weights(DEFAULT_HMM_SPLICE_PAIR_WEIGHTS)
        self.num_threads = int(num_threads)

        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"Chromosome-level HDF5 not found: {self.h5_path}")
        if not os.path.exists(self.genome_fasta):
            raise FileNotFoundError(f"Genome FASTA not found: {self.genome_fasta}")

        (
            self.states_to_num,
            self.num_states,
            self.phase_0_columns,
            self.phase_1_columns,
            self.phase_2_columns,
            self.intron_columns,
        ) = _define_states(self.min_intron_length)

    def _write_gff3_header(self, out_handle) -> None:
        """Write GFF3 header."""

        out_handle.write("##gff-version 3\n")
        out_handle.write(
            "# PlantGeneAnn ab initio gene structure prediction with minimal phase-aware HMM mode\n"
        )

    def _write_gene(
        self,
        out_handle,
        chrom_id: str,
        chrom_length: int,
        strand: str,
        strand_name: str,
        gene_num: int,
        gene_data: Tuple,
    ) -> int:
        """Write one gene to GFF3."""

        cds_list, cds_score, cds_score_list, _ = gene_data

        # ``cds_num`` is the number of previously emitted CDS bases modulo 3.
        # Internal codon positions first/second/third correspond to GFF3
        # phases 0/2/1 at the 5' boundary of the next CDS feature.
        phase_map = [0, 2, 1]
        cds_num = 0
        genomic_cds_features = []

        for i, cds in enumerate(cds_list):
            if strand == "+":
                cds_start0, cds_end0 = cds[0], cds[1]
            else:
                cds_start0 = chrom_length - cds[1]
                cds_end0 = chrom_length - cds[0]

            phase = str(phase_map[cds_num])
            cds_length = cds_end0 - cds_start0
            cds_num = (cds_num + cds_length) % 3
            cds_feature_score = cds_score_list[i] if i < len(cds_score_list) else 0.0
            genomic_cds_features.append(
                (cds_start0, cds_end0, phase, f"{cds_feature_score:.2f}")
            )

        genomic_cds_features.sort(key=lambda feature: feature[0])
        gene_start0 = genomic_cds_features[0][0]
        gene_end0 = genomic_cds_features[-1][1]

        gene_id = f"{chrom_id}.{strand_name}.gene{gene_num:06d}"
        mrna_id = f"{gene_id}.mRNA"
        gene_score = f"{cds_score:.2f}"

        out_handle.write(
            gff3_line(
                seqid=chrom_id,
                feature_type="gene",
                start0=gene_start0,
                end0=gene_end0,
                strand=strand,
                phase=".",
                attributes={
                    "ID": gene_id,
                    "Name": gene_id,
                    "biotype": "protein_coding",
                },
                score=gene_score,
            )
            + "\n"
        )
        out_handle.write(
            gff3_line(
                seqid=chrom_id,
                feature_type="mRNA",
                start0=gene_start0,
                end0=gene_end0,
                strand=strand,
                phase=".",
                attributes={
                    "ID": mrna_id,
                    "Parent": gene_id,
                },
                score=gene_score,
            )
            + "\n"
        )

        for cds_number, (cds_start0, cds_end0, phase, feature_score) in enumerate(
            genomic_cds_features,
            start=1,
        ):
            cds_id = f"{mrna_id}.CDS{cds_number:03d}"
            exon_id = f"{mrna_id}.exon{cds_number:03d}"
            out_handle.write(
                gff3_line(
                    seqid=chrom_id,
                    feature_type="exon",
                    start0=cds_start0,
                    end0=cds_end0,
                    strand=strand,
                    phase=".",
                    attributes={
                        "ID": exon_id,
                        "Parent": mrna_id,
                    },
                    score=feature_score,
                )
                + "\n"
            )
            out_handle.write(
                gff3_line(
                    seqid=chrom_id,
                    feature_type="CDS",
                    start0=cds_start0,
                    end0=cds_end0,
                    strand=strand,
                    phase=phase,
                    attributes={
                        "ID": cds_id,
                        "Parent": mrna_id,
                    },
                    score=feature_score,
                )
                + "\n"
            )

        return gene_num + 1

    def process(self) -> str:
        """Run HMM decoding on all genomic records in the prediction HDF5."""

        pipeline_start = time.monotonic()
        output_dir = os.path.dirname(os.path.abspath(self.output_gff))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        logger.info("Loading genomic records and preparing minimal HMM tasks...")
        tasks = []
        chrom_info = {}
        processed_records = 0
        skipped_records = 0
        zero_candidate_records = 0
        forward_candidate_regions = 0
        reverse_candidate_regions = 0
        scan_start = time.monotonic()

        fasta_index = pyfaidx.Fasta(self.genome_fasta, one_based_attributes=False)
        try:
            with h5py.File(self.h5_path, "r") as h5_file:
                chromosomes_root = h5_file["chromosomes"]
                index_group = h5_file["chromosome_index"]

                chrom_ids = [_decode_h5_string(v) for v in index_group["chrom_id"][:]]
                chrom_groups = [_decode_h5_string(v) for v in index_group["chrom_group"][:]]
                chrom_lengths = [int(v) for v in index_group["chrom_length"][:]]
                total_records = len(chrom_ids)
                scan_milestones = _build_progress_milestones(total_records)

                def log_scan_progress(record_number: int) -> None:
                    """Log cumulative candidate-scan statistics at 10% milestones."""

                    milestone = scan_milestones.get(record_number)
                    if milestone is None:
                        return
                    total_regions = forward_candidate_regions + reverse_candidate_regions
                    logger.info(
                        "Candidate scan %d%%: %d/%d genomic records, "
                        "processed=%d, skipped=%d, regions=%d, elapsed=%.1fs",
                        milestone,
                        record_number,
                        total_records,
                        processed_records,
                        skipped_records,
                        total_regions,
                        time.monotonic() - scan_start,
                    )

                for record_number, (chrom_id, chrom_group_name, chrom_length) in enumerate(
                    zip(chrom_ids, chrom_groups, chrom_lengths),
                    start=1,
                ):
                    if chrom_id not in fasta_index:
                        raise ValueError(
                            f"Genomic record {chrom_id!r} is present in HDF5 but not in genome FASTA."
                        )

                    if chrom_group_name not in chromosomes_root:
                        logger.warning("Genomic record %s not found in HDF5, skipping.", chrom_id)
                        skipped_records += 1
                        log_scan_progress(record_number)
                        continue

                    chrom_group = chromosomes_root[chrom_group_name]
                    if "full_probabilities" not in chrom_group:
                        logger.warning("Genomic record %s missing full_probabilities, skipping.", chrom_id)
                        skipped_records += 1
                        log_scan_progress(record_number)
                        continue

                    full_probs = chrom_group["full_probabilities"][:]
                    if full_probs.ndim != 3 or full_probs.shape[0] != 2 or full_probs.shape[2] != 5:
                        logger.warning(
                            "Genomic record %s: expected full_probabilities shape "
                            "(2, length, 5), got %s; skipping.",
                            chrom_id,
                            full_probs.shape,
                        )
                        skipped_records += 1
                        log_scan_progress(record_number)
                        continue
                    if full_probs.shape[1] != chrom_length:
                        logger.warning(
                            "Genomic record %s: full_probabilities length mismatch "
                            "(%d vs %d), skipping.",
                            chrom_id,
                            full_probs.shape[1],
                            chrom_length,
                        )
                        skipped_records += 1
                        log_scan_progress(record_number)
                        continue

                    chrom_info[chrom_id] = chrom_length
                    sequence_candidate_regions = 0
                    regions_by_strand = {}

                    for strand, strand_idx in [("+", 0), ("-", 1)]:
                        raw_predictions = full_probs[strand_idx, :, :]
                        potential_regions = _detect_gene_location(
                            raw_predictions,
                            chrom_length,
                            num_workers=self.num_threads,
                        )
                        potential_regions = _expand_and_merge_regions(
                            potential_regions,
                            chrom_length,
                            buffer_size=DEFAULT_CANDIDATE_REGION_BUFFER,
                        )
                        region_count = len(potential_regions)
                        sequence_candidate_regions += region_count
                        if strand == "+":
                            forward_candidate_regions += region_count
                        else:
                            reverse_candidate_regions += region_count

                        regions_by_strand[strand] = (
                            strand_idx,
                            potential_regions,
                        )

                    # Read each genomic record sequence once. Slicing this
                    # in-memory string is equivalent to repeated pyfaidx region
                    # fetches but avoids thousands of small indexed reads.
                    chrom_sequence = None
                    if sequence_candidate_regions:
                        chrom_sequence = str(
                            fasta_index[chrom_id][0:chrom_length]
                        ).upper()
                        if len(chrom_sequence) != chrom_length:
                            raise ValueError(
                                f"Genomic record length mismatch for {chrom_id}: "
                                f"FASTA={len(chrom_sequence)}, expected={chrom_length}."
                            )

                    # Preserve the original deterministic task order: positive
                    # strand first, then negative strand, with ascending forward
                    # genomic coordinates within each strand.
                    for strand in ("+", "-"):
                        strand_idx, potential_regions = regions_by_strand[strand]
                        raw_predictions = full_probs[strand_idx, :, :]

                        if strand == "+":
                            for region_start, region_end in potential_regions:
                                region_predictions = raw_predictions[region_start:region_end].copy()
                                region_sequence = chrom_sequence[region_start:region_end]
                                if len(region_sequence) != region_predictions.shape[0]:
                                    raise ValueError(
                                        f"Region length mismatch for {chrom_id} + "
                                        f"{region_start}-{region_end}: sequence "
                                        f"{len(region_sequence)} vs predictions "
                                        f"{region_predictions.shape[0]}."
                                    )
                                tasks.append((
                                    chrom_id,
                                    strand,
                                    region_start,
                                    region_end,
                                    region_predictions,
                                    self.states_to_num,
                                    self.num_states,
                                    self.phase_0_columns,
                                    self.phase_1_columns,
                                    self.phase_2_columns,
                                    self.intron_columns,
                                    self.min_intron_length,
                                    self.splice_event_log_penalty,
                                    region_sequence,
                                    self.non_atg_start_log_penalty,
                                    self.splice_motif_prior_strength,
                                    self.splice_pair_prior_strength,
                                    self.stop_codon_prior_strength,
                                    self.readthrough_prior_strength,
                                    self.min_cds_length,
                                    self.min_gene_length,
                                    self.min_gene_score,
                                ))
                        else:
                            for fwd_start, fwd_end in potential_regions:
                                rc_start = chrom_length - fwd_end
                                rc_end = chrom_length - fwd_start
                                region_predictions = raw_predictions[fwd_start:fwd_end][::-1].copy()
                                forward_sequence = chrom_sequence[fwd_start:fwd_end]
                                region_sequence = _reverse_complement_sequence(forward_sequence)
                                if len(region_sequence) != region_predictions.shape[0]:
                                    raise ValueError(
                                        f"Region length mismatch for {chrom_id} - "
                                        f"{fwd_start}-{fwd_end}: sequence "
                                        f"{len(region_sequence)} vs predictions "
                                        f"{region_predictions.shape[0]}."
                                    )
                                tasks.append((
                                    chrom_id,
                                    strand,
                                    rc_start,
                                    rc_end,
                                    region_predictions,
                                    self.states_to_num,
                                    self.num_states,
                                    self.phase_0_columns,
                                    self.phase_1_columns,
                                    self.phase_2_columns,
                                    self.intron_columns,
                                    self.min_intron_length,
                                    self.splice_event_log_penalty,
                                    region_sequence,
                                    self.non_atg_start_log_penalty,
                                    self.splice_motif_prior_strength,
                                    self.splice_pair_prior_strength,
                                    self.stop_codon_prior_strength,
                                    self.readthrough_prior_strength,
                                    self.min_cds_length,
                                    self.min_gene_length,
                                    self.min_gene_score,
                                ))

                    del full_probs, chrom_sequence

                    processed_records += 1
                    if sequence_candidate_regions == 0:
                        zero_candidate_records += 1
                    log_scan_progress(record_number)
        finally:
            fasta_index.close()

        results = []
        total_tasks = len(tasks)
        successful_tasks = 0
        failed_tasks = 0
        decoded_genes = 0
        decode_start = time.monotonic()
        decode_milestones = _build_progress_milestones(total_tasks)

        logger.info(
            "Running minimal HMM decoding for %d region tasks with %d workers...",
            total_tasks,
            self.num_threads,
        )
        if total_tasks > 0:
            with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
                future_to_task = {
                    executor.submit(_process_region_worker, task): task for task in tasks
                }
                for completed_tasks, future in enumerate(
                    as_completed(future_to_task),
                    start=1,
                ):
                    try:
                        result = future.result()
                        results.append(result)
                        successful_tasks += 1
                        decoded_genes += len(result[3])
                    except Exception as e:
                        failed_tasks += 1
                        task = future_to_task[future]
                        logger.error(
                            "HMM decoding failed for %s %s region %d-%d: %s",
                            task[0],
                            task[1],
                            task[2],
                            task[3],
                            str(e),
                        )

                    milestone = decode_milestones.get(completed_tasks)
                    if milestone is not None:
                        logger.info(
                            "HMM decoding %d%%: %d/%d tasks, "
                            "succeeded=%d, failed=%d, genes=%d, elapsed=%.1fs",
                            milestone,
                            completed_tasks,
                            total_tasks,
                            successful_tasks,
                            failed_tasks,
                            decoded_genes,
                            time.monotonic() - decode_start,
                        )

        grouped_results = defaultdict(lambda: {"forward": [], "reverse": []})
        for chrom_id, strand, _region_start, genes in results:
            if strand == "+":
                grouped_results[chrom_id]["forward"].extend(genes)
            else:
                grouped_results[chrom_id]["reverse"].extend(genes)

        total_genes = 0
        total_forward_genes = 0
        total_reverse_genes = 0
        with open(self.output_gff, "w", encoding="utf-8") as out_handle:
            self._write_gff3_header(out_handle)

            for chrom_id, chrom_length in chrom_info.items():
                out_handle.write(f"##sequence-region {chrom_id} 1 {chrom_length}\n")

                chrom_data = grouped_results.get(chrom_id, {"forward": [], "reverse": []})
                forward_genes = chrom_data["forward"]
                reverse_genes = chrom_data["reverse"]

                forward_genes.sort(key=lambda x: x[-1])
                reverse_genes.sort(key=lambda x: x[-1], reverse=True)
                total_forward_genes += len(forward_genes)
                total_reverse_genes += len(reverse_genes)

                emitted_any = False
                for strand, strand_name, genes in [
                    ("+", "plus", forward_genes),
                    ("-", "minus", reverse_genes),
                ]:
                    gene_num = 1
                    for gene in genes:
                        if emitted_any:
                            out_handle.write("###\n")
                        gene_num = self._write_gene(
                            out_handle,
                            chrom_id,
                            chrom_length,
                            strand,
                            strand_name,
                            gene_num,
                            gene,
                        )
                        emitted_any = True
                        total_genes += 1

        logger.info(
            "HMM annotation summary: decoding tasks=%d "
            "(succeeded=%d, failed=%d); genes=%d (+%d/-%d); "
            "elapsed=%.1fs;",
            total_tasks,
            successful_tasks,
            failed_tasks,
            total_genes,
            total_forward_genes,
            total_reverse_genes,
            time.monotonic() - pipeline_start,
        )
        return self.output_gff
