"""Shared constants for the PlantGeneAnn pipeline.

Centralises column names, HDF5 dtypes, and label mappings that were
shared by ``annotator.py`` and the direct chromosome-level streaming writer.
"""

from typing import List, Tuple

import h5py

# HDF5 helpers

H5_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def _decode_h5_string(value) -> str:
    """Decode one scalar HDF5 string value to Python ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

# Per-window metadata columns

# This dependency-neutral schema is the single source of truth for:
#   * the on-disk chunk_N.tsv column order,
#   * HuggingFace Datasets feature types, and
#   * metadata retained for distributed inference/HDF5 writing.
# Keep dtype names compatible with ``datasets.Value`` without importing the
# comparatively heavy HuggingFace Datasets package from this shared module.
CHUNK_TSV_SCHEMA: Tuple[Tuple[str, str], ...] = (
    ("global_window_index", "int64"),
    ("chrom_id", "string"),
    ("chrom_length", "int64"),
    ("chrom_index", "int64"),
    ("chrom_window_index", "int64"),
    ("center_start", "int64"),
    ("center_end", "int64"),
    ("chunk_id", "int64"),
    ("chunk_local_index", "int64"),
    ("sequence", "string"),
)

CHUNK_TSV_COLUMNS: Tuple[str, ...] = tuple(
    column_name for column_name, _ in CHUNK_TSV_SCHEMA
)

# Preserve the historical public list types while deriving their contents from
# the canonical TSV schema.
REQUIRED_METADATA_COLUMNS: List[str] = [
    column_name
    for column_name, _ in CHUNK_TSV_SCHEMA
    if column_name != "sequence"
]

INTEGER_METADATA_COLUMNS: List[str] = [
    column_name
    for column_name, dtype_name in CHUNK_TSV_SCHEMA
    if column_name != "sequence" and dtype_name == "int64"
]

# Per-base label definitions (15-state transcript-direction model)

# This schema name identifies the biological meaning and exact order of the
# per-strand probability channels. It is recorded in every chromosome-level
# prediction HDF5 file alongside the container-format version below.
# ``v2`` is the non-convertible 15-state successor to the former 5-state
# cache. The container version below is kept separately so HDF5 layout changes
# can be versioned independently of the biological state meaning.
PREDICTION_LABEL_SCHEMA = "plantgeneann_15state_transcript_v2"
PREDICTION_NUM_STRANDS = 2
CHROMOSOME_H5_FILE_FORMAT = "plantgeneann_chromosome_level_predictions"
CHROMOSOME_H5_FORMAT_VERSION = 2
PREDICTION_PROBABILITY_DTYPE = "float16"
PREDICTION_NORMALIZATION = "per_strand_softmax"
PREDICTION_LABEL_DIRECTION = "transcript_direction"
PREDICTION_MODEL_CHANNEL_LAYOUT = (
    "model_logits[:15]=positive;model_logits[15:30]=negative"
)

LABEL_BACKGROUND = 0
LABEL_INTRON_0 = 1
LABEL_INTRON_1 = 2
LABEL_INTRON_2 = 3
LABEL_CDS_FRAME_0 = 4
LABEL_CDS_FRAME_1 = 5
LABEL_CDS_FRAME_2 = 6
LABEL_START = 7
LABEL_DONOR_0 = 8
LABEL_DONOR_1 = 9
LABEL_DONOR_2 = 10
LABEL_ACCEPTOR_0 = 11
LABEL_ACCEPTOR_1 = 12
LABEL_ACCEPTOR_2 = 13
LABEL_STOP = 14

# State suffixes 0/1/2 follow the confirmed model transition graph (for
# example donor-0 -> intron-0 -> acceptor-1).  They are frame-indexed model
# states and must not be written directly as conventional GFF3 CDS phases.
LABEL_NAMES: Tuple[str, ...] = (
    "background",
    "intron_phase_0",
    "intron_phase_1",
    "intron_phase_2",
    "cds_frame_0",
    "cds_frame_1",
    "cds_frame_2",
    "start",
    "donor_phase_0",
    "donor_phase_1",
    "donor_phase_2",
    "acceptor_phase_0",
    "acceptor_phase_1",
    "acceptor_phase_2",
    "stop",
)

PREDICTION_NUM_CLASSES = len(LABEL_NAMES)
PREDICTION_NUM_CHANNELS = PREDICTION_NUM_STRANDS * PREDICTION_NUM_CLASSES

INTRON_LABELS: Tuple[int, ...] = (
    LABEL_INTRON_0,
    LABEL_INTRON_1,
    LABEL_INTRON_2,
)
FRAME_LABELS: Tuple[int, ...] = (
    LABEL_CDS_FRAME_0,
    LABEL_CDS_FRAME_1,
    LABEL_CDS_FRAME_2,
)
DONOR_LABELS: Tuple[int, ...] = (
    LABEL_DONOR_0,
    LABEL_DONOR_1,
    LABEL_DONOR_2,
)
ACCEPTOR_LABELS: Tuple[int, ...] = (
    LABEL_ACCEPTOR_0,
    LABEL_ACCEPTOR_1,
    LABEL_ACCEPTOR_2,
)
CODING_LABELS: Tuple[int, ...] = tuple(
    range(LABEL_CDS_FRAME_0, LABEL_STOP + 1)
)
NON_CODING_LABELS: Tuple[int, ...] = (LABEL_BACKGROUND, *INTRON_LABELS)

# Canonical label-level transition graph for the strict 15-state contract.
# Segment durations (notably introns) are represented by weighted DAG edges;
# this map is the authoritative per-base biological grammar.
DONOR_TO_INTRON = {
    LABEL_DONOR_0: LABEL_INTRON_0,
    LABEL_DONOR_1: LABEL_INTRON_1,
    LABEL_DONOR_2: LABEL_INTRON_2,
}
INTRON_TO_ACCEPTOR = {
    LABEL_INTRON_0: LABEL_ACCEPTOR_1,
    LABEL_INTRON_1: LABEL_ACCEPTOR_2,
    LABEL_INTRON_2: LABEL_ACCEPTOR_0,
}
ALLOWED_LABEL_TRANSITIONS = {
    LABEL_BACKGROUND: (LABEL_BACKGROUND, LABEL_START),
    LABEL_INTRON_0: (LABEL_INTRON_0, LABEL_ACCEPTOR_1),
    LABEL_INTRON_1: (LABEL_INTRON_1, LABEL_ACCEPTOR_2),
    LABEL_INTRON_2: (LABEL_INTRON_2, LABEL_ACCEPTOR_0),
    LABEL_CDS_FRAME_0: (LABEL_CDS_FRAME_1, LABEL_DONOR_1),
    LABEL_CDS_FRAME_1: (
        LABEL_CDS_FRAME_2,
        LABEL_DONOR_2,
        LABEL_STOP,
    ),
    LABEL_CDS_FRAME_2: (LABEL_CDS_FRAME_0, LABEL_DONOR_0),
    LABEL_START: (LABEL_CDS_FRAME_1,),
    LABEL_DONOR_0: (LABEL_INTRON_0,),
    LABEL_DONOR_1: (LABEL_INTRON_1,),
    LABEL_DONOR_2: (LABEL_INTRON_2,),
    LABEL_ACCEPTOR_0: (LABEL_CDS_FRAME_1,),
    LABEL_ACCEPTOR_1: (LABEL_CDS_FRAME_2,),
    LABEL_ACCEPTOR_2: (LABEL_CDS_FRAME_0,),
    LABEL_STOP: (LABEL_BACKGROUND,),
}

# Splice pairs accepted by the strict SMM decoder, in transcript direction.
# Training-data QC imports this same contract so unsupported introns can never
# contribute labels that the production decoder is unable to emit.
ALLOWED_SPLICE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("GT", "AG"),
    ("GC", "AG"),
    ("AT", "AC"),
)

LABEL_MAPPING_STRING: str = ";".join(
    f"{idx}={name}" for idx, name in enumerate(LABEL_NAMES)
)
LABEL_CLASS_ORDER_STRING: str = ",".join(LABEL_NAMES)
