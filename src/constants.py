"""Shared constants for the PlantGeneAnn pipeline.

Centralises column names, HDF5 dtypes, and label mappings that were
shared by ``annotator.py`` and the direct chromosome-level streaming writer.
"""

from typing import List

import h5py

# HDF5 helpers

H5_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def _decode_h5_string(value) -> str:
    """Decode one scalar HDF5 string value to Python ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

# Per-window metadata columns

REQUIRED_METADATA_COLUMNS: List[str] = [
    "global_window_index",
    "chrom_id",
    "chrom_length",
    "chrom_index",
    "chrom_window_index",
    "center_start",
    "center_end",
    "chunk_id",
    "chunk_local_index",
]

INTEGER_METADATA_COLUMNS: List[str] = [
    "global_window_index",
    "chrom_length",
    "chrom_index",
    "chrom_window_index",
    "center_start",
    "center_end",
    "chunk_id",
    "chunk_local_index",
]

# Per-base label definitions (v2 model)

LABEL_NAMES: List[str] = [
    "Intergenic",
    "CDS-phase0",
    "CDS-phase1",
    "CDS-phase2",
    "Intron",
]

LABEL_MAPPING_STRING: str = ";".join(
    f"{idx}={name}" for idx, name in enumerate(LABEL_NAMES)
)
