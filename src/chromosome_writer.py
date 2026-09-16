"""Direct streaming writer for genomic-record-level prediction HDF5 files.

The distributed annotator already owns ordered, center-cropped window
probabilities on Accelerate global rank zero.  This module writes those arrays
directly into their genomic coordinates, avoiding the former window-level HDF5
copy and its subsequent rebuild pass.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple
from urllib.parse import quote

import h5py
import numpy as np

from .constants import (
    CHROMOSOME_H5_FILE_FORMAT,
    CHROMOSOME_H5_FORMAT_VERSION,
    H5_STRING_DTYPE,
    LABEL_CLASS_ORDER_STRING,
    LABEL_MAPPING_STRING,
    PREDICTION_LABEL_DIRECTION,
    PREDICTION_LABEL_SCHEMA,
    PREDICTION_MODEL_CHANNEL_LAYOUT,
    PREDICTION_NORMALIZATION,
    PREDICTION_NUM_CHANNELS,
    PREDICTION_NUM_CLASSES,
    PREDICTION_NUM_STRANDS,
    PREDICTION_PROBABILITY_DTYPE,
    REQUIRED_METADATA_COLUMNS,
    _decode_h5_string,
)

logger = logging.getLogger("PlantGeneAnn.src.chromosome_writer")


def _attribute_text(owner, attribute_name: str) -> Optional[str]:
    """Return one HDF5 attribute as text, or ``None`` when it is absent."""

    if attribute_name not in owner.attrs:
        return None
    return _decode_h5_string(owner.attrs[attribute_name])


def _require_text_attribute(
    owner,
    attribute_name: str,
    expected_value: str,
    *,
    context: str,
) -> None:
    """Raise a clear error when one schema text attribute is not exact."""

    actual_value = _attribute_text(owner, attribute_name)
    if actual_value != expected_value:
        raise RuntimeError(
            f"{context} has incompatible {attribute_name!r}: "
            f"expected {expected_value!r}, got {actual_value!r}."
        )


def _require_integer_attribute(
    owner,
    attribute_name: str,
    expected_value: int,
    *,
    context: str,
) -> None:
    """Raise a clear error when one schema integer attribute is not exact."""

    if attribute_name not in owner.attrs:
        actual_value = None
    else:
        try:
            actual_value = int(owner.attrs[attribute_name])
        except (TypeError, ValueError, OverflowError):
            actual_value = None
    if actual_value != int(expected_value):
        raise RuntimeError(
            f"{context} has incompatible {attribute_name!r}: "
            f"expected {expected_value}, got {actual_value!r}."
        )


def _validate_prediction_schema(
    h5_file: h5py.File,
    *,
    expected_status: Optional[str] = None,
) -> int:
    """Validate the immutable 15-state chromosome-HDF5 root contract.

    The former 5-state cache used the same broad ``file_format`` identifier,
    so a missing or different schema version is treated as an explicitly
    unsupported legacy cache rather than being interpreted permissively.
    """

    context = "Chromosome HDF5 root"
    _require_text_attribute(
        h5_file,
        "file_format",
        CHROMOSOME_H5_FILE_FORMAT,
        context=context,
    )

    if "file_format_version" not in h5_file.attrs:
        raise RuntimeError(
            "Unsupported legacy chromosome HDF5: missing 'file_format_version'. "
            "The 15-state decoder cannot convert a 5-state prediction cache; "
            "rerun deep-learning prediction with the current pipeline."
        )
    try:
        file_format_version = int(h5_file.attrs["file_format_version"])
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "Chromosome HDF5 has a non-integer 'file_format_version'; "
            "rerun prediction with the current 15-state pipeline."
        ) from error
    if file_format_version != CHROMOSOME_H5_FORMAT_VERSION:
        raise RuntimeError(
            "Unsupported chromosome HDF5 schema version "
            f"{file_format_version}; expected {CHROMOSOME_H5_FORMAT_VERSION} "
            f"for {PREDICTION_LABEL_SCHEMA}. Legacy 5-state caches are not "
            "convertible and must be regenerated."
        )

    _require_text_attribute(
        h5_file,
        "write_mode",
        "direct_streaming",
        context=context,
    )
    _require_text_attribute(
        h5_file,
        "coordinate_system",
        "0-based half-open genomic coordinates",
        context=context,
    )
    _require_text_attribute(
        h5_file,
        "prediction_schema",
        PREDICTION_LABEL_SCHEMA,
        context=context,
    )
    _require_integer_attribute(
        h5_file,
        "num_strands",
        PREDICTION_NUM_STRANDS,
        context=context,
    )
    _require_integer_attribute(
        h5_file,
        "num_classes_per_strand",
        PREDICTION_NUM_CLASSES,
        context=context,
    )
    _require_integer_attribute(
        h5_file,
        "model_output_channels",
        PREDICTION_NUM_CHANNELS,
        context=context,
    )
    _require_text_attribute(
        h5_file,
        "probability_dtype",
        PREDICTION_PROBABILITY_DTYPE,
        context=context,
    )
    _require_text_attribute(
        h5_file,
        "probability_normalization",
        PREDICTION_NORMALIZATION,
        context=context,
    )
    _require_text_attribute(
        h5_file,
        "label_direction",
        PREDICTION_LABEL_DIRECTION,
        context=context,
    )
    _require_text_attribute(
        h5_file,
        "model_channel_layout",
        PREDICTION_MODEL_CHANNEL_LAYOUT,
        context=context,
    )
    _require_text_attribute(
        h5_file,
        "full_probability_class_order",
        LABEL_CLASS_ORDER_STRING,
        context=context,
    )
    _require_text_attribute(
        h5_file,
        "label_mapping",
        LABEL_MAPPING_STRING,
        context=context,
    )

    expected_shape_description = (
        f"({PREDICTION_NUM_STRANDS}, genomic_record_length, "
        f"{PREDICTION_NUM_CLASSES})"
    )
    _require_text_attribute(
        h5_file,
        "full_probability_shape",
        expected_shape_description,
        context=context,
    )
    _require_text_attribute(
        h5_file,
        "strand_axis",
        "0=positive;1=negative",
        context=context,
    )
    _require_text_attribute(
        h5_file,
        "gap_policy",
        "no_gaps_allowed",
        context=context,
    )

    if expected_status is not None:
        _require_text_attribute(
            h5_file,
            "status",
            expected_status,
            context=context,
        )

    if "hdf5_chunk_bp" not in h5_file.attrs:
        raise RuntimeError("Chromosome HDF5 root is missing 'hdf5_chunk_bp'.")
    try:
        hdf5_chunk_bp = int(h5_file.attrs["hdf5_chunk_bp"])
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            "Chromosome HDF5 root has an invalid 'hdf5_chunk_bp'."
        ) from error
    if hdf5_chunk_bp <= 0:
        raise RuntimeError(
            "Chromosome HDF5 root has a non-positive 'hdf5_chunk_bp': "
            f"{hdf5_chunk_bp}."
        )
    return hdf5_chunk_bp


def _validate_probability_dataset_geometry(
    dataset: h5py.Dataset,
    *,
    chrom_id: str,
    chrom_length: int,
) -> None:
    """Validate decoding-critical dataset geometry without loading its payload."""

    context = f"Genomic record {chrom_id!r} full_probabilities"
    if not isinstance(dataset, h5py.Dataset):
        raise RuntimeError(f"{context} must be an HDF5 dataset.")
    expected_shape = (
        PREDICTION_NUM_STRANDS,
        int(chrom_length),
        PREDICTION_NUM_CLASSES,
    )
    if dataset.shape != expected_shape:
        raise RuntimeError(
            f"{context} has invalid probability shape {dataset.shape}; "
            f"expected {expected_shape}."
        )
    expected_dtype = np.dtype(PREDICTION_PROBABILITY_DTYPE)
    if dataset.dtype != expected_dtype:
        raise RuntimeError(
            f"{context} has invalid dtype {dataset.dtype}; expected {expected_dtype}."
        )


def _validate_probability_dataset(
    dataset: h5py.Dataset,
    *,
    chrom_id: str,
    chrom_length: int,
    hdf5_chunk_bp: int,
) -> None:
    """Strictly validate one dataset's complete 15-state metadata contract."""

    _validate_probability_dataset_geometry(
        dataset,
        chrom_id=chrom_id,
        chrom_length=chrom_length,
    )
    context = f"Genomic record {chrom_id!r} full_probabilities"
    expected_chunks = (
        1,
        min(int(chrom_length), int(hdf5_chunk_bp)),
        PREDICTION_NUM_CLASSES,
    )
    if dataset.chunks != expected_chunks:
        raise RuntimeError(
            f"{context} has invalid HDF5 chunk geometry {dataset.chunks}; "
            f"expected {expected_chunks}."
        )
    _require_text_attribute(
        dataset,
        "prediction_schema",
        PREDICTION_LABEL_SCHEMA,
        context=context,
    )
    _require_integer_attribute(
        dataset,
        "num_strands",
        PREDICTION_NUM_STRANDS,
        context=context,
    )
    _require_integer_attribute(
        dataset,
        "num_classes_per_strand",
        PREDICTION_NUM_CLASSES,
        context=context,
    )
    _require_text_attribute(
        dataset,
        "probability_dtype",
        PREDICTION_PROBABILITY_DTYPE,
        context=context,
    )
    _require_text_attribute(
        dataset,
        "strand_axis",
        "0=positive;1=negative",
        context=context,
    )
    _require_text_attribute(
        dataset,
        "class_order",
        LABEL_CLASS_ORDER_STRING,
        context=context,
    )
    _require_text_attribute(
        dataset,
        "label_mapping",
        LABEL_MAPPING_STRING,
        context=context,
    )
    _require_text_attribute(
        dataset,
        "probability_normalization",
        PREDICTION_NORMALIZATION,
        context=context,
    )
    _require_text_attribute(
        dataset,
        "label_direction",
        PREDICTION_LABEL_DIRECTION,
        context=context,
    )


def _validate_chromosome_group_layout(
    chrom_group: h5py.Group,
    info: "ChromosomeInfo",
    *,
    hdf5_chunk_bp: int,
    require_complete: bool,
) -> None:
    """Validate one record group's metadata and probability dataset contract."""

    context = f"Genomic record group for {info.chrom_id!r}"
    if not isinstance(chrom_group, h5py.Group):
        raise RuntimeError(f"{context} must be an HDF5 group.")
    _require_text_attribute(
        chrom_group,
        "chrom_id",
        info.chrom_id,
        context=context,
    )
    _require_integer_attribute(
        chrom_group,
        "chrom_length",
        info.chrom_length,
        context=context,
    )
    _require_integer_attribute(
        chrom_group,
        "chrom_index",
        info.chrom_index,
        context=context,
    )
    _require_integer_attribute(
        chrom_group,
        "expected_num_windows",
        info.num_windows,
        context=context,
    )
    _require_text_attribute(
        chrom_group,
        "coordinate_system",
        "0-based half-open",
        context=context,
    )
    _require_text_attribute(
        chrom_group,
        "strand_axis",
        "0=positive;1=negative",
        context=context,
    )

    if "covered_end" not in chrom_group.attrs or "windows_written" not in chrom_group.attrs:
        raise RuntimeError(f"{context} is missing direct-write coverage metadata.")
    try:
        covered_end = int(chrom_group.attrs["covered_end"])
        windows_written = int(chrom_group.attrs["windows_written"])
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            f"{context} has invalid direct-write coverage metadata."
        ) from error
    if covered_end < 0 or covered_end > info.chrom_length:
        raise RuntimeError(
            f"{context} has invalid covered_end={covered_end}; expected a value "
            f"from 0 to {info.chrom_length}."
        )
    if windows_written < 0 or windows_written > info.num_windows:
        raise RuntimeError(
            f"{context} has invalid windows_written={windows_written}; expected a value "
            f"from 0 to {info.num_windows}."
        )
    if require_complete:
        if covered_end != info.chrom_length:
            raise RuntimeError(f"Genomic record {info.chrom_id!r} has incomplete coverage.")
        if windows_written != info.num_windows:
            raise RuntimeError(
                f"Genomic record {info.chrom_id!r} has an incomplete window count."
            )

    if "full_probabilities" not in chrom_group:
        raise RuntimeError(
            f"Genomic record {info.chrom_id!r} is missing full_probabilities."
        )
    _validate_probability_dataset(
        chrom_group["full_probabilities"],
        chrom_id=info.chrom_id,
        chrom_length=info.chrom_length,
        hdf5_chunk_bp=hdf5_chunk_bp,
    )


@dataclass(frozen=True)
class ChromosomeInfo:
    """Expected output geometry for one FASTA genomic record."""

    chrom_id: str
    chrom_length: int
    chrom_index: int
    num_windows: int
    group_name: str


def _chrom_group_name(chrom_id: str) -> str:
    """Return a reversible HDF5-safe group name for a FASTA record ID."""

    encoded = quote(str(chrom_id), safe="")
    if not encoded:
        raise ValueError("Encountered an empty genomic-record ID.")
    return encoded


def _read_and_validate_manifest_index(
    h5_file: h5py.File,
    *,
    require_complete: bool,
    chrom_sequence_info: Optional[Mapping[str, Tuple[int, int]]] = None,
) -> Tuple[Dict[str, ChromosomeInfo], Dict[int, ChromosomeInfo]]:
    """Validate the columnar manifest without opening every record group.

    This is the lightweight path used by streaming prediction and decoding.
    It deliberately limits HDF5 metadata access to the root and the compact
    ``chromosome_index`` datasets. Record-group coverage and probability
    geometry are validated when that record is first accessed.
    """

    if "chromosome_index" not in h5_file or "chromosomes" not in h5_file:
        raise RuntimeError("Chromosome HDF5 is missing required groups.")

    index_group = h5_file["chromosome_index"]
    chromosomes_root = h5_file["chromosomes"]
    if not isinstance(index_group, h5py.Group):
        raise RuntimeError("chromosome_index must be an HDF5 group.")
    if not isinstance(chromosomes_root, h5py.Group):
        raise RuntimeError("chromosomes must be an HDF5 group.")

    required_index_columns = (
        "chrom_id",
        "chrom_group",
        "chrom_length",
        "chrom_index",
        "num_windows",
    )
    missing_columns = [
        column for column in required_index_columns if column not in index_group
    ]
    if missing_columns:
        raise RuntimeError(
            f"chromosome_index is missing datasets: {missing_columns}"
        )
    non_datasets = [
        column
        for column in required_index_columns
        if not isinstance(index_group[column], h5py.Dataset)
    ]
    if non_datasets:
        raise RuntimeError(
            f"chromosome_index entries are not datasets: {non_datasets}"
        )

    chrom_ids = [_decode_h5_string(value) for value in index_group["chrom_id"][:]]
    group_names = [
        _decode_h5_string(value) for value in index_group["chrom_group"][:]
    ]
    chrom_lengths = [int(value) for value in index_group["chrom_length"][:]]
    chrom_indices = [int(value) for value in index_group["chrom_index"][:]]
    num_windows = [int(value) for value in index_group["num_windows"][:]]
    row_counts = {
        len(chrom_ids),
        len(group_names),
        len(chrom_lengths),
        len(chrom_indices),
        len(num_windows),
    }
    if len(row_counts) != 1 or not chrom_ids:
        raise RuntimeError(
            "chromosome_index datasets have inconsistent or empty lengths."
        )
    if chrom_indices != list(range(len(chrom_indices))):
        raise RuntimeError(
            "chromosome_index values must be consecutive FASTA order."
        )
    if len(set(chrom_ids)) != len(chrom_ids):
        raise RuntimeError(
            "chromosome_index contains duplicate genomic-record IDs."
        )
    if len(set(group_names)) != len(group_names):
        raise RuntimeError(
            "chromosome_index contains duplicate chromosome groups."
        )

    _require_integer_attribute(
        h5_file,
        "num_chromosomes",
        len(chrom_ids),
        context="Chromosome HDF5 root",
    )
    _require_integer_attribute(
        h5_file,
        "expected_num_windows",
        sum(num_windows),
        context="Chromosome HDF5 root",
    )
    if require_complete:
        _require_integer_attribute(
            h5_file,
            "num_source_windows",
            sum(num_windows),
            context="Chromosome HDF5 root",
        )
        if "num_source_chunks" not in h5_file.attrs:
            raise RuntimeError(
                "Completed chromosome HDF5 is missing 'num_source_chunks'."
            )
        try:
            num_source_chunks = int(h5_file.attrs["num_source_chunks"])
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(
                "Completed chromosome HDF5 has invalid 'num_source_chunks'."
            ) from error
        if num_source_chunks <= 0:
            raise RuntimeError(
                "Completed chromosome HDF5 must record at least one source chunk."
            )

    manifest: Dict[str, ChromosomeInfo] = {}
    manifest_by_index: Dict[int, ChromosomeInfo] = {}
    for chrom_id, group_name, chrom_length, chrom_index, expected_windows in zip(
        chrom_ids,
        group_names,
        chrom_lengths,
        chrom_indices,
        num_windows,
    ):
        if group_name != _chrom_group_name(chrom_id):
            raise RuntimeError(
                "chromosome_index has a non-canonical group name for "
                f"{chrom_id!r}: {group_name!r}."
            )
        if chrom_length <= 0 or expected_windows <= 0:
            raise RuntimeError(
                f"chromosome_index has invalid geometry for {chrom_id!r}."
            )
        info = ChromosomeInfo(
            chrom_id=chrom_id,
            chrom_length=chrom_length,
            chrom_index=chrom_index,
            num_windows=expected_windows,
            group_name=group_name,
        )
        manifest[chrom_id] = info
        manifest_by_index[chrom_index] = info

    if len(chromosomes_root) != len(manifest):
        raise RuntimeError(
            "chromosomes group count does not match chromosome_index."
        )

    if chrom_sequence_info is not None:
        expected_ids = list(chrom_sequence_info)
        if chrom_ids != expected_ids:
            raise RuntimeError(
                "Chromosome HDF5 record order differs from the extraction manifest."
            )
        for info in manifest.values():
            expected_length, expected_windows = chrom_sequence_info[info.chrom_id]
            if (info.chrom_length, info.num_windows) != (
                int(expected_length),
                int(expected_windows),
            ):
                raise RuntimeError(
                    f"Chromosome HDF5 manifest mismatch for {info.chrom_id!r}."
                )

    return manifest, manifest_by_index


def _read_and_validate_manifest(
    h5_file: h5py.File,
    *,
    hdf5_chunk_bp: int,
    require_complete: bool,
    require_fresh: bool = False,
    chrom_sequence_info: Optional[Mapping[str, Tuple[int, int]]] = None,
) -> Tuple[Dict[str, ChromosomeInfo], Dict[int, ChromosomeInfo]]:
    """Strictly validate the manifest and every record-group metadata object."""

    manifest, manifest_by_index = _read_and_validate_manifest_index(
        h5_file,
        require_complete=require_complete,
        chrom_sequence_info=chrom_sequence_info,
    )
    chromosomes_root = h5_file["chromosomes"]
    expected_group_names = {info.group_name for info in manifest.values()}
    if set(chromosomes_root.keys()) != expected_group_names:
        raise RuntimeError(
            "chromosomes group does not exactly match chromosome_index."
        )
    for info in manifest.values():
        chrom_group = chromosomes_root[info.group_name]
        _validate_chromosome_group_layout(
            chrom_group,
            info,
            hdf5_chunk_bp=hdf5_chunk_bp,
            require_complete=require_complete,
        )
        if require_fresh and (
            int(chrom_group.attrs["covered_end"]) != 0
            or int(chrom_group.attrs["windows_written"]) != 0
        ):
            raise RuntimeError(
                "ChromosomePredictionWriter only accepts a fresh incomplete "
                f"HDF5; {info.chrom_id!r} already has written coverage."
            )

    return manifest, manifest_by_index


def _read_integer_attribute(owner, attribute_name: str, *, context: str) -> int:
    """Read one required integer attribute with a focused validation error."""

    if attribute_name not in owner.attrs:
        raise RuntimeError(f"{context} is missing {attribute_name!r}.")
    try:
        return int(owner.attrs[attribute_name])
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            f"{context} has invalid {attribute_name!r}."
        ) from error


def get_chromosome_probability_dataset(
    chromosomes_root: h5py.Group,
    info: ChromosomeInfo,
    *,
    require_complete: bool = False,
    require_fresh: bool = False,
) -> h5py.Dataset:
    """Return one record dataset after lightweight access-time validation.

    Root schema and the columnar chromosome manifest are authoritative. This
    helper therefore checks only metadata needed for safe writing/decoding:
    group and dataset existence, coverage state, probability shape, and dtype.
    Repeated descriptive attributes and exact chunk geometry remain part of the
    explicit strict validator rather than the normal pipeline hot path.
    """

    if require_complete and require_fresh:
        raise ValueError(
            "Record validation cannot require complete and fresh coverage together."
        )
    if not isinstance(chromosomes_root, h5py.Group):
        raise RuntimeError("chromosomes must be an HDF5 group.")
    if info.group_name not in chromosomes_root:
        raise RuntimeError(
            f"Chromosome HDF5 is missing the group for {info.chrom_id!r}."
        )

    chrom_group = chromosomes_root[info.group_name]
    context = f"Genomic record group for {info.chrom_id!r}"
    if not isinstance(chrom_group, h5py.Group):
        raise RuntimeError(f"{context} must be an HDF5 group.")

    if require_complete or require_fresh:
        covered_end = _read_integer_attribute(
            chrom_group,
            "covered_end",
            context=context,
        )
        windows_written = _read_integer_attribute(
            chrom_group,
            "windows_written",
            context=context,
        )
        expected_covered_end = info.chrom_length if require_complete else 0
        expected_windows_written = info.num_windows if require_complete else 0
        if (
            covered_end != expected_covered_end
            or windows_written != expected_windows_written
        ):
            expected_state = "complete" if require_complete else "fresh"
            raise RuntimeError(
                f"{context} is not {expected_state}: covered_end={covered_end}, "
                f"windows_written={windows_written}."
            )

    if "full_probabilities" not in chrom_group:
        raise RuntimeError(
            f"Genomic record {info.chrom_id!r} is missing full_probabilities."
        )
    dataset = chrom_group["full_probabilities"]
    _validate_probability_dataset_geometry(
        dataset,
        chrom_id=info.chrom_id,
        chrom_length=info.chrom_length,
    )
    return dataset


def _validate_fresh_root_write_counters(h5_file: h5py.File) -> None:
    """Reject a previously written incomplete file without scanning its groups."""

    context = "Chromosome HDF5 root"
    for attribute_name in ("num_source_chunks", "num_source_windows"):
        if attribute_name not in h5_file.attrs:
            continue
        value = _read_integer_attribute(
            h5_file,
            attribute_name,
            context=context,
        )
        if value != 0:
            raise RuntimeError(
                "ChromosomePredictionWriter only accepts a fresh incomplete "
                f"HDF5; root {attribute_name}={value}."
            )


def _normalise_compression(compression: str):
    value = str(compression).lower()
    if value in {"none", "null", "no"}:
        return None
    if value != "gzip":
        raise ValueError("Only gzip compression or 'none' are supported.")
    return "gzip"


def prepare_chromosome_h5_output(output_h5_path: str) -> Tuple[str, str]:
    """Prepare one-output-file semantics for a direct prediction run.

    The prediction-cache command historically overwrote its output. Removing
    both the old completed output and any stale temporary file before inference
    preserves that behavior while preventing old and new chromosome HDF5 files
    from consuming disk space simultaneously.
    """

    output_h5_path = os.path.abspath(output_h5_path)
    temporary_h5_path = f"{output_h5_path}.tmp"
    for stale_path in (temporary_h5_path, output_h5_path):
        if os.path.isdir(stale_path):
            raise IsADirectoryError(
                f"Prediction HDF5 path unexpectedly refers to a directory: {stale_path}"
            )
        if os.path.isfile(stale_path):
            os.remove(stale_path)
            logger.warning("Removed stale prediction output: %s", stale_path)
    return output_h5_path, temporary_h5_path


def cleanup_temporary_chromosome_h5(temporary_h5_path: str) -> bool:
    """Delete one exact temporary prediction HDF5 path if it exists.

    The caller invokes this from a ``finally`` block around initialization,
    inference, and atomic promotion. A successfully promoted file no longer
    exists at the temporary path, so the completed output is never removed.

    Returns:
        ``True`` when a temporary file was deleted, otherwise ``False``.
    """

    temporary_h5_path = os.path.abspath(temporary_h5_path)
    if os.path.isdir(temporary_h5_path):
        raise IsADirectoryError(
            "Temporary prediction HDF5 path unexpectedly refers to a directory: "
            f"{temporary_h5_path}"
        )
    if not os.path.isfile(temporary_h5_path):
        return False

    os.remove(temporary_h5_path)
    logger.info("Removed temporary prediction HDF5: %s", temporary_h5_path)
    return True


def initialize_chromosome_h5(
    output_h5_path: str,
    chrom_sequence_info: Mapping[str, Tuple[int, int]],
    *,
    compression: str = "none",
    compression_level: int = 4,
    hdf5_chunk_bp: int = 1_000_000,
) -> str:
    """Create an incomplete genomic-record HDF5 ready for direct writes.

    ``chrom_sequence_info`` must preserve FASTA order and map each record ID to
    ``(record_length, expected_window_count)``.  The file remains explicitly
    marked ``status=incomplete`` until :class:`ChromosomePredictionWriter`
    validates full coverage after the last inference chunk.
    """

    if not chrom_sequence_info:
        raise ValueError("Cannot initialize predictions for an empty record manifest.")
    if hdf5_chunk_bp <= 0:
        raise ValueError(f"hdf5_chunk_bp must be positive, got {hdf5_chunk_bp}.")
    if not 0 <= int(compression_level) <= 9:
        raise ValueError(
            f"compression_level must be between 0 and 9, got {compression_level}."
        )

    compression_name = _normalise_compression(compression)
    output_h5_path = os.path.abspath(output_h5_path)
    output_dir = os.path.dirname(output_h5_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    manifest = []
    for chrom_index, (chrom_id, values) in enumerate(chrom_sequence_info.items()):
        chrom_length, num_windows = (int(values[0]), int(values[1]))
        if not str(chrom_id):
            raise ValueError("Genomic-record IDs must be non-empty strings.")
        if chrom_length <= 0:
            raise ValueError(
                f"Genomic record {chrom_id!r} has invalid length {chrom_length}."
            )
        if num_windows <= 0:
            raise ValueError(
                f"Genomic record {chrom_id!r} has invalid window count {num_windows}."
            )
        manifest.append(
            ChromosomeInfo(
                chrom_id=str(chrom_id),
                chrom_length=chrom_length,
                chrom_index=chrom_index,
                num_windows=num_windows,
                group_name=_chrom_group_name(str(chrom_id)),
            )
        )

    with h5py.File(output_h5_path, "w") as output_h5:
        output_h5.attrs["file_format"] = CHROMOSOME_H5_FILE_FORMAT
        output_h5.attrs["file_format_version"] = CHROMOSOME_H5_FORMAT_VERSION
        output_h5.attrs["write_mode"] = "direct_streaming"
        output_h5.attrs["status"] = "incomplete"
        output_h5.attrs["coordinate_system"] = "0-based half-open genomic coordinates"
        output_h5.attrs["full_probability_shape"] = (
            f"({PREDICTION_NUM_STRANDS}, genomic_record_length, "
            f"{PREDICTION_NUM_CLASSES})"
        )
        output_h5.attrs["strand_axis"] = "0=positive;1=negative"
        output_h5.attrs["probability_dtype"] = PREDICTION_PROBABILITY_DTYPE
        output_h5.attrs["prediction_schema"] = PREDICTION_LABEL_SCHEMA
        output_h5.attrs["num_strands"] = PREDICTION_NUM_STRANDS
        output_h5.attrs["num_classes_per_strand"] = PREDICTION_NUM_CLASSES
        output_h5.attrs["model_output_channels"] = PREDICTION_NUM_CHANNELS
        output_h5.attrs["probability_normalization"] = PREDICTION_NORMALIZATION
        output_h5.attrs["label_direction"] = PREDICTION_LABEL_DIRECTION
        output_h5.attrs["model_channel_layout"] = PREDICTION_MODEL_CHANNEL_LAYOUT
        output_h5.attrs["full_probability_class_order"] = LABEL_CLASS_ORDER_STRING
        output_h5.attrs["label_mapping"] = LABEL_MAPPING_STRING
        output_h5.attrs["gap_policy"] = "no_gaps_allowed"
        output_h5.attrs["num_chromosomes"] = len(manifest)
        output_h5.attrs["expected_num_windows"] = sum(
            info.num_windows for info in manifest
        )
        output_h5.attrs["hdf5_chunk_bp"] = int(hdf5_chunk_bp)

        chromosomes_root = output_h5.create_group("chromosomes")
        for info in manifest:
            chrom_group = chromosomes_root.create_group(info.group_name)
            chrom_group.attrs["chrom_id"] = info.chrom_id
            chrom_group.attrs["chrom_length"] = info.chrom_length
            chrom_group.attrs["chrom_index"] = info.chrom_index
            chrom_group.attrs["expected_num_windows"] = info.num_windows
            chrom_group.attrs["coordinate_system"] = "0-based half-open"
            chrom_group.attrs["strand_axis"] = "0=positive;1=negative"
            chrom_group.attrs["covered_end"] = 0
            chrom_group.attrs["windows_written"] = 0

            chunk_bp = max(1, min(info.chrom_length, int(hdf5_chunk_bp)))
            # Keep an HDF5 chunk bounded to one strand and one genomic block.
            # This supports later strand-specific decoding without ever loading
            # a complete large genomic record into memory.
            dataset = chrom_group.create_dataset(
                "full_probabilities",
                shape=(
                    PREDICTION_NUM_STRANDS,
                    info.chrom_length,
                    PREDICTION_NUM_CLASSES,
                ),
                dtype=np.dtype(PREDICTION_PROBABILITY_DTYPE),
                chunks=(1, chunk_bp, PREDICTION_NUM_CLASSES),
                compression=compression_name,
                compression_opts=(
                    int(compression_level) if compression_name == "gzip" else None
                ),
                shuffle=compression_name == "gzip",
                fillvalue=0.0,
            )
            dataset.attrs["description"] = (
                "Continuous genomic-record-level full 15-state per-strand softmax "
                "distributions written directly from ordered inference windows."
            )
            dataset.attrs["prediction_schema"] = PREDICTION_LABEL_SCHEMA
            dataset.attrs["num_strands"] = PREDICTION_NUM_STRANDS
            dataset.attrs["num_classes_per_strand"] = PREDICTION_NUM_CLASSES
            dataset.attrs["probability_dtype"] = PREDICTION_PROBABILITY_DTYPE
            dataset.attrs["strand_axis"] = "0=positive;1=negative"
            dataset.attrs["class_order"] = LABEL_CLASS_ORDER_STRING
            dataset.attrs["label_mapping"] = LABEL_MAPPING_STRING
            dataset.attrs["probability_normalization"] = PREDICTION_NORMALIZATION
            dataset.attrs["label_direction"] = PREDICTION_LABEL_DIRECTION
            dataset.attrs["probability_policy"] = (
                "Independent 15-class softmax distribution for each "
                "transcript-direction strand, used as strict segmental "
                "weighted-DAG emissions."
            )

        index_group = output_h5.create_group("chromosome_index")
        index_group.create_dataset(
            "chrom_id",
            data=np.asarray([info.chrom_id for info in manifest], dtype=object).astype(
                H5_STRING_DTYPE
            ),
            dtype=H5_STRING_DTYPE,
        )
        index_group.create_dataset(
            "chrom_group",
            data=np.asarray(
                [info.group_name for info in manifest], dtype=object
            ).astype(H5_STRING_DTYPE),
            dtype=H5_STRING_DTYPE,
        )
        index_group.create_dataset(
            "chrom_length",
            data=np.asarray([info.chrom_length for info in manifest], dtype=np.int64),
            dtype=np.int64,
        )
        index_group.create_dataset(
            "chrom_index",
            data=np.asarray([info.chrom_index for info in manifest], dtype=np.int64),
            dtype=np.int64,
        )
        index_group.create_dataset(
            "num_windows",
            data=np.asarray([info.num_windows for info in manifest], dtype=np.int64),
            dtype=np.int64,
        )

    return output_h5_path


class ChromosomePredictionWriter:
    """Stream ordered prediction windows into one chromosome-level HDF5.

    This object must be created and used only by Accelerate global rank zero.
    Chunks and rows are required to arrive in their original deterministic
    extraction order.
    """

    def __init__(self, h5_path: str):
        self.h5_path = os.path.abspath(h5_path)
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(
                f"Initialized chromosome prediction HDF5 not found: {self.h5_path}"
            )

        self._h5 = h5py.File(self.h5_path, "r+")
        try:
            _validate_prediction_schema(
                self._h5,
                expected_status="incomplete",
            )
            _validate_fresh_root_write_counters(self._h5)
            self._manifest, self._manifest_by_index = (
                _read_and_validate_manifest_index(
                    self._h5,
                    require_complete=False,
                )
            )
        except Exception:
            self._h5.close()
            self._h5 = None
            raise

        self._previous_end = {chrom_id: 0 for chrom_id in self._manifest}
        self._previous_window_index = {
            chrom_id: -1 for chrom_id in self._manifest
        }
        self._windows_written = {chrom_id: 0 for chrom_id in self._manifest}
        self._next_global_window_index = 0
        self._next_chunk_number = 1
        self._last_chrom_index = -1
        self._validated_record_ids = set()
        self._finalized = False

    def write_chunk(
        self,
        chunk_number: int,
        metadata: Mapping[str, np.ndarray],
        probabilities: np.ndarray,
    ) -> None:
        """Validate and write one ordered inference chunk."""

        if self._finalized:
            raise RuntimeError("Cannot write predictions after HDF5 finalization.")
        if int(chunk_number) != self._next_chunk_number:
            raise ValueError(
                f"Expected inference chunk {self._next_chunk_number}, got {chunk_number}."
            )

        missing_columns = [
            column for column in REQUIRED_METADATA_COLUMNS if column not in metadata
        ]
        if missing_columns:
            raise ValueError(
                f"Chunk {chunk_number} metadata is missing columns: {missing_columns}"
            )

        probabilities = np.asarray(probabilities)
        if (
            probabilities.ndim != 4
            or probabilities.shape[1] != PREDICTION_NUM_STRANDS
            or probabilities.shape[2] <= 0
            or probabilities.shape[3] != PREDICTION_NUM_CLASSES
        ):
            raise ValueError(
                f"Chunk {chunk_number} probabilities must have shape "
                f"(num_windows, {PREDICTION_NUM_STRANDS}, center_length, "
                f"{PREDICTION_NUM_CLASSES}), got {probabilities.shape}."
            )
        if not np.issubdtype(probabilities.dtype, np.floating):
            raise ValueError(
                f"Chunk {chunk_number} probabilities must be floating point, "
                f"got {probabilities.dtype}."
            )

        num_rows = int(probabilities.shape[0])
        if num_rows <= 0:
            raise ValueError(f"Chunk {chunk_number} contains no prediction rows.")
        for column in REQUIRED_METADATA_COLUMNS:
            if len(metadata[column]) != num_rows:
                raise ValueError(
                    f"Chunk {chunk_number} column {column!r} has "
                    f"{len(metadata[column])} rows, expected {num_rows}."
                )

        expected_local_indices = np.arange(num_rows, dtype=np.int64)
        if not np.array_equal(metadata["chunk_local_index"], expected_local_indices):
            raise ValueError(
                f"Chunk {chunk_number} has non-consecutive chunk_local_index values."
            )
        if np.any(metadata["chunk_id"] != int(chunk_number)):
            raise ValueError(f"Chunk {chunk_number} has inconsistent chunk_id values.")

        expected_global_indices = np.arange(
            self._next_global_window_index,
            self._next_global_window_index + num_rows,
            dtype=np.int64,
        )
        if not np.array_equal(
            metadata["global_window_index"], expected_global_indices
        ):
            raise ValueError(
                f"Chunk {chunk_number} does not continue the global window order "
                f"at index {self._next_global_window_index}."
            )

        center_length = int(probabilities.shape[2])
        try:
            interval_lengths = metadata["center_end"] - metadata["center_start"]
        except TypeError as error:
            raise ValueError(
                f"Chunk {chunk_number} center coordinates must be numeric arrays."
            ) from error
        if np.any(interval_lengths != center_length):
            raise ValueError(
                f"Chunk {chunk_number} center intervals do not match prediction "
                f"length {center_length}."
            )

        # Validate the full chunk against shadow state before modifying the
        # HDF5 file. Only records touched by this chunk are copied, avoiding
        # O(num_chunks * num_records) work for large scaffold collections while
        # preserving chunk-level atomicity for ordinary validation failures.
        next_previous_end: Dict[str, int] = {}
        next_previous_window_index: Dict[str, int] = {}
        next_windows_written: Dict[str, int] = {}
        next_last_chrom_index = self._last_chrom_index
        write_plans = []
        for row_index in range(num_rows):
            chrom_id = str(metadata["chrom_id"][row_index])
            if chrom_id not in self._manifest:
                raise ValueError(
                    f"Chunk {chunk_number} references unknown genomic record {chrom_id!r}."
                )

            info = self._manifest[chrom_id]
            if chrom_id not in next_previous_end:
                next_previous_end[chrom_id] = self._previous_end[chrom_id]
                next_previous_window_index[chrom_id] = (
                    self._previous_window_index[chrom_id]
                )
                next_windows_written[chrom_id] = self._windows_written[chrom_id]
            chrom_length = int(metadata["chrom_length"][row_index])
            chrom_index = int(metadata["chrom_index"][row_index])
            chrom_window_index = int(metadata["chrom_window_index"][row_index])
            center_start = int(metadata["center_start"][row_index])
            center_end = int(metadata["center_end"][row_index])

            if chrom_length != info.chrom_length or chrom_index != info.chrom_index:
                raise ValueError(
                    f"Chunk {chunk_number} metadata disagrees with the manifest "
                    f"for genomic record {chrom_id!r}."
                )
            if chrom_index < next_last_chrom_index:
                raise ValueError(
                    f"Genomic record order moved backwards from index "
                    f"{next_last_chrom_index} to {chrom_index}."
                )
            if chrom_index > next_last_chrom_index:
                if chrom_index != next_last_chrom_index + 1:
                    raise ValueError(
                        f"Genomic record order skipped from index "
                        f"{next_last_chrom_index} to {chrom_index}."
                    )
                if next_last_chrom_index >= 0:
                    previous_info = self._manifest_by_index[next_last_chrom_index]
                    if (
                        next_previous_end.get(
                            previous_info.chrom_id,
                            self._previous_end[previous_info.chrom_id],
                        )
                        != previous_info.chrom_length
                    ):
                        raise ValueError(
                            f"Genomic record {previous_info.chrom_id!r} ended before "
                            "complete coverage when the next record began."
                        )
                next_last_chrom_index = chrom_index
            expected_window_index = next_previous_window_index[chrom_id] + 1
            if chrom_window_index != expected_window_index:
                raise ValueError(
                    f"Genomic record {chrom_id!r} expected window index "
                    f"{expected_window_index}, got {chrom_window_index}."
                )
            if (
                center_start < 0
                or center_start >= chrom_length
                or center_end <= center_start
            ):
                raise ValueError(
                    f"Invalid center interval for {chrom_id!r}: "
                    f"[{center_start}, {center_end}), length={chrom_length}."
                )

            previous_end = next_previous_end[chrom_id]
            clipped_end = min(center_end, chrom_length)
            if center_start > previous_end:
                raise ValueError(
                    f"Coverage gap for {chrom_id!r}: previous end={previous_end}, "
                    f"current start={center_start}."
                )
            if center_start < previous_end:
                if clipped_end <= previous_end or clipped_end != chrom_length:
                    raise ValueError(
                        f"Invalid overlap for {chrom_id!r}: previous end={previous_end}, "
                        f"current interval=[{center_start}, {center_end})."
                    )

            row_probabilities = probabilities[row_index]
            write_start = max(center_start, previous_end)
            write_end = clipped_end
            probability_start = write_start - center_start
            probability_end = probability_start + (write_end - write_start)
            write_plans.append(
                (
                    info,
                    row_probabilities,
                    write_start,
                    write_end,
                    probability_start,
                    probability_end,
                )
            )

            next_previous_end[chrom_id] = max(previous_end, write_end)
            next_previous_window_index[chrom_id] = chrom_window_index
            next_windows_written[chrom_id] += 1

        # Resolve and validate every touched dataset before the first write. A
        # malformed later record therefore cannot leave an earlier record from
        # the same chunk partially written.
        chromosomes_root = self._h5["chromosomes"]
        datasets: Dict[str, h5py.Dataset] = {}
        newly_validated_record_ids = []
        for chrom_id in next_previous_end:
            info = self._manifest[chrom_id]
            first_access = chrom_id not in self._validated_record_ids
            datasets[chrom_id] = get_chromosome_probability_dataset(
                chromosomes_root,
                info,
                require_fresh=first_access,
            )
            if first_access:
                newly_validated_record_ids.append(chrom_id)

        for (
            info,
            row_probabilities,
            write_start,
            write_end,
            probability_start,
            probability_end,
        ) in write_plans:
            dataset = datasets[info.chrom_id]
            dataset[:, write_start:write_end, :] = row_probabilities[
                :, probability_start:probability_end, :
            ]

        self._previous_end.update(next_previous_end)
        self._previous_window_index.update(next_previous_window_index)
        self._windows_written.update(next_windows_written)
        self._validated_record_ids.update(newly_validated_record_ids)
        self._last_chrom_index = next_last_chrom_index
        self._next_global_window_index += num_rows
        self._next_chunk_number += 1
        self._h5.attrs["num_source_chunks"] = int(chunk_number)
        self._h5.attrs["num_source_windows"] = self._next_global_window_index
        for chrom_id in next_previous_end:
            info = self._manifest[chrom_id]
            chrom_group = chromosomes_root[info.group_name]
            chrom_group.attrs["covered_end"] = self._previous_end[chrom_id]
            chrom_group.attrs["windows_written"] = self._windows_written[chrom_id]
        self._h5.flush()

    def finalize(self, expected_num_chunks: int) -> None:
        """Validate complete coverage and mark the direct-write file complete."""

        if self._finalized:
            raise RuntimeError("Chromosome prediction HDF5 is already finalized.")
        observed_chunks = self._next_chunk_number - 1
        if observed_chunks != int(expected_num_chunks):
            raise ValueError(
                f"Expected {expected_num_chunks} inference chunks, wrote {observed_chunks}."
            )

        expected_total_windows = int(self._h5.attrs["expected_num_windows"])
        if self._next_global_window_index != expected_total_windows:
            raise ValueError(
                f"Expected {expected_total_windows} windows, wrote "
                f"{self._next_global_window_index}."
            )

        for chrom_id, info in self._manifest.items():
            if self._previous_end[chrom_id] != info.chrom_length:
                raise ValueError(
                    f"Genomic record {chrom_id!r} is not fully covered: "
                    f"{self._previous_end[chrom_id]} of {info.chrom_length} bases."
                )
            if self._windows_written[chrom_id] != info.num_windows:
                raise ValueError(
                    f"Genomic record {chrom_id!r} expected {info.num_windows} "
                    f"windows, wrote {self._windows_written[chrom_id]}."
                )

        self._h5.attrs["num_source_chunks"] = observed_chunks
        self._h5.attrs["num_source_windows"] = self._next_global_window_index
        self._h5.flush()
        # The completed status is the final commit marker. A process failure
        # before this second flush leaves an explicitly incomplete temporary
        # file that cannot be promoted.
        self._h5.attrs["status"] = "complete"
        self._h5.flush()
        self._finalized = True

    def close(self) -> None:
        if getattr(self, "_h5", None) is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def validate_chromosome_h5_header(h5_path: str) -> None:
    """Validate only the completed root contract of one prediction HDF5."""

    h5_path = os.path.abspath(h5_path)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Chromosome-level HDF5 not found: {h5_path}")
    with h5py.File(h5_path, "r") as h5_file:
        _validate_prediction_schema(h5_file, expected_status="complete")


def load_chromosome_h5_manifest(
    h5_file: h5py.File,
    *,
    chrom_sequence_info: Optional[Mapping[str, Tuple[int, int]]] = None,
    expected_num_chunks: Optional[int] = None,
) -> Dict[str, ChromosomeInfo]:
    """Load a completed canonical manifest without opening record groups."""

    _validate_prediction_schema(h5_file, expected_status="complete")
    manifest, _ = _read_and_validate_manifest_index(
        h5_file,
        require_complete=True,
        chrom_sequence_info=chrom_sequence_info,
    )
    if expected_num_chunks is not None:
        expected_num_chunks = int(expected_num_chunks)
        if expected_num_chunks <= 0:
            raise ValueError(
                f"expected_num_chunks must be positive, got {expected_num_chunks}."
            )
        _require_integer_attribute(
            h5_file,
            "num_source_chunks",
            expected_num_chunks,
            context="Chromosome HDF5 root",
        )
    return manifest


def validate_chromosome_h5_fast(
    h5_path: str,
    chrom_sequence_info: Optional[Mapping[str, Tuple[int, int]]] = None,
    *,
    expected_num_chunks: Optional[int] = None,
) -> None:
    """Validate completed root/index metadata without per-record object scans."""

    h5_path = os.path.abspath(h5_path)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Chromosome-level HDF5 not found: {h5_path}")
    with h5py.File(h5_path, "r") as h5_file:
        load_chromosome_h5_manifest(
            h5_file,
            chrom_sequence_info=chrom_sequence_info,
            expected_num_chunks=expected_num_chunks,
        )


def validate_chromosome_h5(
    h5_path: str,
    chrom_sequence_info: Optional[Mapping[str, Tuple[int, int]]] = None,
) -> None:
    """Strictly validate a completed 15-state chromosome prediction HDF5.

    Validation is intentionally metadata-only: it checks the schema, manifest,
    completion counters, and every probability dataset's shape and dtype without
    reading the genomic probability payload. Probabilities come directly from
    the pipeline's fixed per-strand softmax, so rescanning the full file would
    duplicate inference work and make validation scale with genome size.
    """

    h5_path = os.path.abspath(h5_path)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Chromosome-level HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5_file:
        hdf5_chunk_bp = _validate_prediction_schema(
            h5_file,
            expected_status="complete",
        )
        _read_and_validate_manifest(
            h5_file,
            hdf5_chunk_bp=hdf5_chunk_bp,
            require_complete=True,
            chrom_sequence_info=chrom_sequence_info,
        )

def promote_chromosome_h5(
    temporary_h5_path: str,
    output_h5_path: str,
    chrom_sequence_info: Mapping[str, Tuple[int, int]],
    *,
    expected_num_chunks: Optional[int] = None,
) -> str:
    """Fast-validate commit metadata and atomically promote a completed HDF5."""

    temporary_h5_path = os.path.abspath(temporary_h5_path)
    output_h5_path = os.path.abspath(output_h5_path)
    validate_chromosome_h5_fast(
        temporary_h5_path,
        chrom_sequence_info,
        expected_num_chunks=expected_num_chunks,
    )
    os.replace(temporary_h5_path, output_h5_path)
    return output_h5_path
