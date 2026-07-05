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
    H5_STRING_DTYPE,
    INTEGER_METADATA_COLUMNS,
    REQUIRED_METADATA_COLUMNS,
    _decode_h5_string,
)

logger = logging.getLogger("PlantGeneAnn.src.chromosome_writer")


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
        output_h5.attrs["file_format"] = "plantgeneann_chromosome_level_predictions"
        output_h5.attrs["write_mode"] = "direct_streaming"
        output_h5.attrs["status"] = "incomplete"
        output_h5.attrs["coordinate_system"] = "0-based half-open genomic coordinates"
        output_h5.attrs["full_probability_shape"] = "(2, genomic_record_length, num_features)"
        output_h5.attrs["strand_axis"] = "0=positive;1=negative"
        output_h5.attrs["probability_dtype"] = "float16"
        output_h5.attrs["full_probability_class_order"] = (
            "intergenic,CDS-phase0,CDS-phase1,CDS-phase2,Intron"
        )
        output_h5.attrs["label_mapping"] = (
            "0=Intergenic;1=CDS-phase0(phase=0);2=CDS-phase1(phase=1);"
            "3=CDS-phase2(phase=2);4=Intron"
        )
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
            dataset = chrom_group.create_dataset(
                "full_probabilities",
                shape=(2, info.chrom_length, 5),
                dtype=np.float16,
                chunks=(2, chunk_bp, 5),
                compression=compression_name,
                compression_opts=(
                    int(compression_level) if compression_name == "gzip" else None
                ),
                shuffle=compression_name == "gzip",
                fillvalue=0.0,
            )
            dataset.attrs["description"] = (
                "Continuous genomic-record-level full 5-class softmax "
                "distributions written directly from ordered inference windows."
            )
            dataset.attrs["class_order"] = (
                "intergenic,CDS-phase0,CDS-phase1,CDS-phase2,Intron"
            )
            dataset.attrs["probability_policy"] = (
                "Full softmax distribution for HMM emission probabilities."
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
        if self._h5.attrs.get("status") != "incomplete":
            self._h5.close()
            raise ValueError(
                f"Direct-write HDF5 must have status='incomplete': {self.h5_path}"
            )

        index_group = self._h5["chromosome_index"]
        chrom_ids = [_decode_h5_string(v) for v in index_group["chrom_id"][:]]
        group_names = [
            _decode_h5_string(v) for v in index_group["chrom_group"][:]
        ]
        chrom_lengths = [int(v) for v in index_group["chrom_length"][:]]
        chrom_indices = [int(v) for v in index_group["chrom_index"][:]]
        num_windows = [int(v) for v in index_group["num_windows"][:]]

        row_counts = {
            len(chrom_ids),
            len(group_names),
            len(chrom_lengths),
            len(chrom_indices),
            len(num_windows),
        }
        if len(row_counts) != 1:
            self._h5.close()
            raise ValueError("chromosome_index datasets have inconsistent lengths.")

        self._manifest: Dict[str, ChromosomeInfo] = {}
        for values in zip(
            chrom_ids, group_names, chrom_lengths, chrom_indices, num_windows
        ):
            chrom_id, group_name, chrom_length, chrom_index, expected_windows = values
            if chrom_id in self._manifest:
                self._h5.close()
                raise ValueError(f"Duplicate genomic-record ID in HDF5: {chrom_id!r}")
            self._manifest[chrom_id] = ChromosomeInfo(
                chrom_id=chrom_id,
                chrom_length=chrom_length,
                chrom_index=chrom_index,
                num_windows=expected_windows,
                group_name=group_name,
            )

        self._manifest_by_index = {
            info.chrom_index: info for info in self._manifest.values()
        }
        if set(self._manifest_by_index) != set(range(len(self._manifest))):
            self._h5.close()
            raise ValueError(
                "chromosome_index values must be consecutive in FASTA order."
            )

        self._previous_end = {chrom_id: 0 for chrom_id in self._manifest}
        self._previous_window_index = {
            chrom_id: -1 for chrom_id in self._manifest
        }
        self._windows_written = {chrom_id: 0 for chrom_id in self._manifest}
        self._next_global_window_index = 0
        self._next_chunk_number = 1
        self._last_chrom_index = -1
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
        if probabilities.ndim != 4 or probabilities.shape[1] != 2 or probabilities.shape[3] != 5:
            raise ValueError(
                f"Chunk {chunk_number} probabilities must have shape "
                f"(num_windows, 2, center_length, 5), got {probabilities.shape}."
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
        interval_lengths = metadata["center_end"] - metadata["center_start"]
        if np.any(interval_lengths != center_length):
            raise ValueError(
                f"Chunk {chunk_number} center intervals do not match prediction "
                f"length {center_length}."
            )

        chromosomes_root = self._h5["chromosomes"]
        touched_chrom_ids = set()
        for row_index in range(num_rows):
            chrom_id = str(metadata["chrom_id"][row_index])
            if chrom_id not in self._manifest:
                raise ValueError(
                    f"Chunk {chunk_number} references unknown genomic record {chrom_id!r}."
                )

            info = self._manifest[chrom_id]
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
            if chrom_index < self._last_chrom_index:
                raise ValueError(
                    f"Genomic record order moved backwards from index "
                    f"{self._last_chrom_index} to {chrom_index}."
                )
            if chrom_index > self._last_chrom_index:
                if chrom_index != self._last_chrom_index + 1:
                    raise ValueError(
                        f"Genomic record order skipped from index "
                        f"{self._last_chrom_index} to {chrom_index}."
                    )
                if self._last_chrom_index >= 0:
                    previous_info = self._manifest_by_index[self._last_chrom_index]
                    if self._previous_end[previous_info.chrom_id] != previous_info.chrom_length:
                        raise ValueError(
                            f"Genomic record {previous_info.chrom_id!r} ended before "
                            "complete coverage when the next record began."
                        )
                self._last_chrom_index = chrom_index
            expected_window_index = self._previous_window_index[chrom_id] + 1
            if chrom_window_index != expected_window_index:
                raise ValueError(
                    f"Genomic record {chrom_id!r} expected window index "
                    f"{expected_window_index}, got {chrom_window_index}."
                )
            if center_start < 0 or center_start >= chrom_length or center_end <= center_start:
                raise ValueError(
                    f"Invalid center interval for {chrom_id!r}: "
                    f"[{center_start}, {center_end}), length={chrom_length}."
                )

            previous_end = self._previous_end[chrom_id]
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
            if not np.all(np.isfinite(row_probabilities)):
                raise ValueError(
                    f"Chunk {chunk_number} contains NaN or infinite probabilities "
                    f"for {chrom_id!r} window {chrom_window_index}."
                )

            write_start = max(center_start, previous_end)
            write_end = clipped_end
            if write_end > write_start:
                probability_start = write_start - center_start
                probability_end = probability_start + (write_end - write_start)
                dataset = chromosomes_root[info.group_name]["full_probabilities"]
                dataset[:, write_start:write_end, :] = row_probabilities[
                    :, probability_start:probability_end, :
                ]

            self._previous_end[chrom_id] = max(previous_end, write_end)
            self._previous_window_index[chrom_id] = chrom_window_index
            self._windows_written[chrom_id] += 1
            touched_chrom_ids.add(chrom_id)

        self._next_global_window_index += num_rows
        self._next_chunk_number += 1
        self._h5.attrs["num_source_chunks"] = int(chunk_number)
        self._h5.attrs["num_source_windows"] = self._next_global_window_index
        for chrom_id in touched_chrom_ids:
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

        self._h5.attrs["status"] = "complete"
        self._h5.attrs["num_source_chunks"] = observed_chunks
        self._h5.attrs["num_source_windows"] = self._next_global_window_index
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


def validate_chromosome_h5(
    h5_path: str,
    chrom_sequence_info: Optional[Mapping[str, Tuple[int, int]]] = None,
) -> None:
    """Validate a completed direct-write chromosome prediction HDF5."""

    h5_path = os.path.abspath(h5_path)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Chromosome-level HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5_file:
        if h5_file.attrs.get("file_format") != "plantgeneann_chromosome_level_predictions":
            raise RuntimeError(f"Unexpected chromosome HDF5 format: {h5_path}")
        if h5_file.attrs.get("write_mode") != "direct_streaming":
            raise RuntimeError(f"HDF5 was not produced by direct streaming: {h5_path}")
        if h5_file.attrs.get("status") != "complete":
            raise RuntimeError(f"Chromosome HDF5 is incomplete: {h5_path}")
        if "chromosomes" not in h5_file or "chromosome_index" not in h5_file:
            raise RuntimeError("Chromosome HDF5 is missing required groups.")

        index_group = h5_file["chromosome_index"]
        required_index_columns = (
            "chrom_id",
            "chrom_group",
            "chrom_length",
            "chrom_index",
            "num_windows",
        )
        missing = [name for name in required_index_columns if name not in index_group]
        if missing:
            raise RuntimeError(f"chromosome_index is missing datasets: {missing}")

        chrom_ids = [_decode_h5_string(v) for v in index_group["chrom_id"][:]]
        group_names = [
            _decode_h5_string(v) for v in index_group["chrom_group"][:]
        ]
        lengths = [int(v) for v in index_group["chrom_length"][:]]
        indices = [int(v) for v in index_group["chrom_index"][:]]
        window_counts = [int(v) for v in index_group["num_windows"][:]]
        row_counts = {
            len(chrom_ids),
            len(group_names),
            len(lengths),
            len(indices),
            len(window_counts),
        }
        if len(row_counts) != 1:
            raise RuntimeError("chromosome_index datasets have inconsistent lengths.")
        if indices != list(range(len(indices))):
            raise RuntimeError("chromosome_index values are not consecutive FASTA order.")
        if len(set(chrom_ids)) != len(chrom_ids):
            raise RuntimeError("chromosome_index contains duplicate genomic-record IDs.")

        chromosomes_root = h5_file["chromosomes"]
        for chrom_id, group_name, chrom_length, num_windows in zip(
            chrom_ids, group_names, lengths, window_counts
        ):
            if group_name not in chromosomes_root:
                raise RuntimeError(
                    f"Missing chromosome group {group_name!r} for {chrom_id!r}."
                )
            chrom_group = chromosomes_root[group_name]
            if int(chrom_group.attrs.get("covered_end", -1)) != chrom_length:
                raise RuntimeError(f"Genomic record {chrom_id!r} has incomplete coverage.")
            if int(chrom_group.attrs.get("windows_written", -1)) != num_windows:
                raise RuntimeError(
                    f"Genomic record {chrom_id!r} has an incomplete window count."
                )
            if "full_probabilities" not in chrom_group:
                raise RuntimeError(
                    f"Genomic record {chrom_id!r} is missing full_probabilities."
                )
            dataset = chrom_group["full_probabilities"]
            if dataset.shape != (2, chrom_length, 5):
                raise RuntimeError(
                    f"Genomic record {chrom_id!r} has invalid probability shape "
                    f"{dataset.shape}."
                )
            if dataset.dtype != np.dtype(np.float16):
                raise RuntimeError(
                    f"Genomic record {chrom_id!r} has invalid dtype {dataset.dtype}."
                )

        if chrom_sequence_info is not None:
            expected_ids = list(chrom_sequence_info)
            if chrom_ids != expected_ids:
                raise RuntimeError(
                    "Chromosome HDF5 record order differs from the extraction manifest."
                )
            for chrom_id, chrom_length, num_windows in zip(
                chrom_ids, lengths, window_counts
            ):
                expected_length, expected_windows = chrom_sequence_info[chrom_id]
                if (chrom_length, num_windows) != (
                    int(expected_length),
                    int(expected_windows),
                ):
                    raise RuntimeError(
                        f"Chromosome HDF5 manifest mismatch for {chrom_id!r}."
                    )


def promote_chromosome_h5(
    temporary_h5_path: str,
    output_h5_path: str,
    chrom_sequence_info: Mapping[str, Tuple[int, int]],
) -> str:
    """Validate and atomically promote a completed temporary HDF5."""

    temporary_h5_path = os.path.abspath(temporary_h5_path)
    output_h5_path = os.path.abspath(output_h5_path)
    validate_chromosome_h5(temporary_h5_path, chrom_sequence_info)
    os.replace(temporary_h5_path, output_h5_path)
    return output_h5_path
