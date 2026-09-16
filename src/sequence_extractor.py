"""FASTA filtering, genomic window extraction, and chunked TSV output."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import pyfaidx
from tqdm import tqdm

from .constants import CHUNK_TSV_COLUMNS
from .runtime import prepare_cached_fasta_index_path

logger = logging.getLogger("PlantGeneAnn.src.sequence_extractor")

def _format_tsv_row(
    global_window_index: int,
    chrom_id: str,
    chrom_length: int,
    chrom_index: int,
    chrom_window_index: int,
    center_start: int,
    center_end: int,
    chunk_id: int,
    chunk_local_index: int,
    sequence: str,
) -> str:
    """Format one window record as a tab-separated line.

    FASTA sequences should contain only bases, but defensively remove
    newline/tab characters so TSV parsing remains stable.
    """
    clean_chrom_id = str(chrom_id).replace("\n", "").replace("\t", " ")
    clean_sequence = str(sequence).replace("\n", "").replace("\t", "")

    row_values = {
        "global_window_index": str(global_window_index),
        "chrom_id": clean_chrom_id,
        "chrom_length": str(chrom_length),
        "chrom_index": str(chrom_index),
        "chrom_window_index": str(chrom_window_index),
        "center_start": str(center_start),
        "center_end": str(center_end),
        "chunk_id": str(chunk_id),
        "chunk_local_index": str(chunk_local_index),
        "sequence": clean_sequence,
    }
    return "\t".join(row_values[column_name] for column_name in CHUNK_TSV_COLUMNS)


def _filter_chromosomes(
    chromosomes: List[Tuple[str, int]], min_length: int
) -> List[str]:
    """Return every FASTA record meeting the configured length threshold.

    Full-contig annotation intentionally makes no decision based on a record
    identifier. Names such as ``random``, ``Un``, ``alt``, ``hap``, and
    ``scaffold`` are valid genomic records and are processed exactly like a
    primary chromosome when they meet ``min_length``.
    """
    filtered: List[str] = []
    for chrom_name, chrom_length in chromosomes:
        if chrom_length < min_length:
            logger.debug(
                "Skipping %s (length: %d < %d)", chrom_name, chrom_length, min_length
            )
            continue
        filtered.append(chrom_name)
    return filtered


class FastaManager:
    """Manage FASTA file access using pyfaidx for efficient random access."""

    def __init__(self, fasta_file: str, cache_path: Optional[str] = None):
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

        self.fasta_file = os.path.abspath(fasta_file)
        self._temporary_cache: Optional[tempfile.TemporaryDirectory] = None
        if cache_path is None:
            # Preserve the small public helper's backwards-compatible call
            # signature without ever falling back to an input-adjacent .fai.
            self._temporary_cache = tempfile.TemporaryDirectory(
                prefix="plantgeneann_faidx_"
            )
            cache_path = self._temporary_cache.name
        self.index_file = prepare_cached_fasta_index_path(
            self.fasta_file,
            cache_path,
        )
        self._faidx: Optional[pyfaidx.Fasta] = None

    @property
    def faidx(self) -> pyfaidx.Fasta:
        """Lazy loading of FASTA index."""
        if self._faidx is None:
            try:
                self._faidx = pyfaidx.Fasta(
                    self.fasta_file,
                    indexname=self.index_file,
                    one_based_attributes=False,
                )
                logger.debug(
                    "Loaded FASTA index for %s from %s",
                    self.fasta_file,
                    self.index_file,
                )
            except Exception as e:
                self.close()
                raise IOError(f"Failed to load FASTA file {self.fasta_file}: {e}")
        return self._faidx

    def get_chromosomes(self) -> List[Tuple[str, int]]:
        """Get all genomic-record names and lengths **without** loading sequences.

        Reads reference lengths directly from the ``.fai`` index, avoiding
        the memory cost of loading full genomic-record sequences.
        """
        chromosomes: List[Tuple[str, int]] = []
        for chrom_name in self.faidx.keys():
            chrom_length = self.faidx.faidx.index[chrom_name].rlen
            chromosomes.append((chrom_name, chrom_length))

        logger.debug("Found %d sequences in FASTA file", len(chromosomes))
        return chromosomes

    def get_sequence(self, chrom_name: str, start: int = 0, end: Optional[int] = None) -> str:
        """Get sequence for a genomic record or a region within it.

        Args:
            chrom_name: Genomic-record identifier (legacy parameter name).
            start: Start position (0-based).
            end: End position (0-based, exclusive).
        """
        if chrom_name not in self.faidx:
            raise ValueError(f"Chromosome {chrom_name} not found in FASTA file")

        try:
            if end is None:
                end = self.faidx.faidx.index[chrom_name].rlen

            sequence = str(self.faidx[chrom_name][start:end])
            return sequence.upper()

        except Exception as e:
            raise IOError(f"Failed to get sequence for {chrom_name}[{start}:{end}]: {e}")

    def close(self):
        """Close the FASTA index."""
        if self._faidx is not None:
            self._faidx.close()
            self._faidx = None
        if self._temporary_cache is not None:
            self._temporary_cache.cleanup()
            self._temporary_cache = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_valid_chromosomes(
    fasta_file: str,
    min_length: int = 1000000,
    cache_path: Optional[str] = None,
) -> List[str]:
    """Get all length-qualified genomic-record IDs from a FASTA file.

    Record names do not influence inference eligibility. This deliberately
    includes primary chromosomes, scaffolds, unplaced contigs, alternate loci,
    haplotypes, and random records when their sequence length is sufficient.
    """
    logger.debug("Processing FASTA file: %s", fasta_file)
    logger.debug("Minimum genomic record length: %d bp", min_length)

    with FastaManager(fasta_file, cache_path=cache_path) as fasta:
        all_chromosomes = fasta.get_chromosomes()

    valid_chromosomes = _filter_chromosomes(all_chromosomes, min_length)

    if valid_chromosomes:
        valid_set = set(valid_chromosomes)
        valid_lengths = [
            length for name, length in all_chromosomes if name in valid_set
        ]
        total_length = sum(valid_lengths)

        logger.info(
            "Selected %d genomic records (total: %s bp)",
            len(valid_chromosomes),
            f"{total_length:,}",
        )

        logger.debug("Selected genomic records:")
        logged_count = 0
        for chrom_name, chrom_length in all_chromosomes:
            if chrom_name in valid_set:
                if logged_count < 10:
                    logger.debug("  %s: %d bp", chrom_name, chrom_length)
                logged_count += 1
        if logged_count > 10:
            logger.debug("  ... and %d more genomic records", logged_count - 10)

    return valid_chromosomes


class ChunkTSVWriter:
    """Incrementally write window records to chunked TSV files.

    Manages file handles for multiple chunk files and writes each supplied
    window immediately. The caller may still materialize the windows for one
    genomic record, but this writer never accumulates genome-wide windows.
    """

    HEADER = list(CHUNK_TSV_COLUMNS)

    def __init__(self, save_dir: str, chunk_size: int):
        self.save_dir = save_dir
        self.chunk_size = chunk_size
        self.global_window_counter = 0
        self.num_chunks = 0
        self._open_files: Dict[int, object] = {}

    def _get_chunk_id(self, global_window_index: int) -> int:
        return global_window_index // self.chunk_size + 1

    def _ensure_chunk_file(self, chunk_id: int) -> object:
        if chunk_id not in self._open_files:
            os.makedirs(self.save_dir, exist_ok=True)
            output_file = os.path.join(self.save_dir, f"chunk_{chunk_id}.tsv")
            f = open(output_file, "w")
            f.write("\t".join(self.HEADER) + "\n")
            self._open_files[chunk_id] = f
            self.num_chunks = max(self.num_chunks, chunk_id)
        return self._open_files[chunk_id]

    def write_window(
        self,
        chrom_id: str,
        chrom_length: int,
        chrom_index: int,
        chrom_window_index: int,
        center_start: int,
        center_end: int,
        sequence: str,
    ):
        """Write one window record to the appropriate chunk TSV file."""
        global_window_index = self.global_window_counter
        chunk_id = self._get_chunk_id(global_window_index)
        chunk_local_index = global_window_index % self.chunk_size

        f = self._ensure_chunk_file(chunk_id)

        line = _format_tsv_row(
            global_window_index=global_window_index,
            chrom_id=chrom_id,
            chrom_length=chrom_length,
            chrom_index=chrom_index,
            chrom_window_index=chrom_window_index,
            center_start=center_start,
            center_end=center_end,
            chunk_id=chunk_id,
            chunk_local_index=chunk_local_index,
            sequence=sequence,
        )
        f.write(line + "\n")

        self.global_window_counter += 1

    def close(self):
        """Close all open chunk TSV files."""
        for f in self._open_files.values():
            f.close()
        self._open_files.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _pad_window(sequence: str, left_pad: int, right_pad: int, target_length: int) -> str:
    """Pad a sequence window with ``N`` to exactly ``target_length`` bases.

    This pure helper accepts all required values explicitly and does not depend
    on a ``SequenceExtractor`` instance.
    """
    if left_pad < 0 or right_pad < 0:
        raise ValueError(
            f"Padding lengths must be non-negative, got left={left_pad}, right={right_pad}."
        )

    padded_sequence = ("N" * left_pad) + sequence + ("N" * right_pad)

    if len(padded_sequence) < target_length:
        padded_sequence += "N" * (target_length - len(padded_sequence))

    if len(padded_sequence) != target_length:
        raise ValueError(
            "Internal slicing error: padded sequence length does not match "
            f"target length ({len(padded_sequence)} != {target_length})."
        )

    return padded_sequence


def _slice_single_chromosome_standalone(
    chrom_id: str,
    chrom_index: int,
    chrom_seq: str,
    sequence_length: int,
    flank_length: int,
    center_length: int,
) -> List[Tuple[str, int, int, int, int, int, str]]:
    """Tile one genomic record using full-length v2 center-output windows.

    The helper accepts explicit geometry parameters instead of reading from a
    config object, which keeps the slicing behavior independently testable.

    Returns:
        List of tuples:
            (
                chrom_id,
                chrom_length,
                chrom_index,
                chrom_window_index,
                center_start,
                center_end,
                sequence,
            )
    """
    valid_sequences = []
    chrom_length = len(chrom_seq)

    def append_window(window_center_start: int, window_center_end: int) -> None:
        if window_center_end - window_center_start != center_length:
            raise ValueError(
                "Internal slicing error: center-output interval length "
                f"{window_center_end - window_center_start} does not match "
                f"center_length={center_length}."
            )

        requested_start = window_center_start - flank_length
        requested_end = window_center_end + flank_length

        fetch_start = max(0, requested_start)
        fetch_end = min(chrom_length, requested_end)

        left_pad = fetch_start - requested_start
        right_pad = requested_end - fetch_end

        raw_window = chrom_seq[fetch_start:fetch_end]
        sequence = _pad_window(
            raw_window,
            left_pad=left_pad,
            right_pad=right_pad,
            target_length=sequence_length,
        )

        valid_sequences.append(
            (
                chrom_id,
                chrom_length,
                chrom_index,
                len(valid_sequences),
                window_center_start,
                window_center_end,
                sequence,
            )
        )

    center_start = 0
    while center_start + center_length <= chrom_length:
        append_window(center_start, center_start + center_length)
        center_start += center_length

    if center_start < chrom_length:
        terminal_center_start = max(0, chrom_length - center_length)
        terminal_center_end = terminal_center_start + center_length
        append_window(terminal_center_start, terminal_center_end)

    return valid_sequences


class SequenceExtractor:
    """Extract fixed-length model input windows from genome sequences.

    PlantGeneAnn v2 is a center-crop segmentation model: each input window has
    length ``sequence_length``, but only the central part is emitted as logits.
    The left and right flanks are context used to improve boundary predictions.

    Therefore, the correct genome tiling strategy is no longer the old
    overlapping-window strategy where input and output are assumed to have the
    same length. Instead, we tile each genomic record by the model's *effective
    output region* and add flanking context around every tile:

        input window:
            [left context][center output region][right context]
            |<- flank ->|<-- center output -->|<- flank ->|

    For genomic-record starts/ends, missing context is padded with ``N`` so that the
    first and last genomic bases still fall into the model's central output
    region and can be predicted.

    Important coordinate convention:
        Window metadata store the genomic center-output interval using 0-based
        half-open coordinates: ``center_start`` is inclusive and ``center_end``
        is exclusive. The ``sequence`` field always has exactly
        ``config.sequence_length`` bases and may include N-padding at genomic-record
        boundaries.

    The chunk TSV files intentionally retain complete per-window metadata
    (genomic-record ID, record-local order, genomic coordinates,
    global order, and chunk-local order). These metadata columns are the source
    of truth for direct distributed writing into chromosome-level prediction
    HDF5 files, which are then decoded into GFF3.
    """

    def __init__(self, config):
        self.config = config
        self.flank_length, self.center_length = self._resolve_v2_window_geometry()

    def _resolve_v2_window_geometry(self) -> Tuple[int, int]:
        """Resolve the configured input/context geometry for window extraction.

        The CLI prediction pipelines use the fixed 40,960-bp model input and
        5,120-bp flank per side defined in ``src.configuration``, yielding a
        30,720-bp genomic center interval. Explicit ``PipelineConfig`` instances
        remain validated here so Python API callers cannot supply invalid
        geometry.

        Returns:
            Tuple ``(flank_length, center_length)`` where
            ``center_length = sequence_length - 2 * flank_length``.
        """
        sequence_length = int(self.config.sequence_length)
        flank_length = int(self.config.flank_length)
        center_length = sequence_length - 2 * flank_length

        if sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be positive, got {sequence_length}."
            )
        if flank_length < 0:
            raise ValueError(
                f"flank_length must be non-negative, got {flank_length}."
            )
        if center_length <= 0:
            raise ValueError(
                "Invalid window geometry: 2 * flank_length must be smaller "
                f"than sequence_length; got sequence_length={sequence_length}, "
                f"flank_length={flank_length}."
            )

        return flank_length, center_length

    def _slice_and_write_chromosome(
        self,
        chrom_id: str,
        chrom_index: int,
        chrom_seq: str,
        tsv_writer: ChunkTSVWriter,
    ) -> Tuple[int, int]:
        """Slice one genomic record and write its windows to chunk TSV files.

        Window tuples for the current record are materialized before writing;
        windows from the entire genome are never accumulated together.

        Returns:
            Tuple of (record_length, num_windows) for this genomic record.
        """
        chrom_length = len(chrom_seq)
        valid_sequences = _slice_single_chromosome_standalone(
            chrom_id=chrom_id,
            chrom_index=chrom_index,
            chrom_seq=chrom_seq,
            sequence_length=int(self.config.sequence_length),
            flank_length=self.flank_length,
            center_length=self.center_length,
        )

        for window in valid_sequences:
            (
                _,
                w_chrom_length,
                _,
                chrom_window_index,
                center_start,
                center_end,
                sequence,
            ) = window
            tsv_writer.write_window(
                chrom_id=chrom_id,
                chrom_length=w_chrom_length,
                chrom_index=chrom_index,
                chrom_window_index=chrom_window_index,
                center_start=center_start,
                center_end=center_end,
                sequence=sequence,
            )

        return chrom_length, len(valid_sequences)

    def process(self):
        """Execute the sequence extraction pipeline.

        Processes genomic records one at a time to avoid loading the entire
        genome or all genome-wide windows into memory. Windows for the current
        record are materialized and then written to chunked TSV files.
        """
        valid_ids = get_valid_chromosomes(
            self.config.input_fasta,
            self.config.min_chrom_length,
            cache_path=self.config.cache_path,
        )

        if not valid_ids:
            raise ValueError("No valid chromosomes found meeting the criteria")

        chrom_sequence_info = {}
        with FastaManager(
            self.config.input_fasta,
            cache_path=self.config.cache_path,
        ) as fasta:
            with ChunkTSVWriter(self.config.cache_path, int(self.config.chunk_size)) as tsv_writer:
                pbar = tqdm(
                    enumerate(valid_ids),
                    total=len(valid_ids),
                    desc="Extracting windows",
                    unit="record",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )
                for chrom_index, chrom_id in pbar:
                    if chrom_id not in fasta.faidx:
                        logger.warning("Genomic record %s not found in FASTA file, skipping.", chrom_id)
                        continue

                    chrom_seq = fasta.get_sequence(chrom_id)
                    if len(chrom_seq) == 0:
                        logger.warning("Genomic record %s is empty, skipping.", chrom_id)
                        continue

                    chrom_length, num_windows = self._slice_and_write_chromosome(
                        chrom_id=chrom_id,
                        chrom_index=chrom_index,
                        chrom_seq=chrom_seq,
                        tsv_writer=tsv_writer,
                    )
                    chrom_sequence_info[chrom_id] = (chrom_length, num_windows)
                    
                    # Update progress bar with current chromosome info
                    pbar.set_postfix_str(f"{chrom_id}: {num_windows} windows")

                    del chrom_seq
                
                pbar.close()

        num_chunks = tsv_writer.num_chunks
        logger.debug("Successfully divided all windows into %d chunks", num_chunks)

        return chrom_sequence_info, num_chunks
