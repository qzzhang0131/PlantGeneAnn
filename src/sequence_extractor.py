import logging
import os
from typing import List, Tuple

from .file_utils import (
    ChunkTSVWriter,
    FastaManager,
    get_valid_chromosomes,
)

logger = logging.getLogger("PlantGeneAnn.src.sequence_extractor")


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
    of truth for the later two-stage HDF5 reconstruction workflow:

        window-level H5 -> genomic-record-level H5 -> GFF3/BigWig.
    """

    def __init__(self, config):
        self.config = config
        self.flank_length, self.center_length = self._resolve_v2_window_geometry()

    def _resolve_v2_window_geometry(self) -> Tuple[int, int]:
        """Resolve and validate v2 input/context geometry from PipelineConfig.

        ``run_annotator.py`` passes ``--sliding_window_size`` as
        ``config.sequence_length`` and ``--flank_window_size`` as
        ``config.flank_length``. For the default v2 model these are 40,960 and
        5,120 respectively, so the effective model output length is:

            40,960 - 2 * 5,120 = 30,720 bp

        Returns:
            Tuple ``(flank_length, center_length)`` where
            ``center_length = sequence_length - 2 * flank_length``.
        """
        sequence_length = int(self.config.sequence_length)

        if not hasattr(self.config, "flank_length"):
            raise AttributeError(
                "PipelineConfig must define flank_length for PlantGeneAnn v2 "
                "center-crop extraction. Please pass --flank_window_size via "
                "run_annotator.py."
            )

        flank_length = int(self.config.flank_length)

        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}.")
        if flank_length < 0:
            raise ValueError(f"flank_length must be non-negative, got {flank_length}.")
        if 2 * flank_length >= sequence_length:
            raise ValueError(
                "Invalid PlantGeneAnn v2 window geometry: 2 * flank_length must "
                "be smaller than sequence_length so that a non-empty center "
                f"prediction region remains. Got sequence_length={sequence_length}, "
                f"flank_length={flank_length}."
            )

        center_length = sequence_length - 2 * flank_length

        return flank_length, center_length

    @staticmethod
    def _pad_window(sequence: str, left_pad: int, right_pad: int, target_length: int) -> str:
        """Pad a sequence window with ``N`` to exactly ``target_length`` bases."""
        return _pad_window(sequence, left_pad, right_pad, target_length)

    def _slice_single_chromosome(
        self,
        chrom_id: str,
        chrom_index: int,
        chrom_seq: str,
    ) -> List[Tuple[str, int, int, int, int, int, str]]:
        """Tile one genomic record using full-length v2 center-output windows.

        Delegates to the shared standalone slicing helper. See
        ``_slice_single_chromosome_standalone`` for details.
        """
        return _slice_single_chromosome_standalone(
            chrom_id=chrom_id,
            chrom_index=chrom_index,
            chrom_seq=chrom_seq,
            sequence_length=int(self.config.sequence_length),
            flank_length=self.flank_length,
            center_length=self.center_length,
        )

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
            self.config.exclude_patterns,
            getattr(self.config, "include_patterns", None),
        )

        if not valid_ids:
            raise ValueError("No valid chromosomes found meeting the criteria")

        chrom_sequence_info = {}
        total_windows = 0

        with FastaManager(self.config.input_fasta) as fasta:
            with ChunkTSVWriter(self.config.cache_path, int(self.config.chunk_size)) as tsv_writer:
                for chrom_index, chrom_id in enumerate(valid_ids):
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
                    total_windows += num_windows

                    del chrom_seq

        num_chunks = tsv_writer.num_chunks
        logger.debug("Successfully divided all windows into %d chunks", num_chunks)

        return chrom_sequence_info, num_chunks
