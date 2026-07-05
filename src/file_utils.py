import logging
import os
from typing import List, Tuple, Dict, Optional
import pyfaidx

logger = logging.getLogger("PlantGeneAnn.src.file_utils")


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

    row = [
        str(global_window_index),
        clean_chrom_id,
        str(chrom_length),
        str(chrom_index),
        str(chrom_window_index),
        str(center_start),
        str(center_end),
        str(chunk_id),
        str(chunk_local_index),
        clean_sequence,
    ]
    return "\t".join(row)


def _filter_chromosomes(
    chromosomes: List[Tuple[str, int]],
    min_length: int,
    exclude_patterns: Optional[List[str]],
    include_patterns: Optional[List[str]],
) -> List[str]:
    """Filter genomic records by length and identifier patterns.

    Shared implementation used by :func:`get_valid_chromosomes`.
    """
    if exclude_patterns is None:
        exclude_patterns = ["random", "Un", "alt", "hap", "scaffold"]
    if include_patterns is None:
        include_patterns = []

    filtered: List[str] = []

    for chrom_name, chrom_length in chromosomes:
        if chrom_length < min_length:
            logger.debug(
                "Excluding %s (length: %d < %d)", chrom_name, chrom_length, min_length
            )
            continue

        if include_patterns:
            if not any(pattern in chrom_name for pattern in include_patterns):
                logger.debug(
                    "Excluding %s (does not match include patterns)", chrom_name
                )
                continue

        if any(pattern in chrom_name for pattern in exclude_patterns):
            logger.debug("Excluding %s (matches exclusion pattern)", chrom_name)
            continue

        filtered.append(chrom_name)

    return filtered


class FastaManager:
    """Manage FASTA file access using pyfaidx for efficient random access."""

    def __init__(self, fasta_file: str):
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

        self.fasta_file = fasta_file
        self._faidx: Optional[pyfaidx.Fasta] = None

    @property
    def faidx(self) -> pyfaidx.Fasta:
        """Lazy loading of FASTA index."""
        if self._faidx is None:
            try:
                self._faidx = pyfaidx.Fasta(self.fasta_file, one_based_attributes=False)
                logger.debug("Loaded FASTA index for %s", self.fasta_file)
            except Exception as e:
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_valid_chromosomes(
    fasta_file: str,
    min_length: int = 1000000,
    exclude_patterns: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Get valid genomic-record IDs from a FASTA file using pyfaidx.

    Returns:
        List of valid genomic-record identifiers that meet the filtering criteria.
    """
    if exclude_patterns is None:
        exclude_patterns = ["random", "Un", "alt", "hap", "scaffold"]

    logger.debug("Processing FASTA file: %s", fasta_file)
    logger.debug("Minimum genomic record length: %d bp", min_length)
    logger.debug("Exclusion patterns: %s", exclude_patterns)
    if include_patterns:
        logger.debug("Inclusion patterns: %s", include_patterns)

    with FastaManager(fasta_file) as fasta:
        all_chromosomes = fasta.get_chromosomes()

    valid_chromosomes = _filter_chromosomes(
        all_chromosomes, min_length, exclude_patterns, include_patterns
    )

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

    HEADER = [
        "global_window_index",
        "chrom_id",
        "chrom_length",
        "chrom_index",
        "chrom_window_index",
        "center_start",
        "center_end",
        "chunk_id",
        "chunk_local_index",
        "sequence",
    ]

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
