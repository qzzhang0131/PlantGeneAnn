"""Window planning and strand-aware label/loss-mask generation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import PreprocessConfig
from .gff import Gene, Transcript
from .labels import paint_transcript
from .qc import GeneQc


@dataclass(frozen=True)
class WindowRecord:
    species: str
    chrom_id: str
    start: int
    end: int
    sequence: str
    labels: np.ndarray
    loss_mask: np.ndarray


def planned_starts(
    length: int,
    window: int,
    step: int,
    flank_length: int = 4096,
) -> tuple[int, ...]:
    """Return full-input starts whose center intervals cover the record.

    ``start`` and ``end`` describe the model input interval, so the first
    start is negative when a left context flank is required. Regular windows
    advance their center interval by ``step``. The final terminal window is
    right-aligned to the record, which may make it overlap the preceding
    regular window. Short records still receive one terminal window and are
    padded as needed by :func:`generate_chromosome_windows`.
    """
    length, window, step, flank_length = map(
        int, (length, window, step, flank_length)
    )
    if length <= 0:
        return ()
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive")
    if flank_length < 0 or 2 * flank_length >= window:
        raise ValueError("flank_length must leave a non-empty center interval")

    center_length = window - 2 * flank_length
    # A larger input step would leave a gap between center-output intervals.
    step = min(step, center_length)
    starts: list[int] = []
    center_start = 0
    while center_start + center_length <= length:
        starts.append(center_start - flank_length)
        center_start += step

    terminal_center_start = max(0, length - center_length)
    terminal_start = terminal_center_start - flank_length
    if not starts or starts[-1] != terminal_start:
        starts.append(terminal_start)
    return tuple(starts)


def count_non_atcg(sequence: str) -> int:
    return sum(base not in "ACGT" for base in sequence)


def generate_chromosome_windows(
    species: str,
    chrom_id: str,
    sequence: str,
    genes: tuple[Gene, ...],
    qc_results: tuple[GeneQc, ...],
    cfg: PreprocessConfig,
    starts: tuple[int, ...] | None = None,
) -> tuple[WindowRecord, ...]:
    min_record_length = getattr(cfg.window, "min_record_length", 32768)
    if len(sequence) < min_record_length:
        return ()

    by_gene = {x.gene_id: x for x in qc_results}
    selected: list[tuple[Gene, Transcript, GeneQc]] = []
    masks: list[GeneQc] = []
    for gene in genes:
        result = by_gene.get(gene.id)
        if result is None or result.seqid != chrom_id or not result.transcripts: continue
        if result.mask < 1.0: masks.append(result)
        # Historical QC retains its pre-QC selected transcript even when that
        # transcript receives mask 0. Such genes contribute only an unsupervised
        # interval: painting their labels first can create a false same-strand
        # owner collision before the mask-0 reset runs below.
        if result.mask > 0.0 and result.selected_transcript_id:
            tx = next(x for x in gene.transcripts if x.id == result.selected_transcript_id)
            selected.append((gene, tx, result))
    records = []
    flank_length = getattr(cfg.window, "flank_length", cfg.window.length // 10)
    if starts is None:
        tiling_step = getattr(cfg.window, "tiling_step", None)
        if tiling_step is None:
            tiling_step = cfg.window.step
        starts = planned_starts(
            len(sequence),
            cfg.window.length,
            tiling_step,
            flank_length,
        )
    for start in starts:
        end = start + cfg.window.length
        # ``start``/``end`` are input coordinates and may extend beyond the
        # FASTA record. Synthetic boundary Ns provide model context but are
        # excluded from the sequence-quality threshold below. The quality
        # threshold follows the model objective and is evaluated only on the
        # center/loss interval, not on either context flank.
        real_start, real_end = max(0, start), min(len(sequence), end)
        left_pad, right_pad = real_start - start, end - real_end
        real_sequence = sequence[real_start:real_end]
        center_genomic_start = start + flank_length
        center_genomic_end = end - flank_length
        real_center_start = max(0, center_genomic_start)
        real_center_end = min(len(sequence), center_genomic_end)
        real_center_sequence = sequence[real_center_start:real_center_end]
        if count_non_atcg(real_center_sequence) > cfg.window.max_non_atcg:
            continue
        seq = ("N" * left_pad) + real_sequence + ("N" * right_pad)
        if len(seq) != cfg.window.length:
            raise ValueError(
                f"Internal window padding error for {chrom_id}:{start}-{end}: "
                f"produced {len(seq)} bases, expected {cfg.window.length}."
            )
        labels = np.zeros((2, cfg.window.length), dtype=np.int8)
        loss_mask = np.ones((2, cfg.window.length), dtype=np.float32)
        if left_pad:
            loss_mask[:, :left_pad] = 0.0
        if right_pad:
            loss_mask[:, cfg.window.length-right_pad:] = 0.0
        owners = np.full((2, cfg.window.length), "", dtype=object)
        for gene, tx, result in selected:
            if tx.end <= start or tx.start >= end: continue
            channel = 0 if tx.strand == "+" else 1
            paint_transcript(labels[channel], owners[channel], tx, gene.id, start, end)
        for result in masks:
            if result.mask_end <= start or result.mask_start >= end: continue
            channel = 0 if result.strand == "+" else 1 if result.strand == "-" else None
            lo, hi = max(start, result.mask_start)-start, min(end, result.mask_end)-start
            channels = (channel,) if channel is not None else (0, 1)
            for ch in channels:
                np.minimum(loss_mask[ch, lo:hi], np.float32(result.mask), out=loss_mask[ch, lo:hi])
                if result.mask == 0.0 and cfg.labels.reset_mask0_labels_to_background:
                    labels[ch, lo:hi] = 0
        # Training crops both flanks before computing a separate objective for
        # each strand. Check that objective interval rather than the
        # padded/context positions, which are intentionally zero-weight.
        center_start, center_end = flank_length, cfg.window.length - flank_length
        if np.any(np.all(loss_mask[:, center_start:center_end] == 0, axis=1)):
            continue
        records.append(WindowRecord(species, chrom_id, start, end, seq, labels, loss_mask))
    return tuple(records)
