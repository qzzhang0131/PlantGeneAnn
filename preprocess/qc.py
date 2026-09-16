"""Transcript-level ORF QC and deterministic gene-level transcript selection."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Optional
from .config import QcConfig
from .fasta import reverse_complement
from .gff import Feature, Gene, Transcript, is_protein_coding
from src.constants import ALLOWED_SPLICE_PAIRS

DNA = set("ACGT")


@dataclass(frozen=True)
class TranscriptQc:
    gene_id: str
    transcript_id: str
    seqid: str
    strand: str
    start: int
    end: int
    cds_length: int
    cds_blocks: int
    mask: float
    zero_reasons: tuple[str, ...]
    half_reasons: tuple[str, ...]
    notes: tuple[str, ...]
    coding_evidence: str
    selected: bool = False


@dataclass(frozen=True)
class GeneQc:
    gene_id: str
    selected_transcript_id: Optional[str]
    mask: float
    mask_start: int
    mask_end: int
    seqid: str
    strand: str
    transcripts: tuple[TranscriptQc, ...]


def transcript_order(features: list[Feature], strand: str) -> list[Feature]:
    return sorted(features, key=lambda f: (f.start, f.end), reverse=strand == "-")


def cds_sequence(tx: Transcript, genome: dict[str, str]) -> str:
    chrom = genome.get(tx.seqid)
    if chrom is None: return ""
    parts = []
    for feature in transcript_order(tx.cds, tx.strand):
        if feature.start < 0 or feature.end > len(chrom): return ""
        part = chrom[feature.start:feature.end]
        parts.append(reverse_complement(part) if tx.strand == "-" else part)
    return "".join(parts).upper()


def inferred_introns(tx: Transcript, mode: str) -> list[tuple[int, int]]:
    if mode == "gff": return [(x.start, x.end) for x in tx.introns]
    blocks = tx.exons if mode == "exon" and len(tx.exons) >= 2 else tx.cds
    if mode == "auto":
        if len(tx.exons) >= 2: blocks = tx.exons
        elif tx.introns: return [(x.start, x.end) for x in tx.introns]
    blocks = sorted(blocks, key=lambda x: (x.start, x.end))
    return [(left.end, right.start) for left, right in zip(blocks, blocks[1:])]


def _marker(attrs: dict[str, str], key: str) -> bool:
    return attrs.get(key, "").lower() == "true"


def _unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _is_protein_coding_gene(gene: Gene, cfg: QcConfig) -> bool:
    values = {value.lower() for value in cfg.protein_coding_values}
    attrs = gene.attrs
    biotype = (
        attrs.get(cfg.gene_biotype_key)
        or attrs.get("biotype")
        or attrs.get("gene_type")
        or attrs.get("gbkey", "")
    ).lower()
    if biotype in values or attrs.get("gene_biotype", "").lower() in values:
        return True
    # Historical fallback for filtered GFFs whose gene feature lost biotype.
    return any(tx.cds for tx in gene.transcripts) and attrs.get("gbkey", "").lower() in {"", "gene"}


def _choose_transcript(gene: Gene, policy: str) -> tuple[Optional[Transcript], list[str], str]:
    notes: list[str] = []
    transcripts = [tx for tx in gene.transcripts if tx.cds] or list(gene.transcripts)
    if not transcripts:
        return None, notes, "none"
    if len(transcripts) == 1:
        return transcripts[0], notes, "single"
    notes.append(f"multiple_transcripts={len(transcripts)}")
    if policy == "fail":
        return transcripts[0], notes, "fail"
    if policy == "first":
        return sorted(transcripts, key=lambda tx: (tx.start, tx.end, tx.id))[0], notes, "first"
    if policy == "phytozome_longest":
        marked = [tx for tx in transcripts if tx.attrs.get("longest", "").strip() == "1"]
        if len(marked) == 1:
            notes.append("selected_phytozome_longest")
            return marked[0], notes, "phytozome_longest"
        if marked:
            notes.append(f"multiple_phytozome_longest={len(marked)}")
            transcripts = marked
        else:
            notes.append("phytozome_longest_not_marked_fallback_longest_cds")
    selected = max(
        transcripts,
        key=lambda tx: (
            sum(feature.length for feature in tx.cds),
            len(tx.cds),
            -tx.start,
            tx.id,
        ),
    )
    notes.append("selected_longest_cds")
    return selected, notes, "longest_cds"


def qc_transcript(
    gene: Gene,
    tx: Transcript,
    genome: dict[str, str],
    cfg: QcConfig,
    evidence: str,
    initial_notes: tuple[str, ...] = (),
) -> TranscriptQc:
    """Apply the historical ``qc_gene_loss_mask.py`` checks to one selected transcript."""
    zero: list[str] = []
    half: list[str] = []
    notes = list(initial_notes)
    attrs = [gene.attrs, tx.attrs] + [feature.attrs for feature in tx.children]

    partial = any(_marker(value, "partial") or "start_range" in value or "end_range" in value for value in attrs)
    if partial:
        _unique(notes if cfg.allow_partial else zero, "partial_marker_present" if cfg.allow_partial else "partial_cds_or_gene")
    exception = any("exception" in value or "transl_except" in value for value in attrs)
    if exception:
        _unique(notes if cfg.allow_annotation_exceptions else zero, "annotation_exception_present" if cfg.allow_annotation_exceptions else "annotation_exception_or_transl_except")
    if any("LOW QUALITY PROTEIN" in value.get("product", "").upper() for value in attrs):
        target = zero if cfg.low_quality_policy == "hard" else half if cfg.low_quality_policy == "half" else notes
        _unique(target, "low_quality_protein")
    if any(_marker(value, "pseudo") for value in attrs):
        _unique(zero, "pseudo_marker")

    if not tx.cds:
        _unique(zero, "no_CDS")
    if tx.seqid != gene.seqid:
        _unique(zero, "transcript_seqid_mismatch")
    if tx.strand not in {"+", "-"}:
        _unique(zero, "invalid_strand")

    for name, blocks in (("CDS", tx.cds), ("exon", tx.exons)):
        ordered = sorted(blocks, key=lambda feature: (feature.start, feature.end))
        if any(feature.start < 0 or feature.end <= feature.start for feature in ordered):
            _unique(zero, f"invalid_{name}_coordinates")
        if any(right.start < left.end for left, right in zip(ordered, ordered[1:])):
            _unique(zero, f"{name}_blocks_overlap")
    ordered_cds = transcript_order(tx.cds, tx.strand)
    if len(ordered_cds) > 2 and any(feature.length <= 2 for feature in ordered_cds[1:-1]):
        _unique(zero, "middle_CDS_length_le_2")

    cumulative = 0
    mismatches = 0
    for index, feature in enumerate(ordered_cds):
        if feature.phase is None:
            _unique(zero, "phase_missing_or_invalid")
            break
        expected = 0 if index == 0 else (3 - cumulative % 3) % 3
        mismatches += int(feature.phase != expected)
        cumulative += feature.length
    if mismatches:
        _unique(notes, "phase_mismatch_repaired")

    sequence = cds_sequence(tx, genome) if tx.cds else ""
    if tx.cds:
        if tx.seqid not in genome:
            _unique(zero, "cds_seqid_not_in_fasta")
        elif any(feature.start < 0 or feature.end > len(genome[tx.seqid]) for feature in tx.cds):
            _unique(zero, "cds_coordinates_out_of_bounds")
        if len(sequence) < cfg.min_cds_length:
            _unique(zero, "CDS_length_less_than_min")
        if len(sequence) % 3:
            _unique(zero, "CDS_length_not_multiple_of_3")
        if any(base not in DNA for base in sequence):
            _unique(zero, "CDS_contains_ambiguous_base")
        if len(sequence) < 3:
            _unique(zero, "missing_start_codon")
            _unique(zero, "missing_stop_codon")
        else:
            start, stop = sequence[:3], sequence[-3:]
            if any(base not in DNA for base in start):
                _unique(zero, "start_codon_contains_ambiguous_base")
            elif start not in cfg.start_codons:
                _unique(zero, "noncanonical_start_codon")
            if any(base not in DNA for base in stop):
                _unique(zero, "stop_codon_contains_ambiguous_base")
            elif stop not in cfg.stop_codons:
                _unique(zero, "missing_or_noncanonical_stop_codon")
            if len(sequence) % 3 == 0:
                codons = [sequence[i:i + 3] for i in range(0, len(sequence) - 2, 3)]
                internal = codons[:-1] if codons and codons[-1] in cfg.stop_codons else codons
                if any(codon in cfg.stop_codons for codon in internal):
                    _unique(zero, "internal_stop")

    introns = inferred_introns(tx, cfg.intron_source)
    source = cfg.intron_source
    if cfg.intron_source == "auto":
        source = "exon" if len(tx.exons) >= 2 else "gff" if tx.introns else "cds" if len(tx.cds) >= 2 else "none"
    if source != "none":
        _unique(notes, f"intron_source={source}")
    chrom = genome.get(tx.seqid)
    for start, end in introns:
        length = end - start
        if length <= 0:
            _unique(zero, "exon_or_CDS_blocks_overlap_or_touch")
            continue
        if length < cfg.hard_min_intron_length:
            _unique(zero, f"intron_length_lt_{cfg.hard_min_intron_length}")
        elif length < cfg.soft_min_intron_length:
            _unique(half, f"intron_length_lt_{cfg.soft_min_intron_length}")
        if chrom is None:
            _unique(zero, "intron_seqid_not_in_fasta")
            continue
        if start < 0 or end > len(chrom):
            _unique(zero, "intron_coordinates_out_of_bounds")
            continue
        intron = chrom[start:end].upper()
        if tx.strand == "-":
            intron = reverse_complement(intron)
        if any(base not in DNA for base in intron):
            _unique(half, "intronic_ambiguous_base")
        if len(intron) < 2:
            _unique(zero, "splice_motif_missing")
        else:
            donor, acceptor = intron[:2], intron[-2:]
            donor_is_unambiguous = all(base in DNA for base in donor)
            acceptor_is_unambiguous = all(base in DNA for base in acceptor)
            if not donor_is_unambiguous:
                _unique(zero, "splice_donor_contains_ambiguous_base")
            if not acceptor_is_unambiguous:
                _unique(zero, "splice_acceptor_contains_ambiguous_base")
            if (
                donor_is_unambiguous
                and acceptor_is_unambiguous
                and (donor, acceptor) not in ALLOWED_SPLICE_PAIRS
            ):
                _unique(zero, "splice_pair_not_supported_by_decoder")

    mask = 0.0 if zero else 0.5 if half else 1.0
    # The historical report retains the transcript chosen before QC even when
    # that transcript receives mask 0. Window masking later removes supervision.
    return TranscriptQc(gene.id, tx.id, tx.seqid, tx.strand, tx.start, tx.end, len(sequence), len(tx.cds), mask, tuple(zero), tuple(half), tuple(notes), evidence, True)


def qc_gene(gene: Gene, genome: dict[str, str], cfg: QcConfig) -> GeneQc:
    """Select first, then QC exactly one transcript as in the historical script."""
    if not _is_protein_coding_gene(gene, cfg):
        return GeneQc(gene.id, None, 1.0, gene.start, gene.end, gene.seqid, gene.strand, ())
    selected, notes, policy = _choose_transcript(gene, cfg.multi_transcript)
    if selected is None:
        missing = TranscriptQc(gene.id, "", gene.seqid, gene.strand, gene.start, gene.end, 0, 0, 0.0, ("no_transcript",), (), tuple(notes), "gene_biotype", False)
        return GeneQc(gene.id, None, 0.0, gene.start, gene.end, gene.seqid, gene.strand, (missing,))
    if cfg.multi_transcript == "fail" and len([tx for tx in gene.transcripts if tx.cds]) > 1:
        notes.append("multiple_transcripts")
    result = qc_transcript(gene, selected, genome, cfg, policy, tuple(notes))
    if cfg.multi_transcript == "fail" and "multiple_transcripts" in notes:
        result = replace(result, mask=0.0, zero_reasons=(*result.zero_reasons, "multiple_transcripts"), selected=False)
    return GeneQc(gene.id, selected.id, result.mask, result.start, result.end, result.seqid, result.strand, (result,))


_QC_GENOME = None
_QC_CONFIG = None


def _init_qc_worker(genome: dict[str, str], cfg: QcConfig) -> None:
    global _QC_GENOME, _QC_CONFIG
    _QC_GENOME = genome
    _QC_CONFIG = cfg


def _qc_gene_worker(gene: Gene) -> GeneQc:
    return qc_gene(gene, _QC_GENOME, _QC_CONFIG)


def mask_same_strand_coding_overlaps(
    genes: tuple[Gene, ...],
    results: tuple[GeneQc, ...],
) -> tuple[GeneQc, ...]:
    """Mask selected genes whose non-background states overlap on one strand.

    One label channel cannot represent two simultaneous gene-state paths. Such
    overlaps occur even in curated RefSeq annotations, so failing the species is
    not useful. Instead, mark every involved gene as unsupervised and remove its
    selected transcript before window labeling. Opposite-strand overlaps remain
    valid because they occupy separate channels.
    """

    # Import locally to avoid making transcript QC depend on NumPy at module
    # import time. This is the same state generator used by window painting, so
    # overlap screening and painting have exactly identical genic semantics.
    from .labels import transcript_labels

    gene_by_id = {gene.id: gene for gene in genes}
    selected = []
    for index, result in enumerate(results):
        if result.mask == 0.0 or result.selected_transcript_id is None:
            continue
        gene = gene_by_id[result.gene_id]
        transcript = next(
            tx for tx in gene.transcripts
            if tx.id == result.selected_transcript_id
        )
        selected.append((index, transcript))

    conflicts: set[int] = set()
    grouped = defaultdict(list)
    for index, transcript in selected:
        grouped[(transcript.seqid, transcript.strand)].append(
            (index, transcript, transcript_labels(transcript))
        )
    for group in grouped.values():
        group.sort(key=lambda item: (item[1].start, item[1].end, item[1].id))
        active = []
        for index, transcript, labels in group:
            active = [item for item in active if item[1].end > transcript.start]
            for other_index, other, other_labels in active:
                overlap_start = max(transcript.start, other.start)
                overlap_end = min(transcript.end, other.end)
                if overlap_start >= overlap_end:
                    continue
                current_genic = labels[
                    overlap_start - transcript.start : overlap_end - transcript.start
                ] != 0
                other_genic = other_labels[
                    overlap_start - other.start : overlap_end - other.start
                ] != 0
                if (current_genic & other_genic).any():
                    conflicts.update((index, other_index))
            active.append((index, transcript, labels))

    if not conflicts:
        return results
    output = list(results)
    for index in sorted(conflicts):
        result = output[index]
        transcripts = tuple(
            replace(
                transcript,
                selected=False,
                zero_reasons=tuple(dict.fromkeys(
                    (*transcript.zero_reasons, "same_strand_label_overlap")
                )),
            )
            for transcript in result.transcripts
        )
        output[index] = replace(
            result,
            selected_transcript_id=None,
            mask=0.0,
            transcripts=transcripts,
        )
    return tuple(output)


def qc_all_genes(genes: tuple[Gene, ...], genome: dict[str, str], cfg: QcConfig, workers: int) -> tuple[GeneQc, ...]:
    """Run historical gene QC in input order using Linux fork workers.

    ``Executor.map`` preserves input order. Fork shares the read-only genome by
    copy-on-write instead of serializing a multi-gigabyte genome into every task.
    Label representation conflicts are handled separately by the window stage and
    must not mutate the historical QC classification.
    """
    if workers == 1:
        return tuple(qc_gene(gene, genome, cfg) for gene in genes)
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_init_qc_worker,
        initargs=(genome, cfg),
    ) as pool:
        return tuple(pool.map(_qc_gene_worker, genes, chunksize=32))
