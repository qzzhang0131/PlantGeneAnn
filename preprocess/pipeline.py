"""Single-species and serial multi-species preprocessing orchestration."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import csv
import json
import logging
from pathlib import Path
import shutil
import yaml
from .config import PreprocessConfig
from .discovery import SpeciesInput
from .fasta import read_fasta
from .gff import parse_gff
from .qc import GeneQc, mask_same_strand_coding_overlaps, qc_all_genes
from .windows import WindowRecord, generate_chromosome_windows
from .writer import write_dataset

logger = logging.getLogger("PlantGeneAnn.preprocess")


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_qc_outputs(qc_results: tuple[GeneQc, ...], qc_dir: Path, cfg: PreprocessConfig) -> None:
    qc_dir.mkdir(parents=True, exist_ok=True)
    rows = [tx for gene in qc_results for tx in gene.transcripts]
    if cfg.output.write_qc_tsv:
        with (qc_dir / "transcript_loss_mask.tsv").open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "gene_id", "transcript_id", "seqid", "strand", "start", "end",
                "cds_length", "cds_blocks", "mask", "selected", "coding_evidence",
                "zero_reasons", "half_reasons", "notes",
            ]
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                data = {key: getattr(row, key) for key in fields}
                for key in ("zero_reasons", "half_reasons", "notes"):
                    data[key] = ",".join(data[key])
                writer.writerow(data)
        with (qc_dir / "selected_transcripts.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("gene_id", "selected_transcript_id", "mask", "seqid", "start", "end", "strand"))
            for gene in qc_results:
                writer.writerow((gene.gene_id, gene.selected_transcript_id or "", gene.mask, gene.seqid, gene.mask_start, gene.mask_end, gene.strand))
    if cfg.output.write_qc_beds:
        with (qc_dir / "genes.mask0.bed").open("w", encoding="utf-8") as zero, (qc_dir / "genes.mask05.bed").open("w", encoding="utf-8") as half:
            for gene in qc_results:
                if not gene.transcripts or gene.mask not in {0.0, 0.5}: continue
                target, score = (zero, 0) if gene.mask == 0 else (half, 500)
                target.write(f"{gene.seqid}\t{gene.mask_start}\t{gene.mask_end}\t{gene.gene_id}\t{score}\t{gene.strand}\n")
    counts = {str(mask): sum(g.mask == mask for g in qc_results if g.transcripts) for mask in (0.0, 0.5, 1.0)}
    _write_json(qc_dir / "summary.json", {"genes_with_protein_coding_transcripts": sum(bool(g.transcripts) for g in qc_results), "transcripts": len(rows), "gene_mask_counts": counts})


_WINDOW_SPECIES = None
_WINDOW_GENES = None
_WINDOW_QC = None
_WINDOW_CONFIG = None


def _init_window_worker(species, genes_by_chrom, qc_results, cfg):
    global _WINDOW_SPECIES, _WINDOW_GENES, _WINDOW_QC, _WINDOW_CONFIG
    _WINDOW_SPECIES = species
    _WINDOW_GENES = genes_by_chrom
    _WINDOW_QC = qc_results
    _WINDOW_CONFIG = cfg


def _window_worker(item):
    chrom, sequence = item
    return generate_chromosome_windows(
        _WINDOW_SPECIES,
        chrom,
        sequence,
        _WINDOW_GENES.get(chrom, ()),
        _WINDOW_QC,
        _WINDOW_CONFIG,
    )


def _window_records(
    species: str,
    fasta_records: tuple[tuple[str, str], ...],
    genes,
    qc_results,
    cfg: PreprocessConfig,
):
    grouped_genes = {}
    for gene in genes:
        grouped_genes.setdefault(gene.seqid, []).append(gene)
    genes_by_chrom = {
        chrom: tuple(grouped_genes.get(chrom, ()))
        for chrom, _ in fasta_records
    }
    if cfg.workers.chromosomes == 1:
        _init_window_worker(species, genes_by_chrom, qc_results, cfg)
        groups = map(_window_worker, fasta_records)
        for group in groups:
            yield from group
        return
    import multiprocessing
    with ProcessPoolExecutor(
        max_workers=cfg.workers.chromosomes,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_init_window_worker,
        initargs=(species, genes_by_chrom, qc_results, cfg),
    ) as pool:
        for group in pool.map(_window_worker, fasta_records, chunksize=1):
            yield from group


def process_species(
    item: SpeciesInput,
    output_root: Path,
    model_path: str,
    cfg: PreprocessConfig,
    overwrite: bool,
) -> dict:
    species_dir = output_root / item.name
    species_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] Reading FASTA", item.name)
    genome = read_fasta(item.fasta)
    fasta_records = tuple(genome.items())
    logger.info("[%s] Parsing GFF", item.name)
    genes = parse_gff(item.gff)
    logger.info("[%s] QC of %d genes", item.name, len(genes))
    # Persist the historical QC result unchanged. Same-strand conflicts are not
    # an ORF-QC criterion; derive a label-safe copy only for mask/label painting.
    qc_results = qc_all_genes(genes, genome, cfg.qc, cfg.workers.qc)
    write_qc_outputs(qc_results, species_dir / "qc", cfg)
    labeling_qc = mask_same_strand_coding_overlaps(genes, qc_results)
    with (species_dir / "resolved_preprocess_config.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(cfg), handle, sort_keys=False)
    records = _window_records(item.name, fasta_records, genes, labeling_qc, cfg)
    dataset_path, rows = write_dataset(records, species_dir, model_path, cfg.output.shard_rows, cfg.workers.tokenization, overwrite)
    manifest = {
        "format_version": 1,
        "species": item.name,
        "fasta": str(item.fasta),
        "gff": str(item.gff),
        "dataset": dataset_path,
        "rows": rows,
        "fasta_records": len(fasta_records),
        "parsed_genes": len(genes),
    }
    _write_json(species_dir / "preprocess_manifest.json", manifest)
    logger.info("[%s] Completed with %d windows", item.name, rows)
    return manifest
