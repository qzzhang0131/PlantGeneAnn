"""Dependency-light GFF3/GTF parsing into a unified gene hierarchy."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote
from .fasta import open_text

GENE_TYPES = {"gene", "pseudogene"}
TRANSCRIPT_TYPES = {"mrna", "transcript", "primary_transcript", "ncrna", "lnc_rna"}
CHILD_TYPES = {"cds", "exon", "intron", "start_codon", "stop_codon"}


@dataclass(frozen=True)
class Feature:
    seqid: str
    type: str
    start: int                 # 0-based inclusive
    end: int                   # 0-based exclusive
    strand: str
    phase: Optional[int]
    attrs: dict[str, str]
    line_no: int

    @property
    def length(self): return self.end - self.start


@dataclass
class Transcript:
    id: str
    gene_id: str
    seqid: str
    strand: str
    start: int
    end: int
    attrs: dict[str, str] = field(default_factory=dict)
    cds: list[Feature] = field(default_factory=list)
    exons: list[Feature] = field(default_factory=list)
    introns: list[Feature] = field(default_factory=list)
    children: list[Feature] = field(default_factory=list)


@dataclass
class Gene:
    id: str
    seqid: str
    strand: str
    start: int
    end: int
    attrs: dict[str, str] = field(default_factory=dict)
    transcripts: list[Transcript] = field(default_factory=list)
    parse_index: int = 0


def parse_attributes(text: str) -> dict[str, str]:
    result = {}
    if text == ".": return result
    for raw in text.strip().strip(";").split(";"):
        raw = raw.strip()
        if not raw: continue
        if "=" in raw:
            key, value = raw.split("=", 1)
        elif " " in raw:
            key, value = raw.split(None, 1)
            value = value.strip().strip('"')
        else:
            key, value = raw, ""
        result[unquote(key.strip())] = unquote(value.strip())
    return result


def _parents(attrs: dict[str, str]) -> list[str]:
    raw = attrs.get("Parent") or attrs.get("parent") or ""
    return [x for x in raw.split(",") if x]


def _feature_id(attrs: dict[str, str], fallback: str) -> str:
    return attrs.get("ID") or attrs.get("transcript_id") or attrs.get("gene_id") or attrs.get("Name") or fallback


def parse_gff(path: str | Path) -> tuple[Gene, ...]:
    genes: dict[str, Gene] = {}
    transcripts: dict[str, Transcript] = {}
    pending: list[tuple[Feature, list[str]]] = []
    gene_index = 0
    with open_text(path) as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip() or raw.startswith("#"): continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 9: continue
            seqid, _source, ftype, start_s, end_s, _score, strand, phase_s, attrs_s = parts
            try:
                start, end = int(start_s) - 1, int(end_s)
            except ValueError:
                continue
            if start < 0 or end <= start: continue
            attrs = parse_attributes(attrs_s)
            phase = int(phase_s) if phase_s in {"0", "1", "2"} else None
            feature = Feature(seqid, ftype, start, end, strand, phase, attrs, line_no)
            lower = ftype.lower()
            if lower in GENE_TYPES:
                gid = _feature_id(attrs, f"gene@{line_no}")
                genes[gid] = Gene(gid, seqid, strand, start, end, attrs, parse_index=gene_index)
                gene_index += 1
            elif lower in TRANSCRIPT_TYPES:
                tid = _feature_id(attrs, f"transcript@{line_no}")
                parents = _parents(attrs)
                gid = parents[0] if parents else attrs.get("gene_id", f"gene_for_{tid}")
                transcripts[tid] = Transcript(tid, gid, seqid, strand, start, end, attrs)
            elif lower in CHILD_TYPES:
                parents = _parents(attrs)
                if not parents:
                    parent = attrs.get("transcript_id")
                    parents = [parent] if parent else []
                pending.append((feature, parents))

    # Create implicit genes/transcripts for minimally annotated GFF/GTF files.
    for transcript in transcripts.values():
        if transcript.gene_id not in genes:
            genes[transcript.gene_id] = Gene(transcript.gene_id, transcript.seqid, transcript.strand, transcript.start, transcript.end, {}, parse_index=gene_index)
            gene_index += 1
        genes[transcript.gene_id].transcripts.append(transcript)

    for feature, parents in pending:
        for parent in parents:
            transcript = transcripts.get(parent)
            if transcript is None and parent in genes:
                tid = f"{parent}.implicit_transcript"
                transcript = transcripts.get(tid)
                if transcript is None:
                    gene = genes[parent]
                    transcript = Transcript(tid, parent, gene.seqid, gene.strand, gene.start, gene.end, {})
                    transcripts[tid] = transcript
                    gene.transcripts.append(transcript)
            if transcript is None:
                # Parent may be absent. Build an implicit transcript/gene so CDS-only GFF works.
                tid = parent or f"transcript@{feature.line_no}"
                gid = f"gene_for_{tid}"
                transcript = transcripts.setdefault(tid, Transcript(tid, gid, feature.seqid, feature.strand, feature.start, feature.end, {}))
                if gid not in genes:
                    genes[gid] = Gene(gid, feature.seqid, feature.strand, feature.start, feature.end, {}, parse_index=gene_index)
                    gene_index += 1
                    genes[gid].transcripts.append(transcript)
            transcript.start = min(transcript.start, feature.start)
            transcript.end = max(transcript.end, feature.end)
            transcript.children.append(feature)
            if feature.type.lower() == "cds": transcript.cds.append(feature)
            elif feature.type.lower() == "exon": transcript.exons.append(feature)
            elif feature.type.lower() == "intron": transcript.introns.append(feature)

    for gene in genes.values():
        for tx in gene.transcripts:
            tx.cds.sort(key=lambda x: (x.start, x.end))
            tx.exons.sort(key=lambda x: (x.start, x.end))
            tx.introns.sort(key=lambda x: (x.start, x.end))
    return tuple(sorted(genes.values(), key=lambda g: g.parse_index))


def is_protein_coding(gene: Gene, transcript: Transcript) -> tuple[bool, str]:
    values = []
    for attrs in (gene.attrs, transcript.attrs):
        for key in (
            "gene_biotype",
            "gene_type",
            "transcript_biotype",
            "transcript_type",
            "biotype",
        ):
            if attrs.get(key):
                values.append(attrs[key].lower().replace("-", "_"))
    if "protein_coding" in values:
        return True, "biotype"
    # CDS fallback is used only when the annotation provides no explicit
    # biotype. Never override an explicit non-coding classification.
    if not values and transcript.cds:
        return True, "inferred_from_cds"
    return False, "none"
