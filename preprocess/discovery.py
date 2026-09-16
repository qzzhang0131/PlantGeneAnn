"""Strict discovery of species FASTA/GFF pairs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

FASTA_SUFFIXES = (".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")
GFF_SUFFIXES = (".gff", ".gff3", ".gff.gz", ".gff3.gz")


@dataclass(frozen=True)
class SpeciesInput:
    name: str
    directory: Path
    fasta: Path
    gff: Path


def _matches(path: Path, suffixes: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return path.is_file() and any(name.endswith(suffix) for suffix in suffixes)


def discover_species(input_dir: str) -> tuple[SpeciesInput, ...]:
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {root}")
    species = []
    errors = []
    directories = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not directories:
        raise ValueError(f"Input directory contains no species subdirectories: {root}")
    for directory in directories:
        files = list(directory.iterdir())
        fastas = sorted(p for p in files if _matches(p, FASTA_SUFFIXES))
        gffs = sorted(p for p in files if _matches(p, GFF_SUFFIXES))
        if len(fastas) != 1 or len(gffs) != 1:
            errors.append(
                f"{directory.name}: expected exactly one FASTA and one GFF/GFF3; "
                f"found FASTA={len(fastas)}, GFF={len(gffs)}"
            )
            continue
        species.append(SpeciesInput(directory.name, directory, fastas[0], gffs[0]))
    if errors:
        raise ValueError("Invalid species input layout:\n" + "\n".join(errors))
    return tuple(species)
