"""FASTA readers and sequence helpers."""
from __future__ import annotations
import gzip
from pathlib import Path
from typing import Iterator

RC_TABLE = str.maketrans("ACGTRYKMSWBDHVNacgtrykmswbdhvn", "TGCAYRMKSWVHDBNtgcayrmkswvhdbn")


def open_text(path: str | Path):
    path = str(path)
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.lower().endswith(".gz") else open(path, "r", encoding="utf-8", errors="replace")


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    name = None
    pieces = []
    seen = set()
    with open_text(path) as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(pieces).upper()
                fields = line[1:].split()
                if not fields:
                    raise ValueError(f"Empty FASTA header at line {line_no}")
                name = fields[0]
                if name in seen:
                    raise ValueError(f"Duplicate FASTA record: {name}")
                seen.add(name)
                pieces = []
            else:
                if name is None:
                    raise ValueError("FASTA sequence appears before the first header")
                pieces.append(line)
    if name is not None:
        yield name, "".join(pieces).upper()


def read_fasta(path: str | Path) -> dict[str, str]:
    return dict(iter_fasta(path))


def reverse_complement(sequence: str) -> str:
    return sequence.translate(RC_TABLE)[::-1].upper()
