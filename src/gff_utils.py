"""Strict GFF3 formatting helpers shared by PlantGeneAnn decoders."""

from __future__ import annotations

from typing import Mapping
from urllib.parse import quote


SOURCE = "PlantGeneAnn"


def _escape_gff3_attribute(value: object) -> str:
    """URL-escape one GFF3 attribute value."""

    return quote(str(value), safe=".:^*$@!+_?-|")


def _format_attributes(attributes: Mapping[str, object]) -> str:
    """Format attributes as a semicolon-separated GFF3 key-value field."""

    return ";".join(
        f"{key}={_escape_gff3_attribute(value)}"
        for key, value in attributes.items()
    )


def gff3_line(
    *,
    seqid: str,
    feature_type: str,
    start0: int,
    end0: int,
    strand: str,
    phase: str,
    attributes: Mapping[str, object],
    source: str = SOURCE,
    score: str = ".",
) -> str:
    """Create one strict 9-column GFF3 line from a half-open interval.

    Internal coordinates use 0-based ``[start0, end0)`` intervals; GFF3 uses
    1-based inclusive coordinates, so the emitted interval is
    ``start0 + 1`` through ``end0``.
    """

    if start0 < 0:
        raise ValueError(f"GFF3 feature start cannot be negative: {start0}")
    if end0 <= start0:
        raise ValueError(
            f"GFF3 feature interval must be non-empty: {start0}-{end0}"
        )

    return "\t".join(
        [
            seqid,
            source,
            feature_type,
            str(start0 + 1),
            str(end0),
            score,
            strand,
            phase,
            _format_attributes(attributes),
        ]
    )
