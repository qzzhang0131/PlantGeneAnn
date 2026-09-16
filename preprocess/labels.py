"""Strict strand-specific 15-state transcript labeling."""
from __future__ import annotations
import numpy as np
from .gff import Transcript


def transcript_labels(tx: Transcript) -> np.ndarray:
    """Return labels over ``[tx.start, tx.end)`` using the production grammar."""
    length = tx.end - tx.start
    labels = np.zeros(length, dtype=np.int8)
    features = []
    cds_genomic = sorted(tx.cds, key=lambda x: (x.start, x.end))
    for left, right in zip(cds_genomic, cds_genomic[1:]):
        if left.end < right.start:
            features.append(("intron", left.end, right.start, None))
    features.extend(("CDS", x.start, x.end, x.phase) for x in tx.cds)
    features.sort(key=lambda x: x[1], reverse=tx.strand == "-")
    cds_features = [x for x in features if x[0] == "CDS"]
    total = sum(end-start for _, start, end, _ in cds_features)
    if not total:
        return labels
    first_len = cds_features[0][2] - cds_features[0][1]
    last_len = cds_features[-1][2] - cds_features[-1][1]
    first_short = first_len if first_len < 3 and len(cds_features) > 1 else 0
    last_short = last_len if last_len < 3 and len(cds_features) > 1 else 0
    head_donor = {0: 9, 1: 10, 2: 8}
    cds_seen = bases_seen = 0
    next_phase = 0
    previous_phase = None
    for kind, start, end, raw_phase in features:
        if kind == "intron":
            if previous_phase is not None:
                labels[start-tx.start:end-tx.start] = 1 + previous_phase
            continue
        cds_seen += 1
        first, last = cds_seen == 1, cds_seen == len(cds_features)
        positions = range(start, end) if tx.strand == "+" else range(end-1, start-1, -1)
        feature_len = end-start
        for offset, pos in enumerate(positions):
            phase = next_phase
            if bases_seen == 0: label = 7
            elif bases_seen == total-1: label = 14
            elif last and last_short == 2 and offset == feature_len-2: label = 5
            elif first and first_short == 2 and offset == 1: label = head_donor.get(raw_phase or 0, 9)
            elif offset == 0 and not first: label = 11 + phase
            elif offset == feature_len-1 and not last: label = 8 + phase
            else: label = 4 + phase
            labels[pos-tx.start] = label
            if first and first_short in (1, 2) and offset == first_short-1:
                previous_phase = ((raw_phase or 0) + 1) % 3
                next_phase = ((raw_phase or 0) + 2) % 3
            else:
                previous_phase = phase
                next_phase = (phase + 1) % 3
            bases_seen += 1
    return labels


def paint_transcript(
    destination: np.ndarray,
    owner: np.ndarray,
    tx: Transcript,
    gene_id: str,
    window_start: int,
    window_end: int,
) -> None:
    """Paint one transcript into a local window and reject same-strand overlaps."""
    overlap_start, overlap_end = max(tx.start, window_start), min(tx.end, window_end)
    if overlap_start >= overlap_end: return
    source = transcript_labels(tx)[overlap_start-tx.start:overlap_end-tx.start]
    local_start, local_end = overlap_start-window_start, overlap_end-window_start
    genic = source != 0
    existing = owner[local_start:local_end] != ""
    if np.any(genic & existing):
        conflict = np.flatnonzero(genic & existing)[0] + overlap_start
        other = owner[conflict-window_start]
        raise ValueError(
            f"Same-strand annotation overlap at {tx.seqid}:{conflict}-{conflict+1}: "
            f"{other} vs {gene_id}/{tx.id}"
        )
    view = destination[local_start:local_end]
    view[genic] = source[genic]
    owner_view = owner[local_start:local_end]
    owner_view[genic] = f"{gene_id}/{tx.id}"
