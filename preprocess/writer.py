"""Parallel tokenization and atomic Hugging Face Dataset writing."""
from __future__ import annotations
import gc
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Iterable
import numpy as np
from .process_runtime import configure_multiprocess_resource_tracker

# Also cover library callers that invoke the writer without going through CLI.
configure_multiprocess_resource_tracker()
from datasets import Dataset, concatenate_datasets, load_from_disk
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from .windows import WindowRecord

_TOKENIZER = None
_MAX_LENGTH = 0


def _init_tokenizer(model_path: str, max_length: int):
    global _TOKENIZER, _MAX_LENGTH
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    _MAX_LENGTH = max_length


def _tokenize(sequences: list[str]) -> list[list[int]]:
    return _TOKENIZER(sequences, padding="max_length", truncation=True, max_length=_MAX_LENGTH, return_tensors=None)["input_ids"]


def _tokenize_sequences(sequences: list[str], model_path: str, max_length: int, workers: int) -> list[list[int]]:
    batch_size = min(32, max(1, len(sequences)))
    batches = [sequences[i:i+batch_size] for i in range(0, len(sequences), batch_size)]
    if workers == 1:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        output = []
        for batch in batches:
            output.extend(tokenizer(batch, padding="max_length", truncation=True, max_length=max_length, return_tensors=None)["input_ids"])
        return output
    # Tokenizers are initialized after fork, so no tokenizer thread state is
    # inherited. Linux fork avoids an additional resource-tracker process and
    # has substantially lower startup overhead than spawning for every shard.
    context = multiprocessing.get_context("fork")
    output = []
    with context.Pool(workers, initializer=_init_tokenizer, initargs=(model_path, max_length)) as pool:
        for values in pool.imap(_tokenize, batches, chunksize=1):
            output.extend(values)
    return output


def _save_part(records: list[WindowRecord], path: Path, model_path: str, workers: int) -> None:
    sequences = [x.sequence for x in records]
    ids = _tokenize_sequences(sequences, model_path, len(sequences[0])+2, workers)
    dataset = Dataset.from_dict({
        "input_ids": ids,
        # The schema has only 15 states. Persisting labels as int8 reduces the
        # largest preprocessing artifact substantially; the training losses cast
        # supervised labels to torch.long before indexing/cross-entropy.
        "labels": [x.labels.astype(np.int8, copy=False) for x in records],
        "attention_mask": [x.loss_mask.astype(np.float32, copy=False) for x in records],
        "species": [x.species for x in records],
        "chrom_id": [x.chrom_id for x in records],
        "window_start": [x.start for x in records],
        "window_end": [x.end for x in records],
        "window_id": [f"{x.chrom_id}:{x.start}-{x.end}" for x in records],
    })
    # Part persistence is an internal implementation detail. Suppress the HF
    # progress bar so one species does not print one bar per staging part.
    disable_progress_bar()
    try:
        dataset.save_to_disk(str(path))
    finally:
        enable_progress_bar()


def write_dataset(
    records: Iterable[WindowRecord],
    species_dir: Path,
    model_path: str,
    shard_rows: int,
    tokenization_workers: int,
    overwrite: bool,
) -> tuple[str, int]:
    final_path = species_dir / "dataset"
    building = species_dir / ".dataset.building"
    if final_path.exists():
        if not overwrite:
            if (final_path / "state.json").is_file() and (final_path / "dataset_info.json").is_file():
                return str(final_path), len(load_from_disk(str(final_path)))
            raise RuntimeError(f"Existing dataset is incomplete: {final_path}")
        shutil.rmtree(final_path)
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir(parents=True)
    parts_root = building / "parts"
    parts_root.mkdir()
    part_paths, buffer, total = [], [], 0
    progress = tqdm(desc=f"Saving {species_dir.name}", unit=" windows", dynamic_ncols=True)
    try:
        for record in records:
            buffer.append(record)
            if len(buffer) == shard_rows:
                path = parts_root / f"part_{len(part_paths):06d}"
                _save_part(buffer, path, model_path, tokenization_workers)
                total += len(buffer); progress.update(len(buffer))
                part_paths.append(path); buffer = []; gc.collect()
        if buffer:
            path = parts_root / f"part_{len(part_paths):06d}"
            _save_part(buffer, path, model_path, tokenization_workers)
            total += len(buffer); progress.update(len(buffer)); part_paths.append(path)
        if not part_paths:
            raise RuntimeError("No sequence windows passed preprocessing filters")
        parts = [load_from_disk(str(path)) for path in part_paths]
        combined = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        materialized = building / "materialized"
        # The single species-level bar above tracks rows as staging parts are
        # durably written. Final materialization must remain visually silent.
        disable_progress_bar()
        try:
            combined.save_to_disk(str(materialized))
        finally:
            enable_progress_bar()
        del combined, parts
        shutil.rmtree(parts_root)
        os.replace(materialized, final_path)
        shutil.rmtree(building)
        return str(final_path), total
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise
    finally:
        progress.close()
