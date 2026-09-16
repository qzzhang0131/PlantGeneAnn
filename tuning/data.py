"""Discovery, validation, splitting, and provenance for HF Dataset collections."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


logger = logging.getLogger("PlantGeneAnn.tuning.data")
REQUIRED_COLUMNS = ("input_ids", "labels", "attention_mask")
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".ipynb_checkpoints",
    "__pycache__",
    "cache",
    "caches",
    "shards",
    "tmp",
    ".tmp",
}
DATASET_MANIFEST_NAME = "dataset_manifest.json"


@dataclass(frozen=True)
class DatasetSource:
    """One discovered Dataset and its rows used for a particular role."""

    dataset_id: int
    path: str
    name: str
    source_rows: int
    rows: int
    role: str
    split_method: str = "explicit"


@dataclass(frozen=True)
class LoadedDatasetCollection:
    """Validated source datasets before or after deterministic combination."""

    dataset: Dataset
    source_datasets: Tuple[Dataset, ...]
    sources: Tuple[DatasetSource, ...]
    sequence_length: int


def is_hf_dataset_directory(path: Path) -> bool:
    """Return whether *path* has the complete Dataset.save_to_disk layout."""

    return (
        path.is_dir()
        and (path / "state.json").is_file()
        and (path / "dataset_info.json").is_file()
        and any(path.glob("*.arrow"))
    )


def discover_hf_dataset_paths(root_path: str) -> Tuple[str, ...]:
    """Find complete HF Dataset directories under one input directory.

    If the input itself is a Dataset, it is returned directly. Otherwise the
    tree is traversed without following symlinks. Once a Dataset is found, its
    descendants are pruned so a source can never be discovered twice.
    """

    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")
    if (root / "dataset_dict.json").is_file():
        raise TypeError(
            f"Dataset path {root} is a DatasetDict. Provide a saved "
            "datasets.Dataset directory for the requested role."
        )
    if is_hf_dataset_directory(root):
        return (str(root),)

    discovered: Dict[str, Path] = {}
    for current_root, directory_names, _file_names in os.walk(
        str(root), followlinks=False
    ):
        current = Path(current_root)
        if (current / "dataset_dict.json").is_file():
            raise TypeError(
                f"Discovered DatasetDict at {current}. Provide individual saved "
                "datasets.Dataset directories instead of split containers."
            )
        if is_hf_dataset_directory(current):
            real_path = str(current.resolve())
            discovered[real_path] = current.resolve()
            directory_names[:] = []
            continue
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in IGNORED_DIRECTORY_NAMES
            and not (current / name).is_symlink()
        )

    if not discovered:
        raise FileNotFoundError(
            "No valid HuggingFace Dataset directories were found under "
            f"{root}. Expected a Dataset.save_to_disk() directory containing "
            "state.json, dataset_info.json, and at least one top-level .arrow file."
        )
    return tuple(sorted(discovered))


def validate_geneann_dataset(dataset: Dataset, *, path: str, role: str) -> int:
    """Run inexpensive structural checks and return the decoder sequence length.

    Dataset construction and loss-mask filtering are handled by ``preprocess``.
    Tuning therefore checks only the object needed by the trainer, required
    columns, and the first row's geometry. Avoid scanning every long window at
    startup: multi-species collections can contain millions of rows.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"{role} source {path} must load as datasets.Dataset, got "
            f"{type(dataset).__name__}."
        )
    if len(dataset) == 0:
        raise ValueError(f"{role} Dataset is empty: {path}")
    missing = set(REQUIRED_COLUMNS) - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"{role} Dataset {path} is missing required columns: {sorted(missing)}."
        )

    # Project to the two fields needed for the geometry check. In particular,
    # do not materialize the potentially large loss mask, even for one row.
    row = dataset.select_columns(["input_ids", "labels"])[0]
    input_ids = np.asarray(row["input_ids"])
    labels = np.asarray(row["labels"])
    if input_ids.ndim != 1 or input_ids.size < 3:
        raise ValueError(
            f"{role} Dataset {path} first row has invalid input_ids shape "
            f"{input_ids.shape}."
        )
    sequence_length = int(input_ids.size - 2)
    if sequence_length % 16 != 0:
        raise ValueError(
            f"{role} Dataset {path} first row sequence length {sequence_length} "
            "is not divisible by decoder factor 16."
        )
    if labels.shape != (2, sequence_length):
        raise ValueError(
            f"{role} Dataset {path} first row labels must have shape "
            f"(2, {sequence_length}), got {labels.shape}."
        )
    if int(input_ids[0]) != 0 or int(input_ids[-1]) != 1:
        raise ValueError(
            f"{role} Dataset {path} first row input_ids must begin with CLS "
            "id 0 and end with SEP id 1."
        )
    return sequence_length


def _combine(datasets: Sequence[Dataset]) -> Dataset:
    if not datasets:
        raise ValueError("Cannot combine an empty Dataset collection.")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(list(datasets))


def _log_collection(sources: Sequence[DatasetSource], *, role: str) -> None:
    total = sum(source.rows for source in sources)
    logger.info("Discovered %d %s Dataset source(s):", len(sources), role)
    for source in sources:
        fraction = source.rows / total if total else 0.0
        logger.info(
            "  [%d] %s rows=%d (%.2f%%), source_rows=%d",
            source.dataset_id,
            source.path,
            source.rows,
            100.0 * fraction,
            source.source_rows,
        )


def load_dataset_collection(root_path: str, *, role: str) -> LoadedDatasetCollection:
    """Discover, load, validate, normalize, and concatenate Dataset sources."""

    paths = discover_hf_dataset_paths(root_path)
    datasets: List[Dataset] = []
    sources: List[DatasetSource] = []
    expected_length: Optional[int] = None
    for dataset_id, path in enumerate(paths):
        try:
            loaded = load_from_disk(path)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load discovered {role} Dataset {path}: {error}"
            ) from error
        if isinstance(loaded, DatasetDict):
            raise TypeError(
                f"{role} source {path} is a DatasetDict with splits "
                f"{list(loaded.keys())}; provide a saved datasets.Dataset directory."
            )
        sequence_length = validate_geneann_dataset(loaded, path=path, role=role)
        if expected_length is None:
            expected_length = sequence_length
        elif sequence_length != expected_length:
            raise ValueError(
                f"Cannot combine {role} Datasets with sequence lengths "
                f"{expected_length} and {sequence_length}: {path}."
            )
        normalized = loaded.select_columns(list(REQUIRED_COLUMNS))
        datasets.append(normalized)
        sources.append(
            DatasetSource(
                dataset_id=dataset_id,
                path=str(Path(path).resolve()),
                name=Path(path).name,
                source_rows=len(loaded),
                rows=len(loaded),
                role=role,
            )
        )
    collection = LoadedDatasetCollection(
        dataset=_combine(datasets),
        source_datasets=tuple(datasets),
        sources=tuple(sources),
        sequence_length=int(expected_length),
    )
    _log_collection(collection.sources, role=role)
    return collection


def split_dataset_collection(
    collection: LoadedDatasetCollection,
    *,
    validation_fraction: float,
    seed: int,
) -> Tuple[LoadedDatasetCollection, LoadedDatasetCollection]:
    """Split each source independently, then combine train and validation parts."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    train_datasets: List[Dataset] = []
    validation_datasets: List[Dataset] = []
    train_sources: List[DatasetSource] = []
    validation_sources: List[DatasetSource] = []

    for dataset, source in zip(collection.source_datasets, collection.sources):
        if len(dataset) < 2:
            raise ValueError(
                f"Dataset {source.path} has {len(dataset)} row(s); at least 2 are "
                "required for a train/validation split."
            )
        validation_rows = max(1, int(round(len(dataset) * validation_fraction)))
        validation_rows = min(validation_rows, len(dataset) - 1)
        split = dataset.train_test_split(
            test_size=validation_rows,
            seed=int(seed) + source.dataset_id,
            shuffle=True,
        )
        train_part = split["train"]
        validation_part = split["test"]
        train_datasets.append(train_part)
        validation_datasets.append(validation_part)
        train_sources.append(
            DatasetSource(
                dataset_id=source.dataset_id,
                path=source.path,
                name=source.name,
                source_rows=source.source_rows,
                rows=len(train_part),
                role="train",
                split_method=f"per_source_random_seed_{int(seed) + source.dataset_id}",
            )
        )
        validation_sources.append(
            DatasetSource(
                dataset_id=source.dataset_id,
                path=source.path,
                name=source.name,
                source_rows=source.source_rows,
                rows=len(validation_part),
                role="validation",
                split_method=f"per_source_random_seed_{int(seed) + source.dataset_id}",
            )
        )

    train_collection = LoadedDatasetCollection(
        dataset=_combine(train_datasets),
        source_datasets=tuple(train_datasets),
        sources=tuple(train_sources),
        sequence_length=collection.sequence_length,
    )
    validation_collection = LoadedDatasetCollection(
        dataset=_combine(validation_datasets),
        source_datasets=tuple(validation_datasets),
        sources=tuple(validation_sources),
        sequence_length=collection.sequence_length,
    )
    logger.warning(
        "Using development-only per-source random validation splits. Overlapping "
        "windows or genes can leak; use chromosome/gene-disjoint validation for "
        "biological evaluation."
    )
    _log_collection(train_collection.sources, role="train")
    _log_collection(validation_collection.sources, role="validation")
    return train_collection, validation_collection


def _manifest_sources(sources: Sequence[DatasetSource]) -> List[Dict]:
    total = sum(source.rows for source in sources)
    result = []
    for source in sources:
        item = asdict(source)
        item["fraction"] = source.rows / total if total else 0.0
        result.append(item)
    return result


def save_dataset_manifest(
    output_dir: str,
    *,
    train: LoadedDatasetCollection,
    validation: LoadedDatasetCollection,
) -> str:
    """Persist exact source order, rows, split method, and mixture proportions."""

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, DATASET_MANIFEST_NAME)
    manifest = {
        "format_version": 1,
        "sampling_strategy": "window_proportional",
        "sequence_length": train.sequence_length,
        "train_total_rows": len(train.dataset),
        "validation_total_rows": len(validation.dataset),
        "train": _manifest_sources(train.sources),
        "validation": _manifest_sources(validation.sources),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
