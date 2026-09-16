import argparse
import gc
import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from src.chromosome_writer import ChromosomePredictionWriter
from src.configuration import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INFERENCE_MIXED_PRECISION,
)
from src.constants import (
    INTEGER_METADATA_COLUMNS,
    PREDICTION_NUM_CHANNELS,
    PREDICTION_NUM_CLASSES,
    PREDICTION_NUM_STRANDS,
    REQUIRED_METADATA_COLUMNS,
)
from src.runtime import configure_single_threaded_libraries, setup_logger

logger = logging.getLogger("PlantGeneAnn.annotator")


class GenomeAnnotator:
    """Run distributed 15-state annotation inference with SegmentCaduceusPh.

    The model predicts two independent strand-specific categorical tracks with
    15 mutually exclusive labels per strand. The 30 output channels contain
    all positive-strand labels followed by all negative-strand labels.

    Per-strand labels are encoded as:

        0: background
        1-3: frame-indexed intron states 0/1/2
        4-6: CDS frame states 0/1/2 (first/second/third codon base)
        7: START (first start-codon base)
        8-10: donor states 0/1/2
        11-13: acceptor states 0/1/2
        14: STOP (last stop-codon base)

    State suffixes follow the confirmed label transition graph and are not
    conventional GFF3 phases. GFF3 phase is calculated later from cumulative
    CDS length in transcript direction.

    For each inference chunk, the in-memory probability tensor has shape
    ``(N, 2, L, C)``:
        - N: number of sequence windows in the chunk
        - 2: strand axis, where 0 = positive strand and 1 = negative strand
        - L: center-output length after any model-side cropping
        - C: number of mutually exclusive classes per strand (15)

    Values are complete per-strand 15-way softmax distributions saved as
    ``float16`` for strict segmental weighted-DAG emissions.
    """

    def __init__(
        self,
        model_path: str,
        cache_path: str,
        output_h5_path: str,
        num_chunks: int,
        batch_size: int,
        num_workers: int,
    ):
        self.model_path = model_path
        self.cache_path = cache_path
        self.output_h5_path = os.path.abspath(output_h5_path)
        self.num_chunks = num_chunks
        self.batch_size = batch_size
        self.num_workers = int(num_workers)
        if self.num_workers < 0:
            raise ValueError("DataLoader num_workers cannot be negative.")
        configure_single_threaded_libraries()
        # The pipeline uses a fixed 15-state label schema. Compatibility is
        # enforced against actual logits at inference time rather than by
        # reading or validating model configuration metadata.
        self.num_features = PREDICTION_NUM_CLASSES
        self.accelerator = Accelerator(
            mixed_precision=DEFAULT_INFERENCE_MIXED_PRECISION
        )
        self.device = self.accelerator.device
        try:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {self.model_path}: {e}"
            ) from e

        self.model.to(self.device)
        self.model.eval()
        self.model = self.accelerator.prepare(self.model)

    def _logits_to_strand_probabilities(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Convert 30-channel logits to two 15-class softmax distributions.

        Returns:
            ``(batch, 2, length, 15)`` float16 probabilities, independently
            normalized over the 15 labels of each strand.
        """
        if logits.ndim != 3:
            raise ValueError(
                "PlantGeneAnn logits must have shape (batch, length, channels), "
                f"but got {tuple(logits.shape)}."
            )
        if logits.shape[0] <= 0 or logits.shape[1] <= 0:
            raise ValueError(
                "PlantGeneAnn logits must contain at least one batch item and "
                f"one sequence position, got {tuple(logits.shape)}."
            )
        expected_channels = PREDICTION_NUM_CHANNELS
        if logits.shape[-1] != expected_channels:
            raise ValueError(
                "PlantGeneAnn 15-state logits must contain exactly "
                f"{expected_channels} channels (= {PREDICTION_NUM_STRANDS} "
                f"strands * {PREDICTION_NUM_CLASSES} states), but got "
                f"{logits.shape[-1]}."
            )

        positive_logits = logits[..., :PREDICTION_NUM_CLASSES]
        negative_logits = logits[..., PREDICTION_NUM_CLASSES:]

        positive_probs = torch.softmax(positive_logits.float(), dim=-1)
        negative_probs = torch.softmax(negative_logits.float(), dim=-1)

        full_probs = torch.stack(
            (positive_probs, negative_probs),
            dim=1,
        ).to(torch.float16)

        return full_probs

    def _extract_window_metadata(self, datasets, chunk_number: int) -> Dict[str, np.ndarray]:
        """Extract per-window genomic metadata from a tokenized chunk.
        
        ``SequenceExtractor`` writes these columns to ``chunk_N.tsv`` and
        ``SequenceTokenizer`` preserves them in the HuggingFace Dataset. Persisting
        these values with the gathered predictions allows global rank zero to
        write each window directly into genomic-record coordinates, especially
        for multi-sequence FASTA/FNA inputs.
        
        Coordinate convention:
            ``center_start`` and ``center_end`` are 0-based half-open genomic
            coordinates on the corresponding ``chrom_id`` sequence.
        """
        missing_columns = [
            column
            for column in REQUIRED_METADATA_COLUMNS
            if column not in datasets.column_names
        ]
        if missing_columns:
            raise ValueError(
                f"Tokenized chunk {chunk_number} is missing required genomic "
                f"metadata columns: {missing_columns}. Please regenerate the "
                "cache with SequenceExtractor and SequenceTokenizer."
            )

        metadata: Dict[str, np.ndarray] = {}

        for column in INTEGER_METADATA_COLUMNS:
            values = np.asarray(datasets[column], dtype=np.int64)
            if len(values) != len(datasets):
                raise ValueError(
                    f"Metadata column '{column}' in chunk {chunk_number} has "
                    f"{len(values)} rows, expected {len(datasets)}."
                )
            metadata[column] = values

        chrom_ids = np.asarray([str(chrom_id) for chrom_id in datasets["chrom_id"]], dtype=object)
        if len(chrom_ids) != len(datasets):
            raise ValueError(
                f"Metadata column 'chrom_id' in chunk {chunk_number} has "
                f"{len(chrom_ids)} rows, expected {len(datasets)}."
            )
        metadata["chrom_id"] = chrom_ids

        return metadata

    def _build_dataloader(self, datasets) -> DataLoader:
        """Build a robust DataLoader for one tokenized chunk.

        Tuned for GPU inference throughput within one chunk:
        - ``prefetch_factor=2`` keeps two batches ready per worker, reducing
          GPU idle time without excessive host memory pressure.
        - Workers are deliberately non-persistent because each chunk creates a
          different DataLoader backed by different Arrow files. Keeping old
          workers alive would retain file mappings and accumulate processes.
        """
        dataset_len = len(datasets)
        if dataset_len == 0:
            raise ValueError("Tokenized dataset is empty; cannot run model inference on an empty chunk.")

        effective_batch_size = min(dataset_len, self.batch_size)
        effective_num_workers = min(dataset_len, self.num_workers)

        dataloader_kwargs: Dict = {
            "batch_size": effective_batch_size,
            "num_workers": effective_num_workers,
            "pin_memory": torch.cuda.is_available(),
            "shuffle": False,
        }

        if effective_num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = 2
            dataloader_kwargs["persistent_workers"] = False

        return DataLoader(datasets, **dataloader_kwargs)

    @staticmethod
    def _store_gathered_batch(
        gathered_indices: np.ndarray,
        gathered_full_probs: np.ndarray,
        ordered_full_probs: np.ndarray,
        seen_rows: np.ndarray,
        chunk_number: int,
    ) -> None:
        """Store one gathered batch in deterministic tokenized-dataset order.

        Accelerate may shard a DataLoader across processes and may duplicate tail
        samples so every process sees the same number of batches. Therefore, the
        physical gather order is not authoritative. ``chunk_local_index`` is
        used as the row key, and the first occurrence of a duplicated key wins,
        matching the previous stable-sort/de-duplication behaviour without
        allocating several full-chunk array copies.
        """

        if gathered_indices.ndim != 1:
            raise ValueError(
                f"Chunk {chunk_number} gathered indices must be 1D, "
                f"but got shape {gathered_indices.shape}."
            )
        if gathered_full_probs.shape[0] != gathered_indices.shape[0]:
            raise ValueError(
                f"Chunk {chunk_number} full_probs/index row mismatch: "
                f"{gathered_full_probs.shape[0]} probs vs "
                f"{gathered_indices.shape[0]} indices."
            )
        if gathered_full_probs.shape[1:] != ordered_full_probs.shape[1:]:
            raise ValueError(
                f"Chunk {chunk_number} gathered probability shape "
                f"{gathered_full_probs.shape[1:]} does not match the preallocated "
                f"shape {ordered_full_probs.shape[1:]}."
            )
        if seen_rows.ndim != 1 or len(seen_rows) != len(ordered_full_probs):
            raise ValueError(
                f"Chunk {chunk_number} has an invalid prediction coverage bitmap."
            )
        if len(gathered_indices) == 0:
            return

        gathered_indices = np.asarray(gathered_indices, dtype=np.int64)
        invalid_mask = (gathered_indices < 0) | (
            gathered_indices >= len(ordered_full_probs)
        )
        if np.any(invalid_mask):
            invalid_indices = np.unique(gathered_indices[invalid_mask])
            raise ValueError(
                f"Chunk {chunk_number} gathered out-of-range chunk_local_index "
                f"values: {invalid_indices[:10].tolist()}; expected range "
                f"[0, {len(ordered_full_probs)})."
            )

        # np.unique returns the first position for each key. Filtering against
        # seen_rows then preserves the first occurrence across earlier batches.
        unique_indices, first_positions = np.unique(
            gathered_indices,
            return_index=True,
        )
        new_mask = ~seen_rows[unique_indices]
        new_indices = unique_indices[new_mask]
        if len(new_indices) == 0:
            return

        ordered_full_probs[new_indices] = gathered_full_probs[
            first_positions[new_mask]
        ]
        seen_rows[new_indices] = True

    def evaluate(
        self,
        model,
        test_dataloader,
        expected_num_rows: int,
        center_length: int,
        chunk_number: int,
    ) -> np.ndarray:
        """Run distributed inference into one preallocated, ordered chunk array.

        Returns:
            On the main process, a ``float16`` array with shape
            ``(expected_num_rows, 2, center_length, num_features)``. Non-main
            processes return an empty array after participating in collectives.
        """
        if expected_num_rows <= 0:
            raise ValueError(
                f"Chunk {chunk_number} expected_num_rows must be positive."
            )
        if center_length <= 0:
            raise ValueError(f"Chunk {chunk_number} center_length must be positive.")

        model.eval()
        if self.accelerator.is_main_process:
            ordered_full_probs = np.empty(
                (
                    expected_num_rows,
                    2,
                    center_length,
                    self.num_features,
                ),
                dtype=np.float16,
            )
            seen_rows = np.zeros(expected_num_rows, dtype=np.bool_)
        else:
            ordered_full_probs = np.empty(
                (0, 2, center_length, self.num_features),
                dtype=np.float16,
            )
            seen_rows = np.empty((0,), dtype=np.bool_)

        crop_logged = False

        with torch.inference_mode(), self.accelerator.autocast():
            for batch in tqdm(
                test_dataloader,
                desc="Model Inference",
                leave=False,
                disable=not self.accelerator.is_local_main_process,
            ):
                input_ids = batch["input_ids"]
                chunk_local_indices = batch["chunk_local_index"].to(
                    device=input_ids.device,
                    dtype=torch.long,
                )
                outputs = model(input_ids=input_ids)

                strand_full_probs = (
                    self._logits_to_strand_probabilities(outputs.logits)
                )

                model_output_length = int(strand_full_probs.shape[2])
                if model_output_length < center_length:
                    raise ValueError(
                        f"Chunk {chunk_number}: model output length "
                        f"({model_output_length}) is shorter than the expected "
                        f"center_length ({center_length}). The model cannot produce "
                        "enough output for the requested window configuration."
                    )
                if model_output_length > center_length:
                    crop_total = model_output_length - center_length
                    if crop_total % 2 != 0:
                        raise ValueError(
                            f"Chunk {chunk_number}: cannot symmetrically crop "
                            f"model output length {model_output_length} to "
                            f"center length {center_length}."
                        )
                    crop_start = crop_total // 2
                    crop_end = crop_start + center_length
                    strand_full_probs = strand_full_probs[
                        :, :, crop_start:crop_end, :
                    ]
                    if self.accelerator.is_main_process and not crop_logged:
                        logger.debug(
                            "Chunk %d: cropped each gathered batch from %d to %d "
                            "bases (crop region: [%d, %d)).",
                            chunk_number,
                            model_output_length,
                            center_length,
                            crop_start,
                            crop_end,
                        )
                        crop_logged = True

                expected_batch_shape = (
                    int(input_ids.shape[0]),
                    PREDICTION_NUM_STRANDS,
                    center_length,
                    PREDICTION_NUM_CLASSES,
                )
                if tuple(strand_full_probs.shape) != expected_batch_shape:
                    raise RuntimeError(
                        f"Chunk {chunk_number}: expected cropped 15-state "
                        f"probabilities with shape {expected_batch_shape}, got "
                        f"{tuple(strand_full_probs.shape)}."
                    )

                gathered_indices = self.accelerator.gather(chunk_local_indices)
                gathered_full_probs = self.accelerator.gather(strand_full_probs)

                if self.accelerator.is_main_process:
                    batch_indices = gathered_indices.cpu().numpy().astype(
                        np.int64,
                        copy=False,
                    )
                    batch_full_probs = gathered_full_probs.cpu().numpy().astype(
                        np.float16,
                        copy=False,
                    )
                    self._store_gathered_batch(
                        gathered_indices=batch_indices,
                        gathered_full_probs=batch_full_probs,
                        ordered_full_probs=ordered_full_probs,
                        seen_rows=seen_rows,
                        chunk_number=chunk_number,
                    )
                    del batch_indices, batch_full_probs

                del outputs, strand_full_probs
                del gathered_indices, gathered_full_probs
                del input_ids, chunk_local_indices

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.accelerator.is_main_process and not np.all(seen_rows):
            missing_indices = np.flatnonzero(~seen_rows)
            raise ValueError(
                f"Chunk {chunk_number} distributed inference did not return "
                f"predictions for {len(missing_indices)} of {expected_num_rows} "
                f"rows; first missing indices={missing_indices[:10].tolist()}."
            )

        if self.accelerator.is_main_process:
            expected_chunk_shape = (
                expected_num_rows,
                PREDICTION_NUM_STRANDS,
                center_length,
                PREDICTION_NUM_CLASSES,
            )
            if tuple(ordered_full_probs.shape) != expected_chunk_shape:
                raise RuntimeError(
                    f"Chunk {chunk_number}: expected final 15-state "
                    f"probabilities with shape {expected_chunk_shape}, got "
                    f"{tuple(ordered_full_probs.shape)}."
                )

        return ordered_full_probs

    def process_chromosome(
        self,
        n: int,
        datasets_dir: str,
        chromosome_writer: Optional[ChromosomePredictionWriter],
    ):
        """Infer one chunk and stream its probabilities to genomic coordinates."""
        if self.accelerator.is_main_process:
            logger.debug("Processing chunk %d/%d", n, self.num_chunks)

        dataset_path = os.path.join(datasets_dir, f"chunk_{n}")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Cache file for chunk {n} not found: {dataset_path}")

        # These sentinels keep the cleanup block safe when an exception occurs
        # before one or more resources have been assigned.
        datasets = None
        dataloader = None
        prepared_dataloader = None
        window_metadata = None
        ordered_full_probs = None
        try:
            datasets = load_from_disk(dataset_path)
            num_windows = len(datasets)
            window_metadata = self._extract_window_metadata(datasets, n)
            expected_center_lengths = (
                window_metadata["center_end"] - window_metadata["center_start"]
            )
            if np.any(expected_center_lengths != expected_center_lengths[0]):
                raise ValueError(
                    f"Chunk {n} contains inconsistent center interval lengths."
                )
            center_length = int(expected_center_lengths[0])

            # Keep ``chunk_local_index`` in every batch as the authoritative row
            # key for deterministic order restoration after distributed gather.
            datasets.set_format(
                type="torch",
                columns=["input_ids", "chunk_local_index"],
            )
            dataloader = self._build_dataloader(datasets)
            prepared_dataloader = self.accelerator.prepare(dataloader)

            ordered_full_probs = self.evaluate(
                self.model,
                prepared_dataloader,
                expected_num_rows=num_windows,
                center_length=center_length,
                chunk_number=n,
            )

            if self.accelerator.is_main_process:
                if chromosome_writer is None:
                    raise RuntimeError(
                        "Global rank zero is missing the chromosome writer."
                    )
                chromosome_writer.write_chunk(
                    chunk_number=n,
                    metadata=window_metadata,
                    probabilities=ordered_full_probs,
                )

            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                logger.info(
                    "Inference chunk %d/%d completed",
                    n,
                    self.num_chunks,
                )
        finally:
            # Accelerator keeps references to every prepared DataLoader. Clear
            # those references after each one-pass chunk so Arrow mappings and
            # worker processes cannot accumulate across a large genome. The
            # prepared model remains strongly referenced by ``self.model``.
            ordered_full_probs = None
            window_metadata = None
            prepared_dataloader = None
            dataloader = None
            datasets = None
            self.accelerator.free_memory()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process(self):
        """Infer all chunks and directly finalize chromosome-level predictions."""

        chromosome_writer = None
        if self.accelerator.is_main_process:
            chromosome_writer = ChromosomePredictionWriter(self.output_h5_path)

        try:
            for i in range(self.num_chunks):
                self.process_chromosome(
                    i + 1,
                    self.cache_path,
                    chromosome_writer,
                )

            if self.accelerator.is_main_process:
                chromosome_writer.finalize(self.num_chunks)
            self.accelerator.wait_for_everyone()
        finally:
            if chromosome_writer is not None:
                chromosome_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PlantGeneAnn v2 distributed annotator.")
    parser.add_argument("--model_path", required=True, help="Specify the path to the PlantGeneAnn v2 prediction model.")
    parser.add_argument("--cache_path", required=True, help="Path to cache.")
    parser.add_argument("--output_h5_path", required=True, help="Initialized chromosome-level HDF5 temporary path.")
    parser.add_argument("--num_chunks", type=int, required=True, help="Number of tokenized chunks.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="The number of samples in a batch.")
    parser.add_argument("--num_workers", type=int, required=True, help="Derived DataLoader workers per Accelerate rank.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    # Accelerate starts this script in independent processes. Only global rank
    # zero owns a console handler; other ranks remain silent.
    process_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    setup_logger(verbose=args.verbose, enabled=process_rank in {"", "0"})

    annotator = GenomeAnnotator(
        model_path=args.model_path,
        cache_path=args.cache_path,
        output_h5_path=args.output_h5_path,
        num_chunks=args.num_chunks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    annotator.process()
