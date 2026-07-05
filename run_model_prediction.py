import argparse
import logging
import os
import shutil
import sys
import time

from datasets import config as datasets_config

from src.pipeline_utils import (
    cleanup_prediction_runtime_cache,
    detect_num_processes,
    ensure_prediction_disk_space,
    run_accelerate_subprocess,
)
from src.chromosome_writer import (
    cleanup_temporary_chromosome_h5,
    initialize_chromosome_h5,
    prepare_chromosome_h5_output,
    promote_chromosome_h5,
)
from src.configuration import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FLANK_LENGTH,
    DEFAULT_INFERENCE_MIXED_PRECISION,
    DEFAULT_MIN_CHROM_LENGTH,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEQUENCE_LENGTH,
    PipelineConfig,
)
from src.logging_config import setup_logger
from src.sequence_extractor import SequenceExtractor
from src.sequence_tokenizer import SequenceTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("PlantGeneAnn")


def _remove_chunk_tsv_files(cache_path: str, num_chunks: int) -> int:
    """Remove TSV files generated for the current tokenization run.

    Only the exact ``chunk_1.tsv`` through ``chunk_{num_chunks}.tsv`` paths are
    removed. Avoiding a wildcard prevents this normal post-tokenization cleanup
    from deleting unrelated or stale files that were not produced by this run.

    Returns:
        Number of TSV files removed.
    """

    removed_count = 0
    for chunk_number in range(1, num_chunks + 1):
        chunk_tsv_path = os.path.join(cache_path, f"chunk_{chunk_number}.tsv")
        if os.path.isfile(chunk_tsv_path):
            os.remove(chunk_tsv_path)
            removed_count += 1
    return removed_count


def _remove_stale_cache_entries(cache_path: str) -> None:
    """Remove known PlantGeneAnn intermediates from a dedicated cache directory."""

    cleanup_prediction_runtime_cache(cache_path, keep_datasets=False)

    for name in (
        "chromosome_predictions.h5",
        "chromosome_predictions.h5.tmp",
    ):
        path = os.path.join(cache_path, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)



def _build_config(args: argparse.Namespace, cache_path: str) -> PipelineConfig:
    """Create the shared pipeline config used by extraction/tokenization."""

    return PipelineConfig(
        input_fasta=args.genome_file,
        model_path=args.model_path,
        cache_path=cache_path,
        sequence_length=args.sliding_window_size,
        flank_length=args.flank_window_size,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_tokenize_proc=args.num_tokenize_threads,
        min_chrom_length=args.min_chromosome_size,
    )


def _run_prediction_cache_impl(
    args: argparse.Namespace,
    cache_path: str,
) -> str:
    """Implement extraction, tokenization, and direct prediction writing."""

    if args.clean_cache:
        logger.debug("Cleaning known stale intermediates from cache: %s", cache_path)
        _remove_stale_cache_entries(cache_path)

    datasets_dir = os.path.join(cache_path, "datasets")
    datasets_config.HF_DATASETS_CACHE = datasets_dir
    os.environ["HF_HOME"] = os.path.join(cache_path, "huggingface")
    os.environ["HF_DATASETS_CACHE"] = datasets_dir

    config = _build_config(args, cache_path)

    step_start = time.time()
    logger.info("[Step 1/3]: Extracting sequence windows...")
    sequence_extractor = SequenceExtractor(config)
    chrom_sequence_info, num_chunks = sequence_extractor.process()
    total_windows = sum(num_windows for _, num_windows in chrom_sequence_info.values())
    logger.info(
        "[Step 1/3]: Completed - records=%d, windows=%d, chunks=%d, elapsed=%.1fs",
        len(chrom_sequence_info),
        total_windows,
        num_chunks,
        time.time() - step_start,
    )

    step_start = time.time()
    logger.info("[Step 2/3]: Tokenizing sequence windows...")
    if os.path.exists(datasets_dir):
        shutil.rmtree(datasets_dir)
    for chunk_number in range(1, num_chunks + 1):
        stale_chunk_dir = os.path.join(cache_path, f"chunk_{chunk_number}")
        if os.path.isdir(stale_chunk_dir):
            shutil.rmtree(stale_chunk_dir)
    sequence_tokenizer = SequenceTokenizer(
        config,
        keep_datasets=args.keep_datasets,
    )
    sequence_tokenizer.process(num_chunks)
    if not args.keep_datasets:
        if os.path.exists(datasets_dir):
            shutil.rmtree(datasets_dir)
        removed_tsv_count = _remove_chunk_tsv_files(cache_path, num_chunks)
        logger.debug(
            "Removed %d chunk TSV file(s) after tokenization "
            "(--keep_datasets disabled)",
            removed_tsv_count,
        )
    logger.info("[Step 2/3]: Completed - elapsed=%.1fs", time.time() - step_start)

    step_start = time.time()
    num_processes = args.num_processes if args.num_processes is not None else detect_num_processes()
    if num_processes <= 0:
        raise ValueError(f"--num_processes must be positive, got {num_processes}.")

    chromosome_h5_path = os.path.abspath(
        args.output_h5 or os.path.join(cache_path, "chromosome_predictions.h5")
    )
    ensure_prediction_disk_space(
        chromosome_h5_path=chromosome_h5_path,
        chrom_sequence_info=chrom_sequence_info,
    )
    chromosome_h5_path, temporary_chromosome_h5_path = (
        prepare_chromosome_h5_output(chromosome_h5_path)
    )
    try:
        initialize_chromosome_h5(
            temporary_chromosome_h5_path,
            chrom_sequence_info,
            compression=args.compression,
            compression_level=args.compression_level,
            hdf5_chunk_bp=(
                args.sliding_window_size - 2 * args.flank_window_size
            ),
        )
        logger.info(
            "[Step 3/3]: Running model inference - %d process(es)",
            num_processes,
        )
        base_dir = os.path.dirname(os.path.abspath(__file__))
        annotator_script = os.path.join(base_dir, "annotator.py")
        accelerate_cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes", str(num_processes),
            "--num_machines", "1",
            "--mixed_precision", DEFAULT_INFERENCE_MIXED_PRECISION,
            "--dynamo_backend", "no",
            annotator_script,
            "--model_path", args.model_path,
            "--cache_path", cache_path,
            "--output_h5_path", temporary_chromosome_h5_path,
            "--num_chunks", str(num_chunks),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
        ]
        if args.verbose:
            accelerate_cmd.append("--verbose")
        run_accelerate_subprocess(accelerate_cmd)
        promote_chromosome_h5(
            temporary_chromosome_h5_path,
            chromosome_h5_path,
            chrom_sequence_info,
        )
    finally:
        try:
            cleanup_temporary_chromosome_h5(temporary_chromosome_h5_path)
        except OSError as cleanup_error:
            logger.warning(
                "Could not remove temporary prediction HDF5 %s: %s",
                temporary_chromosome_h5_path,
                cleanup_error,
            )
    logger.info("[Step 3/3]: Completed - elapsed=%.1fs", time.time() - step_start)

    return chromosome_h5_path


def run_prediction_cache(args: argparse.Namespace) -> str:
    """Run prediction and always clean caches that were not requested persistently."""

    cache_path = os.path.abspath(args.cache_path)
    os.makedirs(cache_path, exist_ok=True)

    try:
        return _run_prediction_cache_impl(args, cache_path)
    finally:
        cleanup_prediction_runtime_cache(
            cache_path,
            keep_datasets=bool(getattr(args, "keep_datasets", False)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PlantGeneAnn model inference and directly stream genomic-record-level "
            "probability HDF5 without running GFF/HMM decoding."
        )
    )
    parser.add_argument("-i", "--genome_file", required=True, help="Input genome FA/FNA/FASTA file.")
    parser.add_argument("-m", "--model_path", required=True, help="Path to the trained prediction model.")
    parser.add_argument(
        "--cache_path",
        required=True,
        help="Persistent working cache directory. Use a dedicated directory for this script.",
    )
    parser.add_argument(
        "--output_h5",
        default=None,
        help="Output genomic-record-level HDF5 path. Default: <cache_path>/chromosome_predictions.h5.",
    )
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for sequence windows (default:%(default)s).")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size (default:%(default)s).")
    parser.add_argument("--num_processes", type=int, default=None, help="Accelerate worker processes.")
    parser.add_argument("--num_tokenize_threads", type=int, default=16, help="Tokenizer worker processes.")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader worker processes (default:%(default)s).")
    parser.add_argument("--sliding_window_size", type=int, default=DEFAULT_SEQUENCE_LENGTH, help="Model input window length (default:%(default)s).")
    parser.add_argument("--flank_window_size", type=int, default=DEFAULT_FLANK_LENGTH, help="Flank/context length (default:%(default)s).")
    parser.add_argument(
        "--min_chromosome_size",
        type=int,
        default=DEFAULT_MIN_CHROM_LENGTH,
        help="Skip genomic records (chromosomes/scaffolds/contigs) shorter than this length.",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "gzip"),
        default="none",
        help="Compression for chromosome_predictions.h5.",
    )
    parser.add_argument("--compression_level", type=int, default=4, help="Gzip compression level.")
    parser.add_argument(
        "--keep_datasets",
        action="store_true",
        help=(
            "Keep the HuggingFace datasets cache, tokenized chunk_N datasets, "
            "temporary tokenization shards, and chunk_N.tsv sequence-window files. "
            "By default all four categories are removed. The disposable "
            "huggingface/ runtime cache is always removed."
        ),
    )
    parser.add_argument(
        "--clean_cache",
        action="store_true",
        help="Remove known stale PlantGeneAnn intermediates from cache_path before running.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(verbose=args.verbose)

    start = time.time()
    logger.info(
        "Starting prediction cache: input=%s, model=%s, cache=%s",
        args.genome_file,
        args.model_path,
        os.path.abspath(args.cache_path),
    )

    try:
        chromosome_h5_path = run_prediction_cache(args)
    except KeyboardInterrupt:
        logger.warning("Prediction-cache pipeline cancelled by user")
        raise SystemExit(130) from None
    except Exception:
        logger.exception("Prediction-cache pipeline failed")
        raise SystemExit(1) from None

    logger.info(
        "Prediction cache completed: output=%s, elapsed=%.1fs",
        chromosome_h5_path,
        time.time() - start,
    )


if __name__ == "__main__":
    main()
