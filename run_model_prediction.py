import argparse
import logging
import os
import time

from src.configuration import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MIN_CHROM_LENGTH,
    PipelineConfig,
    build_pipeline_config_from_args,
)
from src.prediction_pipeline import run_prediction_stage
from src.runtime import (
    get_default_prediction_h5_path,
    resolve_num_cpu_threads,
    setup_logger,
)

logger = logging.getLogger("PlantGeneAnn")


def _build_config(args: argparse.Namespace, cache_path: str) -> PipelineConfig:
    """Create the shared pipeline config used by extraction/tokenization."""

    return build_pipeline_config_from_args(
        args,
        cache_path,
        num_cpu_threads=resolve_num_cpu_threads(args.num_cpu_threads),
    )


def _run_prediction_cache_impl(
    args: argparse.Namespace,
    cache_path: str,
) -> str:
    """Implement extraction, tokenization, and direct prediction writing."""

    config = _build_config(args, cache_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return run_prediction_stage(
        config,
        output_h5_path=(
            args.output_h5
            or get_default_prediction_h5_path(cache_path)
        ),
        annotator_script=os.path.join(base_dir, "annotator.py"),
        num_processes=args.num_processes,
        compression="none",
        compression_level=4,
        verbose=args.verbose,
        total_steps=3,
    )


def run_prediction_cache(args: argparse.Namespace) -> str:
    """Run prediction; the stage owns and removes all runtime cache."""

    cache_path = os.path.abspath(args.cache_path)
    os.makedirs(cache_path, exist_ok=True)
    return _run_prediction_cache_impl(args, cache_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PlantGeneAnn model inference and directly stream genomic-record-level "
            "probability HDF5 without running GFF/segmental decoding."
        )
    )
    parser.add_argument(
        "-i",
        "--genome_file",
        required=True,
        help="Input genome FA/FNA/FASTA file, optionally gzip-compressed (.gz).",
    )
    parser.add_argument(
        "-c",
        "--num_cpu_threads",
        dest="num_cpu_threads",
        metavar="CPU_THREADS",
        type=int,
        default=None,
        help=(
            "Total CPU thread budget shared by tokenization and per-rank "
            "DataLoader workers. Default: CPUs available to the current "
            "process affinity."
        ),
    )
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
    parser.add_argument(
        "--min_chromosome_size",
        type=int,
        default=DEFAULT_MIN_CHROM_LENGTH,
        help="Skip genomic records (chromosomes/scaffolds/contigs) shorter than this length.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(verbose=args.verbose)

    start = time.time()
    logger.info(
        "Starting PlantGeneAnn model prediction: input=%s, model=%s",
        args.genome_file,
        args.model_path,
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
