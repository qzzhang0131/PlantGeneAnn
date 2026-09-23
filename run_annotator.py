import argparse
import logging
import os
import tempfile
import time

from src.configuration import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DECODER_MIN_CDS_LENGTH,
    DEFAULT_DECODER_MIN_INTRON_LENGTH,
    DEFAULT_DECODER_MIN_MEAN_GENE_LOG_ODDS,
    DEFAULT_MIN_CHROM_LENGTH,
    PipelineConfig,
    build_pipeline_config_from_args,
)
from src.prediction_pipeline import run_prediction_stage
from src.runtime import (
    FastaRuntimePolicy,
    cleanup_prediction_artifacts,
    get_default_prediction_h5_path,
    resolve_num_cpu_threads,
    setup_logger,
)
from src.segmental_decoder import SegmentalDecoder

logger = logging.getLogger("PlantGeneAnn")


def _cleanup_intermediates(cache_path: str) -> None:
    """Remove prediction intermediates while preserving the cache directory."""

    cleanup_prediction_artifacts(
        cache_path,
        remove_default_outputs=True,
        best_effort=True,
    )


def _build_config(args: argparse.Namespace, cache_path: str) -> PipelineConfig:
    """Build one resolved prediction config shared with two-step inference."""

    return build_pipeline_config_from_args(
        args,
        cache_path,
        num_cpu_threads=resolve_num_cpu_threads(args.num_cpu_threads),
    )


def _run_pipeline(args, cache_path: str, annotator_script: str) -> None:
    """Execute inference followed by strict 15-state segmental SMM decoding."""

    annotator_config = _build_config(args, cache_path)
    prediction_h5 = run_prediction_stage(
        annotator_config,
        output_h5_path=get_default_prediction_h5_path(cache_path),
        annotator_script=annotator_script,
        num_processes=args.num_processes,
        compression="none",
        fasta_runtime_policy=FastaRuntimePolicy.RETAIN_FOR_DECODING,
        verbose=args.verbose,
        total_steps=4,
    )

    logger.info("Running gene decoding...")
    decoder = SegmentalDecoder(
        cache_path=cache_path,
        chromosome_h5_path=prediction_h5,
        genome_fasta=args.genome_file,
        output_gff=os.path.abspath(args.output_file),
        min_intron_length=args.min_intron_length,
        min_cds_length=args.min_cds_length,
        min_mean_gene_log_odds=args.min_mean_gene_log_odds,
        num_cpu_threads=annotator_config.num_cpu_threads,
    )
    decoder.process(step_current=4, step_total=4)
    # Note: Completion log is now emitted by decoder.process() itself


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PlantGeneAnn strict 15-state segmental annotation pipeline"
    )
    parser.add_argument(
        "-i",
        "--genome_file",
        required=True,
        help="Input genome FASTA file, optionally gzip-compressed (.gz).",
    )
    parser.add_argument("-m", "--model_path", required=True, help="Prediction model path.")
    parser.add_argument(
        "-o", "--output_file", required=True, help="Output .gff or .gff3 path."
    )
    parser.add_argument(
        "-c",
        "--num_cpu_threads",
        dest="num_cpu_threads",
        metavar="CPU_THREADS",
        type=int,
        default=None,
        help=(
            "Total CPU thread budget shared by tokenization, DataLoader workers, "
            "candidate scanning, and DAG decoding. Default: CPUs available to "
            "the current process affinity."
        ),
    )
    parser.add_argument(
        "--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Inference chunk size (default:%(default)s).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="GPU inference batch size (default:%(default)s).",
    )
    parser.add_argument(
        "--num_processes", type=int, default=None,
        help="Accelerate worker count; default uses visible GPUs.",
    )
    parser.add_argument("--cache_path", default="auto", help="Cache directory (default:auto).")
    parser.add_argument(
        "--min_chromosome_size", type=int, default=DEFAULT_MIN_CHROM_LENGTH,
        help="Minimum input record length (default:%(default)s).",
    )
    parser.add_argument(
        "--min_intron_length", type=int,
        default=DEFAULT_DECODER_MIN_INTRON_LENGTH,
        help="Hard minimum intron length (default:%(default)s).",
    )
    parser.add_argument(
        "--min_cds_length", type=int, default=DEFAULT_DECODER_MIN_CDS_LENGTH,
        help="Hard minimum complete CDS length inside the DAG (default:%(default)s).",
    )
    parser.add_argument(
        "--min_mean_gene_log_odds",
        type=float,
        default=DEFAULT_DECODER_MIN_MEAN_GENE_LOG_ODDS,
        help=(
            "Minimum additive DAG score divided by complete genomic gene span "
            "required for GFF3 output (default:%(default)s)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    setup_logger(verbose=args.verbose)
    output_ext = os.path.splitext(args.output_file)[1].lower()
    if output_ext not in (".gff", ".gff3"):
        parser.error("Output file must have .gff or .gff3 extension.")
    output_path = os.path.abspath(args.output_file)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    annotator_script = os.path.join(base_dir, "annotator.py")
    pipeline_start = time.time()
    logger.info(
        "Starting PlantGeneAnn annotation: input=%s, model=%s",
        args.genome_file,
        args.model_path,
    )

    try:
        if args.cache_path == "auto":
            tmp_base = os.path.join(base_dir, "tmp")
            os.makedirs(tmp_base, exist_ok=True)
            try:
                with tempfile.TemporaryDirectory(prefix="tmp_", dir=tmp_base) as cache_path:
                    _run_pipeline(args, cache_path, annotator_script)
            finally:
                try:
                    os.rmdir(tmp_base)
                except (FileNotFoundError, OSError):
                    pass
        else:
            cache_path = os.path.abspath(args.cache_path)
            os.makedirs(cache_path, exist_ok=True)
            try:
                _run_pipeline(args, cache_path, annotator_script)
            finally:
                _cleanup_intermediates(cache_path)
        logger.info(
            "Annotation completed: output=%s, elapsed=%.1fs",
            output_path,
            time.time() - pipeline_start,
        )
    except KeyboardInterrupt:
        logger.warning("Pipeline cancelled by user")
        raise SystemExit(130) from None
    except Exception:
        logger.exception("Pipeline failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
