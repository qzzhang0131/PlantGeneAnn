import argparse
import logging
import os
import time

from src.configuration import (
    CANDIDATE_SCAN_MAX_WORKERS,
    DEFAULT_DECODER_MIN_CDS_LENGTH,
    DEFAULT_DECODER_MIN_INTRON_LENGTH,
    DEFAULT_DECODER_MIN_MEAN_GENE_LOG_ODDS,
)
from src.runtime import resolve_num_cpu_threads, setup_logger
from src.segmental_decoder import SegmentalDecoder

logger = logging.getLogger("PlantGeneAnn")


def run_decoder(args: argparse.Namespace) -> str:
    """Decode an existing chromosome-level 15-state probability HDF5."""

    chromosome_h5 = os.path.abspath(args.chromosome_h5)
    output_gff = os.path.abspath(args.output_file)
    output_ext = os.path.splitext(output_gff)[1].lower()
    if output_ext not in (".gff", ".gff3"):
        raise ValueError("Output file must end with .gff or .gff3.")
    output_directory = os.path.dirname(output_gff)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    cache_path = (
        os.path.abspath(args.cache_path)
        if args.cache_path
        else os.path.dirname(chromosome_h5)
    )
    num_cpu_threads = resolve_num_cpu_threads(args.num_cpu_threads)
    decoder = SegmentalDecoder(
        cache_path=cache_path,
        chromosome_h5_path=chromosome_h5,
        genome_fasta=args.genome_file,
        output_gff=output_gff,
        min_intron_length=args.min_intron_length,
        min_cds_length=args.min_cds_length,
        min_mean_gene_log_odds=args.min_mean_gene_log_odds,
        num_cpu_threads=num_cpu_threads,
    )
    return decoder.process(step_current=1, step_total=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PlantGeneAnn strict 15-state segmental weighted-DAG decoding "
            "without rerunning model inference."
        )
    )
    parser.add_argument(
        "-i",
        "--genome_file",
        required=True,
        help="Input genome FASTA, optionally gzip-compressed (.gz).",
    )
    parser.add_argument(
        "--chromosome_h5", required=True,
        help="Chromosome-level PlantGeneAnn 15-state probability HDF5.",
    )
    parser.add_argument("-o", "--output_file", required=True, help="Output GFF3 path.")
    parser.add_argument(
        "--cache_path", default=None,
        help="Optional cache directory; defaults to the HDF5 directory.",
    )
    parser.add_argument(
        "--min_intron_length", type=int,
        default=DEFAULT_DECODER_MIN_INTRON_LENGTH,
        help="Hard minimum intron length (default:%(default)s).",
    )
    parser.add_argument(
        "--min_cds_length", type=int, default=DEFAULT_DECODER_MIN_CDS_LENGTH,
        help="Hard complete-CDS minimum enforced inside the DAG (default:%(default)s).",
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
    parser.add_argument(
        "-c",
        "--num_cpu_threads",
        dest="num_cpu_threads",
        metavar="CPU_THREADS",
        type=int,
        default=None,
        help=(
            "Total CPU thread budget. Candidate scanning uses at most "
            f"{CANDIDATE_SCAN_MAX_WORKERS} workers; DAG decoding may use the "
            "full budget. Default: CPUs available to the current process affinity."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(verbose=args.verbose)
    started = time.time()
    logger.info(
        "Starting PlantGeneAnn gene decoding: predictions=%s, output=%s",
        args.chromosome_h5,
        args.output_file,
    )
    logger.info("Running gene decoding...")
    try:
        output_gff = run_decoder(args)
    except KeyboardInterrupt:
        logger.warning("Cancelled by user")
        raise SystemExit(130) from None
    except Exception:
        logger.exception("Failed")
        raise SystemExit(1) from None
    logger.info(
        "Completed - output=%s, elapsed=%.1fs",
        output_gff,
        time.time() - started,
    )


if __name__ == "__main__":
    main()
