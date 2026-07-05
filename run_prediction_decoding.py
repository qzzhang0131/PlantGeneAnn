import argparse
import logging
import os
import time

from src.chromosome_writer import validate_chromosome_h5
from src.configuration import (
    DEFAULT_HMM_MIN_CDS_LENGTH,
    DEFAULT_HMM_MIN_GENE_LENGTH,
    DEFAULT_HMM_MIN_GENE_SCORE,
    DEFAULT_HMM_MIN_INTRON_LENGTH,
    DEFAULT_HMM_SPLICE_EVENT_PROB,
)
from src.hmm_decoder import HMMDecoder
from src.logging_config import setup_logger

logger = logging.getLogger("PlantGeneAnn")


def run_hmm(args: argparse.Namespace) -> str:
    """Run HMM decoding from an existing genomic-record-level probability HDF5."""

    chromosome_h5 = os.path.abspath(args.chromosome_h5)
    if not os.path.exists(chromosome_h5):
        raise FileNotFoundError(f"Chromosome-level HDF5 not found: {chromosome_h5}")
    validate_chromosome_h5(chromosome_h5)

    output_gff = os.path.abspath(args.output_file)
    output_ext = os.path.splitext(output_gff)[1].lower()
    if output_ext not in (".gff", ".gff3"):
        raise ValueError(f"Output file must end with .gff or .gff3, got {output_ext!r}.")

    output_dir = os.path.dirname(output_gff)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cache_path = os.path.abspath(args.cache_path) if args.cache_path else os.path.dirname(chromosome_h5)
    num_threads = args.num_hmm_threads if args.num_hmm_threads is not None else os.cpu_count() or 1

    decoder = HMMDecoder(
        cache_path=cache_path,
        chromosome_h5_path=chromosome_h5,
        genome_fasta=args.genome_file,
        output_gff=output_gff,
        min_intron_length=args.min_intron_length,
        min_cds_length=args.min_cds_length,
        min_gene_length=args.min_gene_length,
        min_gene_score=args.min_gene_score,
        splice_event_prob=args.hmm_splice_event_prob,
        num_threads=num_threads,
    )
    return decoder.process()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PlantGeneAnn HMM decoding from an existing chromosome_predictions.h5 "
            "containing genomic-record predictions, without rerunning deep-learning inference."
        )
    )
    parser.add_argument("-i", "--genome_file", required=True, help="Input genome FA/FNA/FASTA file.")
    parser.add_argument(
        "--chromosome_h5",
        required=True,
        help="Path to genomic-record-level PlantGeneAnn probability HDF5.",
    )
    parser.add_argument("-o", "--output_file", required=True, help="Output GFF3 path.")
    parser.add_argument(
        "--cache_path",
        default=None,
        help="Optional cache path for HMMDecoder. Default: directory containing --chromosome_h5.",
    )
    parser.add_argument(
        "--min_intron_length",
        type=int,
        default=DEFAULT_HMM_MIN_INTRON_LENGTH,
        help="Minimum intron length used by HMM state topology (default:%(default)s).",
    )
    parser.add_argument(
        "--min_cds_length",
        type=int,
        default=DEFAULT_HMM_MIN_CDS_LENGTH,
        help="Minimum total CDS length for emitted genes (default:%(default)s).",
    )
    parser.add_argument(
        "--min_gene_length",
        type=int,
        default=DEFAULT_HMM_MIN_GENE_LENGTH,
        help="Minimum genomic gene span for emitted genes (default:%(default)s).",
    )
    parser.add_argument(
        "--min_gene_score",
        type=float,
        default=DEFAULT_HMM_MIN_GENE_SCORE,
        help="Minimum mean CDS score for emitted genes (default:%(default)s).",
    )
    parser.add_argument(
        "--num_hmm_threads",
        type=int,
        default=8,
        help="Number of CPU workers for HMM decoding (default:8).",
    )
    parser.add_argument(
        "--hmm_splice_event_prob",
        type=float,
        default=DEFAULT_HMM_SPLICE_EVENT_PROB,
        help=(
            "Prior probability for opening one HMM splice event/intron. Smaller values "
            "penalize extra exons more strongly; use 1.0 to disable this penalty "
            "(default:%(default)s)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(verbose=args.verbose)

    start = time.time()
    logger.info(
        "Starting HMM decoding: predictions=%s, output=%s",
        args.chromosome_h5,
        args.output_file,
    )
    logger.info("[Step 1/1]: Running HMM decoding...")

    try:
        output_gff = run_hmm(args)
    except KeyboardInterrupt:
        logger.warning("[Step 1/1]: Cancelled by user")
        raise SystemExit(130) from None
    except Exception:
        logger.exception("[Step 1/1]: Failed")
        raise SystemExit(1) from None

    logger.info(
        "[Step 1/1]: Completed - output=%s, elapsed=%.1fs",
        output_gff,
        time.time() - start,
    )


if __name__ == "__main__":
    main()
