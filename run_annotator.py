import argparse
import logging
import os
import shutil
import sys
import tempfile
import time

from datasets import config
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
    DEFAULT_HMM_MIN_CDS_LENGTH,
    DEFAULT_HMM_MIN_GENE_LENGTH,
    DEFAULT_HMM_MIN_GENE_SCORE,
    DEFAULT_HMM_MIN_INTRON_LENGTH,
    DEFAULT_HMM_SPLICE_EVENT_PROB,
    DEFAULT_INFERENCE_MIXED_PRECISION,
    DEFAULT_MIN_CHROM_LENGTH,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEQUENCE_LENGTH,
    PipelineConfig,
)
from src.logging_config import setup_logger
from src.sequence_extractor import SequenceExtractor
from src.sequence_tokenizer import SequenceTokenizer
from src.hmm_decoder import HMMDecoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("PlantGeneAnn")

INTERMEDIATE_NAMES = (
    "chromosome_predictions.h5",
    "chromosome_predictions.h5.tmp",
)


def _cleanup_intermediates(cache_path: str) -> None:
    """Remove intermediate pipeline files inside *cache_path*.

    The shared runtime-cache helper removes ``huggingface/``, ``datasets/``,
    ``shards/``, tokenized ``chunk_N`` datasets, and pre-tokenization TSV
    files. This function additionally removes one-step prediction HDF5 files.
    The user-specified cache directory itself is preserved.
    """

    cleanup_prediction_runtime_cache(cache_path, keep_datasets=False)

    for name in INTERMEDIATE_NAMES:
        path = os.path.join(cache_path, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
        except OSError as e:
            logger.warning("Could not remove %s: %s", path, e)


def _run_pipeline(args, cache_path: str, annotator_script: str) -> None:
    """Execute the four-step direct-write annotation pipeline."""

    datasets_dir = os.path.join(cache_path, "datasets")
    config.HF_DATASETS_CACHE = datasets_dir

    hf_cache_root = os.path.join(cache_path, "huggingface")
    os.environ["HF_HOME"] = hf_cache_root
    os.environ["HF_DATASETS_CACHE"] = datasets_dir

    annotator_config = PipelineConfig(
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

    # Step 1: Sequence Extraction
    step_start = time.time()
    logger.info("[Step 1/4]: Extracting sequences from genome...")
    sequence_extractor = SequenceExtractor(annotator_config)
    chrom_sequence_info, num_chunks = sequence_extractor.process()
    total_windows = sum(nw for _, nw in chrom_sequence_info.values())
    logger.info(
        "[Step 1/4]: Completed - records=%d, windows=%d, chunks=%d, elapsed=%.1fs",
        len(chrom_sequence_info),
        total_windows,
        num_chunks,
        time.time() - step_start,
    )

    # Step 2: Sequence Tokenization
    step_start = time.time()
    logger.info("[Step 2/4]: Tokenizing sequences...")
    sequence_tokenizer = SequenceTokenizer(annotator_config)
    sequence_tokenizer.process(num_chunks)
    if os.path.exists(datasets_dir):
        shutil.rmtree(datasets_dir)
    logger.info("[Step 2/4]: Completed - elapsed=%.1fs", time.time() - step_start)

    # Step 3: Multi-GPU inference directly into genomic-record coordinates.
    step_start = time.time()
    num_processes = args.num_processes if args.num_processes is not None else detect_num_processes()
    if num_processes <= 0:
        raise ValueError(f"--num_processes must be positive, got {num_processes}.")
    chromosome_h5_path = os.path.abspath(
        os.path.join(cache_path, "chromosome_predictions.h5")
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
            compression="none",
            hdf5_chunk_bp=(
                args.sliding_window_size - 2 * args.flank_window_size
            ),
        )
        logger.info(
            "[Step 3/4]: Running model inference - %d process(es)",
            num_processes,
        )

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
    logger.info(
        "[Step 3/4]: Completed - elapsed=%.1fs",
        time.time() - step_start,
    )

    # Step 4: Decode the final phase-consistent gene structures and write GFF3.
    step_start = time.time()
    logger.info("[Step 4/4]: Running phase-aware HMM decoding...")
    num_hmm_threads = (
        args.num_hmm_threads
        if args.num_hmm_threads is not None
        else os.cpu_count() or 1
    )
    hmm_decoder = HMMDecoder(
        cache_path=cache_path,
        genome_fasta=args.genome_file,
        output_gff=os.path.abspath(args.output_file),
        min_intron_length=args.min_intron_length,
        min_cds_length=args.min_cds_length,
        min_gene_length=args.min_gene_length,
        min_gene_score=args.min_gene_score,
        splice_event_prob=args.hmm_splice_event_prob,
        num_threads=num_hmm_threads,
    )
    hmm_decoder.process()
    logger.info(
        "[Step 4/4]: Completed - elapsed=%.1fs",
        time.time() - step_start,
    )


def main():
    parser = argparse.ArgumentParser(description="PlantGeneAnn annotation pipeline")
    parser.add_argument("-i", "--genome_file", required=True, help="The genome FA/FNA/FASTA file to be predicted.")
    parser.add_argument("-m", "--model_path", required=True,
                        help="Specify the path to the prediction model.")
    parser.add_argument("-o", "--output_file", required=True,
                        help="Output GFF3 file path (must end with .gff or .gff3).")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="The size of the chunks processed by annotator model (default:%(default)s).")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size depending on GPU memory (default:%(default)s).")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="Number of local Accelerate worker processes. If omitted, automatically uses all GPUs visible to the current process.")
    parser.add_argument("--num_tokenize_threads", type=int, default=8,
                        help="Number of CPU cores used to tokenize the sequence (default:8).")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="The number of CPU cores used by DataLoader to load data in parallel (default:%(default)s).")
    parser.add_argument("--cache_path", type=str, default="auto",
                        help="Specify the path to cache (default:auto).")
    parser.add_argument("--sliding_window_size", type=int, default=DEFAULT_SEQUENCE_LENGTH,
                        help="Model input-window length used to segment each genomic record (default:%(default)s).")
    parser.add_argument("--flank_window_size", type=int, default=DEFAULT_FLANK_LENGTH,
                        help="Context length on each side of the center prediction region (default:%(default)s).")
    parser.add_argument("--min_chromosome_size", type=int, default=DEFAULT_MIN_CHROM_LENGTH,
                        help="Minimum genomic-record length (bp). Shorter records are skipped (default:%(default)s).")
    parser.add_argument("--min_gene_length", type=int, default=DEFAULT_HMM_MIN_GENE_LENGTH,
                        help="The shortest gene length. Gene lengths below this value will be filtered out (default:%(default)s).")
    parser.add_argument("--min_gene_score", type=float, default=DEFAULT_HMM_MIN_GENE_SCORE,
                        help="The lowest gene score (default:%(default)s).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging output.")

    # HMM decoding options
    parser.add_argument("--min_intron_length", type=int, default=DEFAULT_HMM_MIN_INTRON_LENGTH,
                        help="Minimum intron length used by HMM state topology (default:%(default)s).")
    parser.add_argument("--min_cds_length", type=int, default=DEFAULT_HMM_MIN_CDS_LENGTH,
                        help="Minimum total CDS length for emitted HMM genes (default:%(default)s).")
    parser.add_argument("--num_hmm_threads", type=int, default=8,
                        help="Number of CPU threads for HMM parallel decoding. "
                        "If omitted, uses all available CPU cores.")    
    parser.add_argument("--hmm_splice_event_prob", type=float, default=DEFAULT_HMM_SPLICE_EVENT_PROB,
                        help="Prior probability for opening one HMM splice event/intron. "
                             "Smaller values penalize extra exons more strongly; use 1.0 to disable "
                             "this penalty (default:%(default)s).")

    args = parser.parse_args()

    # Configure the single console logger after CLI parsing.
    setup_logger(verbose=args.verbose)
    logger = logging.getLogger("PlantGeneAnn")

    output_ext = os.path.splitext(args.output_file)[1].lower()
    if output_ext not in (".gff", ".gff3"):
        parser.error(
            f"Output file must have .gff or .gff3 extension, got: {output_ext!r}"
        )

    output_path = os.path.abspath(args.output_file)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    annotator_script = os.path.join(BASE_DIR, "annotator.py")

    pipeline_start = time.time()

    logger.info(
        "Starting HMM annotation: input=%s, model=%s, output=%s",
        args.genome_file,
        args.model_path,
        output_path,
    )

    try:
        if args.cache_path == "auto":
            tmp_base = os.path.join(BASE_DIR, "tmp")
            os.makedirs(tmp_base, exist_ok=True)
            try:
                with tempfile.TemporaryDirectory(prefix="tmp_", dir=tmp_base) as cache_path:
                    logger.debug("Cache directory: %s (auto, will be removed on exit)", cache_path)
                    _run_pipeline(args, cache_path, annotator_script)
            finally:
                try:
                    os.rmdir(tmp_base)
                except FileNotFoundError:
                    pass
                except OSError:
                    # Another concurrent run may still own a temporary child
                    # directory. Never recursively remove the shared root.
                    logger.debug(
                        "Temporary root is not empty; leaving it in place: %s",
                        tmp_base,
                    )
        else:
            cache_path = os.path.abspath(args.cache_path)
            os.makedirs(cache_path, exist_ok=True)
            logger.debug("Cache directory: %s (user-specified, intermediates will be cleaned)", cache_path)
            try:
                _run_pipeline(args, cache_path, annotator_script)
            finally:
                _cleanup_intermediates(cache_path)

        total_time = time.time() - pipeline_start
        logger.info(
            "Annotation completed: output=%s, elapsed=%.1fs",
            output_path,
            total_time,
        )

    except KeyboardInterrupt:
        logger.warning("Pipeline cancelled by user")
        raise SystemExit(130) from None
    except Exception:
        logger.exception("Pipeline failed")
        raise SystemExit(1) from None

    return


if __name__ == "__main__":
    main()
