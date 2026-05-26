import argparse
import os
import shutil
import subprocess
from datasets import config
from src.configuration import PipelineConfig
from src.sequence_extractor import SequenceExtractor
from src.sequence_tokenizer import SequenceTokenizer
from src.bigwig_gff_writer import BigWigwriter, GFFwriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(description="PlantGenoANN annotation pipeline")
    parser.add_argument("-i","--genome_file", required=True, help="The genome FA/FNA file to be predicted.")
    parser.add_argument("-s","--species", required=True, help="The species name to be predicted.")
    parser.add_argument("-m","--model_path", required=True,
                        help="Specify the path to the prediction model.")
    parser.add_argument("-o","--output_file", required=True, help="Output file name.")
    parser.add_argument("-f","--output_format", required=True, default="bigwig", help="Choose to write predictions to bigwig files or a standard gff file (default:bigwig).")
    parser.add_argument("--chunk_size", type=int, default=3200, 
                        help="The size of the chunks processed by annotator model (default:3,200).")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size depending on GPU memory (default:8).")
    parser.add_argument("--num_tokenize_threads", type=int, default=16, 
                        help="Number of CPU cores used to tokenize the sequence (default:16).")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="The number of CPU cores to load data in parallel (default:8).")
    parser.add_argument("--cache_path", type=str, default="auto",
                        help='Specify the path to cache (default:auto).')
    parser.add_argument("--sliding_window_size", type=int, default=32768, 
                        help="The length of the sliding window used to segment the chromosome (default:32,768).")
    parser.add_argument("--flank_window_size", type=int, default=4096, 
                        help="The flank window length between two consecutive sliding windows (default:4,096).")
    parser.add_argument("--min_chromosome_size", type=int, default=1000000, 
                        help="Minimum chromosome size (bp) for annotating. The size below this value will not be annotate in given FA/FNA/FASTA files (default:1,000,000).")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Gff output format only: The minimum threshold of probability when judging whether a nucleotide is valid (default:0.50).")
    parser.add_argument("--min_gene_length", type=int, default=50,
                        help="Gff output format only: The shortest gene length. Gene lengths below this value will be filtered out (default:50).")
    parser.add_argument("--min_intron_length", type=int, default=9,
                        help="Gff output format only: The shortest intron length. Intron lengths below this value will be filtered out (default:9).")
    parser.add_argument("--min_cds_length", type=int, default=9,
                        help="Gff output format only: The shortest CDS length. CDS lengths below this value will be filtered out (default:9).")
    parser.add_argument("--min_gene_conf_score", type=float, default=0.60, help="Gff output format only: The lowest gene confidence score (default:0.60).")
    parser.add_argument("--min_intron_conf_score", type=float, default=0.70, help="Gff output format only: The lowest intron confidence score (default:0.70).")
    parser.add_argument("--min_cds_conf_score", type=float, default=0.70, help="Gff output format only: The lowest CDS confidence score (default:0.70).")
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    annotator_script = os.path.join(BASE_DIR, "annotator.py")
    if args.cache_path == "auto":
        cache_path = os.path.join(BASE_DIR, "tmp")
        os.makedirs(cache_path, exist_ok=True)
        datasets_dir = os.path.join(cache_path, "datasets")
    else:
        os.makedirs(args.cache_path, exist_ok=True)
        cache_path = args.cache_path
        datasets_dir = os.path.join(args.cache_path, "datasets")
    config.HF_DATASETS_CACHE = datasets_dir
    
    annotator_config = PipelineConfig(
        input_fasta = args.genome_file,
        species = args.species,
        output_file = args.output_file,
        output_format = args.output_format,
        model_path = args.model_path,
        cache_path = cache_path,
        sequence_length = args.sliding_window_size,
        overlap_offset = args.flank_window_size * 2,
        chunk_size = args.chunk_size,
        threshold = args.threshold,
        min_gene_length = args.min_gene_length,
        min_intron_length = args.min_intron_length,
        min_cds_length = args.min_cds_length,
        min_gene_conf_score = args.min_gene_conf_score,
        min_intron_conf_score = args.min_intron_conf_score,
        min_cds_conf_score = args.min_cds_conf_score,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        num_tokenize_proc = args.num_tokenize_threads,
        min_chrom_length = args.min_chromosome_size,
    )

    """Execute the complete genome annotation pipeline"""
    print("=" * 60)
    print("Starting PlantGeneAnn Prediction Pipeline....")
    print("=" * 60)
    try:
        # Step 1: Sequence Extraction
        # print("1. Extracting sliding windows...")
        sequence_extractor = SequenceExtractor(annotator_config)
        chrom_sequence_info, num_chunks = sequence_extractor.process()

        # Step 2: Sequence Tokenization
        # print("\n2. Tokenizing sliding windows...")
        sequence_tokenizer = SequenceTokenizer(annotator_config)
        sequence_tokenizer.process(num_chunks)
        if os.path.exists(datasets_dir):
            shutil.rmtree(datasets_dir)

        # Step 3: Genome Annotation
        # print("\n3. Performing model inference...")
        subprocess.run([
            "python", 
            "-m", 
            "accelerate.commands.launch",
            annotator_script,
            "--model_path", args.model_path, 
            "--cache_path", cache_path,
            "--num_chunks", f"{num_chunks}",
            "--batch_size", f"{args.batch_size}",
            "--num_workers", f"{args.num_workers}"
        ], check=True)
        print(f"Model inference completed.")
        
        # print("\n4. Writing predictions to a GFF3 file...")
        if args.output_format == "gff":
            gff_writer = GFFwriter(annotator_config)
            gff_writer.process(chrom_sequence_info)
        else:
            bigwig_writer = BigWigwriter(annotator_config)
            bigwig_writer.process(chrom_sequence_info)

        # Step 4: Cleanup intermediate files
        try:
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
        except Exception as e:
            print(f"Warning: Could not clean {cache_path}: {e}")

        print("=" * 60)
        print("Pipeline Completed Successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise
    
    return
    
if __name__ == '__main__':
    main()
    
