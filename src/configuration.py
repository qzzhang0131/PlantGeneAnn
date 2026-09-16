import json
import os
from dataclasses import dataclass
from typing import Any

DEFAULT_SEQUENCE_LENGTH = 40960
DEFAULT_FLANK_LENGTH = 5120
DEFAULT_CHUNK_SIZE = 3200
DEFAULT_BATCH_SIZE = 16
DEFAULT_MIN_CHROM_LENGTH = 40_960
DEFAULT_INFERENCE_MIXED_PRECISION = "bf16"

# Strict 15-state segmental decoder hard constraints. Sequence motifs are not
# weighted and no obsolete soft-prior/rescue thresholds are retained.
DEFAULT_DECODER_MIN_INTRON_LENGTH = 20
DEFAULT_DECODER_MIN_CDS_LENGTH = 60
DEFAULT_DECODER_MIN_MEAN_GENE_LOG_ODDS = 0.50
CANDIDATE_SCAN_MAX_WORKERS = 20


@dataclass
class PipelineConfig:
    """Configuration parameters for genome annotation pipeline"""
    
    # Input file paths
    input_fasta: str
    model_path: str
    cache_path: str
    
    # Sequence processing parameters
    sequence_length: int
    flank_length: int
    chunk_size: int
    # Inference and preprocessing CPU budget
    batch_size: int
    num_cpu_threads: int
    
    # Genomic-record filtering parameters
    min_chrom_length: int

    def __post_init__(self):
        """Validate and initialize configuration"""
        # Validate input file and model path exist
        if not os.path.exists(self.input_fasta):
            raise FileNotFoundError(f"Input FASTA file not found: {self.input_fasta}.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}.")
        
        # Validate local pipeline geometry only. Model/config compatibility is
        # intentionally owned by the user and is not read from HuggingFace.
        if self.sequence_length <= 0:
            raise ValueError("Sliding window size must be positive.")
        if self.flank_length < 0:
            raise ValueError("Flank window size cannot be negative.")
        if 2 * self.flank_length >= self.sequence_length:
            raise ValueError(
                "2 * flank window size must be smaller than the sliding "
                "window size so a non-empty center region remains."
            )
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Inference batch size must be positive.")
        if self.num_cpu_threads <= 0:
            raise ValueError("num_cpu_threads must be positive.")
        if self.min_chrom_length <= 0:
            raise ValueError("Minimum chromosome length must be positive.")

    @property
    def center_length(self) -> int:
        """Return the genomic center interval emitted from each input window."""

        return self.sequence_length - 2 * self.flank_length
    
    @classmethod
    def from_json(cls, config_path: str):
        """Load configuration from a JSON file and return a PipelineConfig instance"""
        with open(config_path, "r") as config_file:
            config_data = json.load(config_file)
            return cls(**config_data)


def build_pipeline_config_from_args(
    args: Any,
    cache_path: str,
    *,
    num_cpu_threads: int,
) -> PipelineConfig:
    """Build the shared config with the pipeline's fixed model geometry."""

    return PipelineConfig(
        input_fasta=args.genome_file,
        model_path=args.model_path,
        cache_path=cache_path,
        sequence_length=DEFAULT_SEQUENCE_LENGTH,
        flank_length=DEFAULT_FLANK_LENGTH,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_cpu_threads=num_cpu_threads,
        min_chrom_length=args.min_chromosome_size,
    )
