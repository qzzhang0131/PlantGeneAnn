import os
import json
from dataclasses import dataclass
from typing import List

DEFAULT_SEQUENCE_LENGTH = 40960
DEFAULT_FLANK_LENGTH = 5120
DEFAULT_CHUNK_SIZE = 6400
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 8
DEFAULT_MIN_CHROM_LENGTH = 500_000
DEFAULT_INFERENCE_MIXED_PRECISION = "bf16"

DEFAULT_HMM_MIN_INTRON_LENGTH = 20
DEFAULT_HMM_MIN_CDS_LENGTH = 60
DEFAULT_HMM_MIN_GENE_LENGTH = 60
DEFAULT_HMM_MIN_GENE_SCORE = 0.60
DEFAULT_HMM_SPLICE_EVENT_PROB = 0.03

DEFAULT_HMM_SPLICE_MOTIF_PRIOR_STRENGTH = 1.0
DEFAULT_HMM_SPLICE_PAIR_PRIOR_STRENGTH = 1.0
DEFAULT_HMM_STOP_CODON_PRIOR_STRENGTH = 1.0
DEFAULT_HMM_READTHROUGH_PRIOR_STRENGTH = 1.0

DEFAULT_HMM_DONOR_MOTIF_WEIGHTS = {
    "GT": 1.0,
    "GC": 0.5,
    "AT": 0.2,
    "OTHER": 0.01,
}
DEFAULT_HMM_ACCEPTOR_MOTIF_WEIGHTS = {
    "AG": 1.0,
    "AC": 0.2,
    "OTHER": 0.01,
}
DEFAULT_HMM_SPLICE_PAIR_WEIGHTS = {
    ("GT", "AG"): 1.0,
    ("GC", "AG"): 1.0,
    ("AT", "AC"): 1.0,
    "OTHER": 0.01,
}
DEFAULT_HMM_STOP_CODON_WEIGHTS = {
    "TAA": 1.0,
    "TAG": 1.0,
    "TGA": 1.0,
    "OTHER": 0.01,
}
DEFAULT_HMM_READTHROUGH_CODON_WEIGHTS = {
    "TAA": 0.01,
    "TAG": 0.01,
    "TGA": 0.01,
    "OTHER": 1.0,
}


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
    # Inference parameters
    batch_size: int
    num_workers: int
    num_tokenize_proc: int
    
    # Genomic-record filtering parameters
    min_chrom_length: int
    exclude_patterns: List[str] = None
    
    def __post_init__(self):
        """Validate and initialize configuration"""
        # Validate input file and model path exist
        if not os.path.exists(self.input_fasta):
            raise FileNotFoundError(f"Input FASTA file not found: {self.input_fasta}.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}.")
        
        # Set default exclusion patterns
        if self.exclude_patterns is None:
            self.exclude_patterns = ["random", "Un", "alt", "hap"]
        
        # Validate parameters
        if self.sequence_length <= 0:
            raise ValueError("Sliding window size must be positive.")
        if self.flank_length >= self.sequence_length:
            raise ValueError("Flank window size must be smaller than sliding window size.")
        if self.flank_length < 0:
            raise ValueError("Flank window size cannot be negative.")
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")
        if self.min_chrom_length <= 0:
            raise ValueError("Minimum chromosome length must be positive.")
    
    @classmethod
    def from_json(cls, config_path: str):
        """Load configuration from a JSON file and return a PipelineConfig instance"""
        with open(config_path, "r") as config_file:
            config_data = json.load(config_file)
            return cls(**config_data)
