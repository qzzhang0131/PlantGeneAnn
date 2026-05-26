import os
import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PipelineConfig:
    """Configuration parameters for genome annotation pipeline"""
    
    # Input file paths
    input_fasta: str
    species: str
    output_file: str
    output_format: str
    model_path: str
    cache_path: str
    
    # Sequence processing parameters
    sequence_length: int
    overlap_offset: int
    chunk_size: int
    threshold: float
    min_gene_length: int
    min_intron_length: int
    min_cds_length: int
    min_gene_conf_score: float
    min_intron_conf_score: float
    min_cds_conf_score: float
    
    # Inference parameters
    batch_size: int
    num_workers: int
    num_tokenize_proc: int
    
    # Chromosome filtering parameters
    min_chrom_length: int
    exclude_patterns: List[str] = None
    
    def __post_init__(self):
        """Validate and initialize configuration"""
        # Validate input file and model path exist
        if not os.path.exists(self.input_fasta):
            raise FileNotFoundError(f"Input FASTA file not found: {self.input_fasta}.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}.")
        if not os.path.isdir(self.output_file):
            raise ValueError("Output file must be a directory not a file.")
        
        # Set default exclusion patterns
        if self.exclude_patterns is None:
            self.exclude_patterns = ["random", "Un", "alt", "hap"]
        
        # Validate parameters
        if self.output_format not in {"bigwig","gff"}:
            raise ValueError("Output format must be in 'bigwig' or 'gff'.")
        if self.sequence_length <= 0:
            raise ValueError("Sliding window size must be positive.")
        if self.overlap_offset >= self.sequence_length:
            raise ValueError("Overlap window size must be smaller than sliding window size.")
        if self.overlap_offset < 0:
            raise ValueError("Overlap window size cannot be negative.")
        if self.threshold <= 0.0 or self.threshold >=1.0:
            raise ValueError("Threshold must between 0 to 1.")
        if self.chunk_size <= 0:
            raise ValueError("Chunk size cannot be negative.")
        if self.min_chrom_length <= self.sequence_length:
            raise ValueError("Minimum chromosome length must be at least sequence length.")
    
    @classmethod
    def from_json(cls, config_path: str):
        """Load configuration from a JSON file and return a PipelineConfig instance"""
        with open(config_path, "r") as config_file:
            config_data = json.load(config_file)
            return cls(**config_data)
