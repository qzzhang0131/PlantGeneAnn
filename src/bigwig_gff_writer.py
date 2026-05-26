import os
import re
import h5py
import pyBigWig
import numpy as np
from typing import Union, List, Dict, Any, Optional
from .utils.file_utils import get_valid_chromosomes, get_one_chromosome
from .utils.write_gff_utils import genoann_to_gff
from .utils.prediction_utils import (
    return_first_other_last_predictions,
    return_first_last_predictions,
    combined_overlap_predictions,
    combined_predictions
)

def generate_final_predictions(
    chrom_length: int,
    chrom_num_sequence: int, 
    sequence_length: int,
    overlap_offset: int,
    chrom_prediction: np.ndarray,
)-> np.ndarray:
    if chrom_num_sequence - 1 == 1:
        first_prediction, last_prediction = return_first_last_predictions(
            chrom_prediction,
            sequence_length,
            overlap_offset,
        )
        final_preds = combined_predictions(
            first_prediction,
            last_prediction,
            chrom_length,
        )
    else:
        first_prediction, other_predictions, last_prediction = return_first_other_last_predictions(
            chrom_prediction,
            sequence_length,
            overlap_offset,
        )
        final_preds = combined_overlap_predictions(
            first_prediction,
            last_prediction,
            other_predictions,
            chrom_num_sequence - 1,
            chrom_length,
        )

    return final_preds

class ChunkedH5Reader:
    """
    A reader for HDF5 files containing multiple chunks of data with shape (n, 6, L).
    Supports cross-chunk reading with automatic indexing.
    """
    
    def __init__(self, h5_file_path: str, chunk_ids: Optional[List[str]] = None):
        """
        Initialize the chunked HDF5 reader.
        
        Args:
            h5_file_path: Path to the HDF5 file
            chunk_ids: List of chunk IDs to read. If None, reads all chunks in the file.
        """
        self.h5_file_path = h5_file_path
        self.file = None  # File handle for context manager
        self.chunk_ids = chunk_ids
        self.chunk_info = self._build_index()
        self.total_n = sum(info['n'] for info in self.chunk_info.values())
        self._verify_consistency()
    
    def _extract_chunk_number(self, chunk_id: str) -> int:
        """
        Extract the numeric part from a chunk ID in format chunk_{k}.
        
        Args:
            chunk_id: Chunk ID string
            
        Returns:
            Extracted integer k, or -1 if extraction fails
        """
        patterns = [
            r'chunk_(\d+)',      # chunk_123
            r'chunk(\d+)',       # chunk123
            r'chunk_(\d+).*',    # chunk_123_suffix
            r'.*chunk_(\d+)',    # prefix_chunk_123
        ]
        
        for pattern in patterns:
            match = re.match(pattern, chunk_id)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        # Fallback: extract all digits
        try:
            numbers = re.findall(r'\d+', chunk_id)
            if numbers:
                return int(numbers[0])
        except:
            pass
        
        return -1  # Could not extract number
    
    def _verify_consistency(self) -> None:
        """Verify that all chunks have consistent dimensions."""
        if not self.chunk_info:
            return
        
        # Use first chunk as reference
        first_info = next(iter(self.chunk_info.values()))
        L_ref = first_info['shape'][2]
        
        for chunk_id, info in self.chunk_info.items():
            shape = info['shape']
            if len(shape) != 3:
                raise ValueError(f"Chunk {chunk_id} has incorrect dimensions. Expected 3, got {len(shape)}")
            if shape[1] != 6:
                raise ValueError(f"Chunk {chunk_id} second dimension should be 6, got {shape[1]}")
            if shape[2] != L_ref:
                raise ValueError(f"Chunk {chunk_id} L dimension ({shape[2]}) doesn't match reference L ({L_ref})")
    
    def _build_index(self) -> Dict[str, Dict[str, Any]]:
        """Build index of chunks sorted by numeric order of chunk_{k}."""
        with h5py.File(self.h5_file_path, 'r') as f:
            chunk_info = {}
            global_offset = 0
            
            # Determine which chunks to process
            if self.chunk_ids is None:
                target_chunks = list(f.keys())
            else:
                target_chunks = self.chunk_ids
            
            # Sort chunks by their numeric ID
            target_chunks = sorted(
                target_chunks, 
                key=lambda x: self._extract_chunk_number(x)
            )
            
            for chunk_id in target_chunks:
                if chunk_id not in f:
                    raise KeyError(f"Chunk {chunk_id} not found in file")
                
                chunk = f[chunk_id]
                n_i = chunk.shape[0]
                
                chunk_info[chunk_id] = {
                    'shape': chunk.shape,
                    'global_start': global_offset,
                    'global_end': global_offset + n_i,
                    'n': n_i,
                    'dtype': chunk.dtype,
                    'dataset_path': chunk_id,
                    'chunk_number': self._extract_chunk_number(chunk_id)
                }
                global_offset += n_i
        
        return chunk_info
    
    def __enter__(self) -> 'ChunkedH5Reader':
        """Context manager entry: open the HDF5 file."""
        self.file = h5py.File(self.h5_file_path, 'r')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: close the HDF5 file."""
        if self.file:
            self.file.close()
            self.file = None
    
    def __len__(self) -> int:
        """Return total number of samples across all chunks."""
        return self.total_n
    
    def __getitem__(self, key: Union[int, slice, List[int], np.ndarray]) -> np.ndarray:
        """
        Get data using various indexing methods.
        
        Args:
            key: Integer, slice, list of indices, or numpy array of indices
            
        Returns:
            Requested data as numpy array
        """
        if isinstance(key, int):
            return self._get_single(key)
        elif isinstance(key, slice):
            return self._get_slice(key)
        elif isinstance(key, (list, np.ndarray)):
            return self._get_indices(key)
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")
    
    def _get_single(self, global_idx: int) -> np.ndarray:
        """Get a single sample by global index."""
        # Handle negative indices
        if global_idx < 0:
            global_idx = self.total_n + global_idx
        
        # Validate index range
        if not (0 <= global_idx < self.total_n):
            raise IndexError(f"Index {global_idx} out of range [0, {self.total_n})")
        
        # Find which chunk contains this index
        for chunk_id, info in self.chunk_info.items():
            if info['global_start'] <= global_idx < info['global_end']:
                local_idx = global_idx - info['global_start']
                if self.file:
                    return self.file[chunk_id][local_idx]
                else:
                    with h5py.File(self.h5_file_path, 'r') as f:
                        return f[chunk_id][local_idx]
        
        raise IndexError(f"Could not find index {global_idx}")
    
    def _get_slice(self, slice_obj: slice) -> np.ndarray:
        """Get a slice of data across potentially multiple chunks."""
        start, stop, step = slice_obj.indices(self.total_n)
        
        # Handle empty slices
        if start >= stop:
            if not self.chunk_info:
                raise ValueError("No chunk data available")
            first_info = next(iter(self.chunk_info.values()))
            return np.zeros((0, 6, first_info['shape'][2]), dtype=first_info['dtype'])
        
        # Determine if we should use already opened file
        if self.file:
            f = self.file
            need_close = False
        else:
            f = h5py.File(self.h5_file_path, 'r')
            need_close = True
        
        try:
            results = []
            
            for chunk_id, info in self.chunk_info.items():
                chunk_start = info['global_start']
                chunk_end = info['global_end']
                
                # Calculate overlap between requested slice and this chunk
                overlap_start = max(start, chunk_start)
                overlap_end = min(stop, chunk_end)
                
                if overlap_start < overlap_end:
                    # Convert to chunk-local indices
                    local_start = overlap_start - chunk_start
                    local_end = overlap_end - chunk_start
                    
                    if step == 1:
                        chunk_data = f[chunk_id][local_start:local_end]
                    else:
                        # Apply step size
                        local_indices = np.arange(local_start, local_end, step, dtype=int)
                        chunk_data = f[chunk_id][local_indices]
                    
                    results.append(chunk_data)
            
            if results:
                return np.vstack(results)
            else:
                # Return empty array with correct shape
                first_info = next(iter(self.chunk_info.values()))
                return np.zeros((0, 6, first_info['shape'][2]), dtype=first_info['dtype'])
        
        finally:
            if need_close:
                f.close()
    
    def _get_indices(self, indices: Union[List[int], np.ndarray]) -> np.ndarray:
        """Get multiple non-contiguous indices."""
        indices = np.asarray(indices)
        
        # Handle negative indices
        mask_negative = indices < 0
        indices[mask_negative] = self.total_n + indices[mask_negative]
        
        # Validate index range
        if np.any((indices < 0) | (indices >= self.total_n)):
            raise IndexError("Indices out of range")
        
        # Determine if we should use already opened file
        if self.file:
            f = self.file
            need_close = False
        else:
            f = h5py.File(self.h5_file_path, 'r')
            need_close = True
        
        try:
            # Group indices by chunk
            chunks_data = {}
            for idx in np.unique(indices):
                for chunk_id, info in self.chunk_info.items():
                    if info['global_start'] <= idx < info['global_end']:
                        if chunk_id not in chunks_data:
                            chunks_data[chunk_id] = []
                        chunks_data[chunk_id].append(idx - info['global_start'])
                        break
            
            # Read data from each chunk
            results = []
            for chunk_id, local_indices in chunks_data.items():
                chunk_data = f[chunk_id][local_indices]
                results.append(chunk_data)
            
            # Reorder to match original index order
            if results:
                combined = np.vstack(results)
                # Reorder to original index sequence
                sorted_indices = np.argsort(indices)
                unsorted_indices = np.argsort(sorted_indices)
                return combined[unsorted_indices]
            else:
                first_info = next(iter(self.chunk_info.values()))
                return np.zeros((0, 6, first_info['shape'][2]), dtype=first_info['dtype'])
        
        finally:
            if need_close:
                f.close()
    
    def iter_chunks(self):
        """Iterate over all chunks in numeric order."""
        if self.file:
            f = self.file
            need_close = False
        else:
            f = h5py.File(self.h5_file_path, 'r')
            need_close = True
        
        try:
            for chunk_id in self.chunk_info.keys():
                yield f[chunk_id][:]
        finally:
            if need_close:
                f.close()
    
    def read_all(self) -> np.ndarray:
        """Read all data from all chunks."""
        return np.vstack(list(self.iter_chunks()))
    
    def get_chunk_info(self) -> Dict[str, Dict]:
        """Get information about all chunks."""
        return self.chunk_info.copy()
    
    def get_shape(self) -> tuple:
        """Get the combined shape of all data."""
        if not self.chunk_info:
            return (0, 6, 0)
        
        first_info = next(iter(self.chunk_info.values()))
        L = first_info['shape'][2]
        return (self.total_n, 6, L)
    
    def close(self) -> None:
        """Manually close the file if opened."""
        if self.file:
            self.file.close()
            self.file = None

class GFFwriter:
    """Write from model predictions to a GFF3 file"""
    
    def __init__(self, config):
        self.config = config
        self.h5_predictions_path = os.path.join(self.config.cache_path, "model_predictions.h5")
        if not os.path.exists(self.h5_predictions_path):
            raise FileNotFoundError(f"Model predictions H5 file not found: {self.h5_predictions_path}")
    
    def process(self, chrom_sequence_info):
        valid_ids = get_valid_chromosomes(
            self.config.input_fasta,
            self.config.min_chrom_length,
            self.config.exclude_patterns,
            getattr(self.config, 'include_patterns', None)
        )
        output_file = os.path.join(self.config.output_file, f"{self.config.species}_plantgenoann_predictions.gff3")
        with open(output_file, "w") as f:
            f.write("##gff-version 3\n")
            with ChunkedH5Reader(self.h5_predictions_path) as h5reader:
                current_idx = 0
                for chrom_id in valid_ids:
                    chrom_length, chrom_num_sequence = chrom_sequence_info[chrom_id]
                    chrom_prediction = h5reader[current_idx:current_idx + chrom_num_sequence]
                    final_preds = generate_final_predictions(
                        chrom_length,
                        chrom_num_sequence, 
                        self.config.sequence_length,
                        self.config.overlap_offset,
                        chrom_prediction
                    )
                    gff_units = genoann_to_gff(
                        probs=final_preds,
                        seqid=chrom_id,
                        sequence=get_one_chromosome(self.config.input_fasta, chrom_id),
                        threshold=self.config.threshold,
                        min_gene_len=self.config.min_gene_length,
                        min_intron_len=self.config.min_intron_length,
                        min_cds_len=self.config.min_cds_length,
                        min_gene_conf=self.config.min_gene_conf_score,
                        min_intron_conf=self.config.min_intron_conf_score,
                        min_cds_conf=self.config.min_cds_conf_score,
                        source="PlantGenoANN",
                    )
                    if not gff_units:
                        continue
                    else:
                        for unit in gff_units:
                            for line in unit["lines"]:
                                f.write(line + "\n")
                    current_idx += chrom_num_sequence
            h5reader.close()
        f.close()
        print("Writing predictions to a GFF3 file completed.")
        
        return

class BigWigwriter:
    """Write from model predictions to a BigWig file"""
    
    def __init__(self, config):
        self.config = config
        self.h5_predictions_path = os.path.join(self.config.cache_path, "model_predictions.h5")
        if not os.path.exists(self.h5_predictions_path):
            raise FileNotFoundError(f"Model predictions H5 file not found: {self.h5_predictions_path}")

    def clip_signals_to_bigwig(
        self,
        signal: np.ndarray,
        chrom: str,
        chrom_length: int,
        floor_threshold: float,
        ceiling_threshold: float,
        bw_files,
    ) -> None:
        L = signal.shape[1]

        signal = signal.astype(np.float32)

        for track_idx in range(6):
            bw = bw_files[track_idx]

            track = signal[track_idx, :].copy()
            track[track < floor_threshold] = 0.0
            track[track > ceiling_threshold] = 1.0

            if np.all(track == 0.0):
                bw.addEntries([chrom], [0], ends=[chrom_length], values=[0.0])
                continue

            starts = []
            ends = []
            values = []
            start = 0
            val = track[0]

            for i in range(1, L):
                if track[i] != val:
                    starts.append(start)
                    ends.append(i)
                    values.append(float(val))
                    start = i
                    val = track[i]

            starts.append(start)
            ends.append(L)
            values.append(float(val))

            bw.addEntries([chrom] * len(starts), starts, ends=ends, values=values)
    
    def process(self, chrom_sequence_info):
        valid_ids = get_valid_chromosomes(
            self.config.input_fasta,
            self.config.min_chrom_length,
            self.config.exclude_patterns,
            getattr(self.config, 'include_patterns', None)
        )
        track_idx_map = {
            0:"gene_in_positive_strand",
            1:"gene_in_negative_strand",
            2:"exon_in_positive_strand",
            3:"exon_in_negative_strand",
            4:"CDS_in_positive_strand",
            5:"CDS_in_negative_strand",
        }
        bw_files = []
        for track_idx in range(6):
            bw_path = os.path.join(self.config.output_file, f"{self.config.species}_{track_idx_map[track_idx]}.bw")
            bw = pyBigWig.open(bw_path, "w")
            
            # Key point: Write header for ALL chromosomes at once
            header = [(chrom, chrom_sequence_info[chrom][0]) for chrom in valid_ids]
            bw.addHeader(header)
            
            bw_files.append(bw)

        with ChunkedH5Reader(self.h5_predictions_path) as h5reader:
            current_idx = 0
            for chrom_id in valid_ids:
                chrom_length, chrom_num_sequence = chrom_sequence_info[chrom_id]
                chrom_prediction = h5reader[current_idx:current_idx + chrom_num_sequence]
                final_preds = generate_final_predictions(
                    chrom_length,
                    chrom_num_sequence, 
                    self.config.sequence_length,
                    self.config.overlap_offset,
                    chrom_prediction
                )
                self.clip_signals_to_bigwig(
                    final_preds,
                    chrom_id,
                    chrom_length,
                    0.10,
                    0.90,
                    bw_files
                )
                current_idx += chrom_num_sequence
        h5reader.close()

        for bw in bw_files:
            bw.close()

        print("Writing predictions to BigWig files completed.")
        
        return