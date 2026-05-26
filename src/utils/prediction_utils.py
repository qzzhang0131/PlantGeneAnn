import numpy as np

def return_first_last_predictions(all_predictions: np.ndarray, seq_len: int = 16384, 
                            offset_len: int = 0) -> (np.ndarray, np.ndarray):
    hf_offset_len = offset_len // 2
    
    return all_predictions[0, :, :seq_len - hf_offset_len], all_predictions[-1, :, hf_offset_len:]

def return_first_other_last_predictions(all_predictions: np.ndarray, seq_len: int = 16384, 
                            offset_len: int = 0) -> (np.ndarray, np.ndarray, np.ndarray):
    hf_offset_len = offset_len // 2
    
    return all_predictions[0, :, :seq_len - hf_offset_len], all_predictions[1:-1, :, hf_offset_len:seq_len - hf_offset_len], all_predictions[-1, :, hf_offset_len:]

def combined_predictions(first_prediction: np.ndarray, last_prediction: np.ndarray, chrom_len: int) -> np.ndarray:
    first_last_seq_len = first_prediction.shape[-1]
    tail = chrom_len - first_last_seq_len
    predictions = np.zeros((6, chrom_len), dtype=np.float16)
    
    predictions[:, :first_last_seq_len] = first_prediction
    predictions[:, first_last_seq_len:] = last_prediction[:, first_last_seq_len - tail:first_last_seq_len]
    
    return predictions
    
def combined_overlap_predictions(first_prediction: np.ndarray, last_prediction: np.ndarray, other_predictions: np.ndarray,
                num_valid_examples: int, chrom_len: int) -> np.ndarray:
    first_last_seq_len = first_prediction.shape[-1]
    res = other_predictions.shape[-1]
    l = first_last_seq_len + (num_valid_examples - 1) * res
    tail = chrom_len - l
    
    predictions = np.zeros((6, chrom_len), dtype=np.float16)
    predictions[:, :first_last_seq_len] = first_prediction
    
    for k in range(1, num_valid_examples):
        current_len = first_last_seq_len + (k - 1) * res
        predictions[:, current_len:current_len + res] = other_predictions[k - 1, ...]
    
    predictions[:, chrom_len - tail:chrom_len] = last_prediction[:, first_last_seq_len - tail:first_last_seq_len]
    
    return predictions
