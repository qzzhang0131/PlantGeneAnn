import os
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

class PlantGeneAnnISMEngine:
    def __init__(
            self, 
            fasta_file: str, 
            repo_id: str, 
            output_path: str, 
            batch_size: int = 24, 
            seq_len: int = 49152, 
            slice_len: int = 32768
        ):
        self.fasta_file = fasta_file
        self.repo_id = repo_id
        self.output_path = output_path
        self.batch_size = batch_size
        
        self.seq_len = seq_len
        self.slice_len = slice_len
        
        self.window_left = self.seq_len // 2
        self.window_right = self.seq_len // 2 - 1
        self.center_idx = self.window_left 
        
        self.slice_left = self.slice_len // 2
        self.slice_right = self.slice_len // 2 - 1
        
        self.accelerator = Accelerator()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            self.repo_id, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        model.eval()
        self.model = self.accelerator.prepare(model)
        self.genome = SeqIO.index(self.fasta_file, "fasta")
        
        self.rule1 = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        self.rule2 = {'A': 'C', 'T': 'G', 'C': 'A', 'G': 'T'}
        self.rule3 = {'A': 'G', 'T': 'C', 'C': 'T', 'G': 'A'}

    def process_region(self, chrom: str, start_pos: int, end_pos: int, strand: str):
        self.accelerator.wait_for_everyone()
        
        seq_dict = self._extract_and_mutate(chrom, start_pos, end_pos)
        logits_49k_dict = self._inference(seq_dict)
        sliced_logits_dict = self._slice(logits_49k_dict)
        ism_score = self._calculate_ism_score(sliced_logits_dict, strand)

        if self.accelerator.is_main_process:   
            np.save(self.output_path, ism_score)
            print(f"ISM score has been saved in {self.output_path}")

        self.accelerator.wait_for_everyone()

    def _extract_and_mutate(self, chrom, start_pos, end_pos):
        if chrom not in self.genome: 
            raise ValueError(f"Chromosome {chrom} not found in reference.")
            
        chrom_seq = str(self.genome[chrom].seq).upper()
        chrom_len = len(chrom_seq)
        
        seqs = {"ref": [], "rule1": [], "rule2": [], "rule3": []}
        
        for pos in range(start_pos, end_pos + 1):
            zero_based_pos = pos - 1 
            left_bound, right_bound = zero_based_pos - self.window_left, zero_based_pos + self.window_right + 1
            pad_left = abs(left_bound) if left_bound < 0 else 0
            pad_right = right_bound - chrom_len if right_bound > chrom_len else 0
            capped_start, capped_end = max(0, left_bound), min(right_bound, chrom_len)
            
            ref_seq = ("N" * pad_left) + chrom_seq[capped_start:capped_end] + ("N" * pad_right)
            
            original_base = ref_seq[self.center_idx]
            m1 = self.rule1.get(original_base, original_base)
            m2 = self.rule2.get(original_base, original_base)
            m3 = self.rule3.get(original_base, original_base)
            
            seqs["ref"].append(ref_seq)
            seqs["rule1"].append(ref_seq[:self.center_idx] + m1 + ref_seq[self.center_idx+1:])
            seqs["rule2"].append(ref_seq[:self.center_idx] + m2 + ref_seq[self.center_idx+1:])
            seqs["rule3"].append(ref_seq[:self.center_idx] + m3 + ref_seq[self.center_idx+1:])
            
        return seqs

    def _inference(self, seq_dict):
        logits_out = {}
        for rule_type, seq_list in seq_dict.items():
            tokens = self.tokenizer(seq_list, return_tensors="pt", padding="longest")["input_ids"]
            dataloader = self.accelerator.prepare(DataLoader(TensorDataset(tokens), batch_size=self.batch_size, shuffle=False))
            
            all_logits = []
            if self.accelerator.is_local_main_process:
                pbar = tqdm(total=len(dataloader), desc=f"Inference [{rule_type.upper()}]")
                
            with torch.no_grad():
                for batch in dataloader:
                    filtered_logits = self.model(input_ids=batch[0]).logits[..., [6, 7, 8, 9]] 
                    gathered_logits = self.accelerator.gather_for_metrics(filtered_logits)
                    all_logits.append(gathered_logits.to(torch.float32).cpu().numpy())
                    if self.accelerator.is_local_main_process: 
                        pbar.update(1)
            
            if self.accelerator.is_local_main_process: 
                pbar.close()
                
            if self.accelerator.is_main_process: 
                logits_out[rule_type] = np.concatenate(all_logits, axis=0)
                
        return logits_out

    def _slice(self, logits_dict):
        if not self.accelerator.is_main_process: 
            return None
            
        start_idx, end_idx = self.center_idx - self.slice_left, self.center_idx + self.slice_right + 1
        sliced_dict = {}
        for rule_type, arr in logits_dict.items():
            sliced_dict[rule_type] = arr[:, start_idx:end_idx, :]
            
        return sliced_dict

    def _calculate_ism_score(self, sliced_dict, strand):
        if not self.accelerator.is_main_process: 
            return
            
        ref_arr = sliced_dict["ref"]
        rule1_arr = sliced_dict["rule1"]
        rule2_arr = sliced_dict["rule2"]
        rule3_arr = sliced_dict["rule3"]
        
        mut_1 = np.sum(ref_arr - rule1_arr, axis=1)
        mut_2 = np.sum(ref_arr - rule2_arr, axis=1)
        mut_3 = np.sum(ref_arr - rule3_arr, axis=1)
        mut_avg = (mut_1 + mut_2 + mut_3) / 3.0
        
        if strand == "+":
            valid_tracks = mut_avg[:, [0, 2]] 
            ism_score = 0.5 * np.sum(valid_tracks, axis=1)
        else:
            valid_tracks = mut_avg[:, [1, 3]] 
            ism_score = 0.5 * np.sum(valid_tracks, axis=1)
            ism_score = ism_score[::-1]

        return ism_score
