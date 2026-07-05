# PlantGeneAnn: Plant Gene Annotation Model

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/qzzhang/PlantGeneAnn)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 📖 Introduction
**PlantGeneAnn** is a genome foundation model that enables accurate **ab initio gene structure annotation** of plant genomes. Built upon the **[PlantBiMoE](https://github.com/HUST-Keep-Lin/PlantBiMoE)** architecture with a 1D U-Net-style embeddings decoder head, it automates the prediction of gene structures—including protein-coding genes, CDSs, and exons—on both forward and reverse strands. Beyond standard annotation, PlantGeneAnn serves as a **long-context plant genome foundation model**, adaptable via fine-tuning to predict diverse omic signal tracks such as RNA-seq and ATAC-seq.


## 🤗 Model Access
The pre-trained weights for **PlantGeneAnn** are hosted on Hugging Face:
|Model Name|Access Link|
| :--- | :--- |
| PlantGeneAnn-v1.5-flower-plants |https://huggingface.co/qzzhang/PlantGeneAnn-v1.5-flower-plants|
| PlantGeneAnn-v1.0-model-plants |https://huggingface.co/qzzhang/PlantGeneAnn-v1.0-model-plants|
| PlantGeneAnn-v1.0-multi-species |https://huggingface.co/qzzhang/PlantGeneAnn-v1.0-multi-species|

> **Note**: **The latest annotation pipeline only supports the v1.5-flower-plants model.** If you need to use the v1.0-model-plants or v1.0-multi-species models, please download the corresponding previous version of the annotation pipeline from the Releases page.

## 📁 Repository Structure
* `run_annotator.py`: Main entry point (extraction, tokenization, inference dispatch).
* `annotator.py`: Core inference script utilizing [accelerate](https://github.com/huggingface/accelerate) library (bf16 precision).
* `src/`: Functional modules for sequence processing, model wrapping, and output files generation.

## ⚙️ Installation & Environment
The model requires the [mamba-ssm](https://github.com/state-spaces/mamba) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) libraries for the core backbone.
```
# 1. Clone repository & create environment
git clone https://github.com/qzzhang0131/PlantGeneAnn.git
cd PlantGeneAnn
conda create -n PlantGeneAnn python=3.8 -y
conda activate PlantGeneAnn

# 2. Install dependencies (Crucial for Triton JIT & CUDA extensions)
conda install -c nvidia -c conda-forge cuda-toolkit=12.1.0 libxcrypt -y
pip install -r requirements.txt

# 3. Compile core CUDA libraries (May take 10-20 minutes)
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
MAX_JOBS=4 pip install causal-conv1d==1.2.0.post2 mamba-ssm==1.2.0.post1 flash-attn==2.5.6 --no-build-isolation
```
---

## 🚀 Quick Start (Usage)

You can use PlantGeneAnn in two ways: directly using the [transformers](https://github.com/huggingface/transformers) library for model inference and obtaining embeddings, or running the complete pipeline script to generate prediction tracks or standard GFF annotation files.

### 1. Direct Model Inference
You can retrieve both genomic feature probabilities and sequence embeddings using the following snippet:

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
repo_id = "qzzhang/PlantGeneAnn-model-plants"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# The number of DNA tokens (excluding the [CLS] and [SEP] token) needs to be divisible by 8 
# as required by the U-Net downsampling blocks. 
sequences = ["ACTAGAGCGAGAGAAA","TTTGAGAGCGCGCGGA"] 

# Tokenize
tokenized_sequences = tokenizer(
    sequences, 
    return_tensors="pt", 
    padding="longest"
)["input_ids"]

# Infer
model.to("cuda")
model.eval()
with torch.no_grad():
    outs = model(input_ids=tokenized_sequences.to("cuda"))

# Obtain the logits over the genomic features
# Shape: [batch, sequence_length, num_features]
logits = outs.logits

# Get probabilities associated with CDS on the forward strand (+)
pos_strand_cds_probs = model.get_feature_logits(feature="CDS", strand="+", logtis=logits).detach()
print(f"CDS probabilities on the forward strand: {pos_strand_cds_probs}")

# Get the sequence embeddings
# Shape: [batch, sequence_length, 1024]
hidden_states = outs.hidden_states.detach()
print(f"Sequence embeddings shape is: {hidden_states.shape}")
```

### 2. Full Prediction Pipeline
To run the full prediction pipeline, use the `run_annotator.py` script. The pipeline will automatically handle sliding windows, multi-GPU model inference, and standard output format.

**🛠️ 2.1 Basic Configuration:**
* `-i`: The genome FA/FNA file to be predicted.
* `-s`: The species name to be predicted.
* `-m`: Specify the path to the prediction model (downloaded weights from HuggingFace above).
* `-o`: Specify the output path.
* `-f`: Choose to write predictions to BigWig files or a standard GFF3 file (default: "bigwig").

**Save Full Prediction Tracks to BigWig Files:**
```bash
python run_annotator.py \
    -i ./example/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa \
    -s Arabidopsis_lyrata \
    -m ./PlantGeneAnn-model-plants \
    -o ./example \
    -f bigwig
```
**Write Prediction Tracks to a Standard GFF3 File:**
```bash
python run_annotator.py \
    -i ./example/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa \
    -s Arabidopsis_lyrata \
    -m ./PlantGeneAnn-model-plants \
    -o ./example \
    -f gff
```

**🛠️ 2.2 Advanced Pipeline Configuration (Optional):**

PlantGeneAnn prediction pipeline is highly customizable. You can adjust sliding windows, confidence thresholds, and hardware utilization to fit your specific needs:

**Hardware & Processing:**
* `--chunk_size`: The size of the chunks processed by annotator model (default: 3,200).
* `--batch_size`: The number of samples in a batch (default: 8).
* `--num_workers`: The number of CPU cores to load data in parallel (default: 8).
* `--num_tokenize_threads`: Number of CPU cores used to tokenize the sequence (default: 16).
* `--cache_path`: Specify the path to cache intermediate datasets (default: "auto").

**Sequence & Window Settings:**
* `--sliding_window_size`: Length of the sliding window used to segment the chromosome (default: 32,768).
* `--flank_window_size`: Flank window length between two consecutive sliding windows (default: 4,096).
* `--min_chromosome_size`: Minimum chromosome size for annotating (default: 1,000,000 (1MB)).

**Filtering & Thresholds (only with gff output format):**
* `--threshold`: Minimum probability threshold for valid nucleotides (default: 0.50).
* `--min_gene_conf_score`: The lowest gene confidence score (default: 0.60).
* `--min_intron_conf_score`: The lowest intron confidence score (default: 0.70).
* `--min_cds_conf_score`: The lowest CDS confidence score (default: 0.70).
* `--min_gene_length`, `--min_intron_length`, `--min_cds_length`: Filter out predicted elements shorter than these values.

*For a full list of parameters, simply run `python run_annotator.py --help`.*

## ⚡ Hardware Requirements

PlantGeneAnn inference **requires NVIDIA GPUs with Ampere architecture or newer** (e.g., RTX 30-series, RTX 40-series, A100, H100, etc.).  

### 1. Inference Time for Different Plant Genomes

The following table shows approximate **single-GPU inference time (hours)** for typical plant genomes:

| Plant Species | Genome Size | RTX 3080Ti | RTX 3090 | RTX 4080 | RTX 4090 |
|---------------|-------------|----------|----------|----------|----------|
| Arabidopsis   | 116MB       | 0.33     | 0.29     | 0.23     | 0.16     |
| Rice          | 364MB       | 1.01     | 0.90     | 0.75     | 0.50     |
| Soybean       | 949MB       | 2.60     | 2.32     | 1.90     | 1.31     |
| Maize         | 2.07GB      | 5.85     | 5.19     | 4.31     | 2.95     |

> **Note**: Actual runtime may vary depending on GPU driver version, system load, and exact hardware configuration. These values are for reference only.

**Multi-GPU Support:** PlantGeneAnn supports multi-GPU parallel inference through [accelerate](https://github.com/huggingface/accelerate) library. With N GPUs, the inference time is approximately 1/N of the single-GPU reference time (near-linear scaling).

### 2. Recommended Batch Size by GPU VRAM

Under default configuration, we recommend the following `batch_size` settings based on GPU VRAM:

| GPU VRAM | 8GB | 16GB | 24GB | 40GB |
|------------|------|------|------|------|
| batch size | 2    | 4    | 8    | 16   |

---
## 📝 Citation

If you use PlantGeneAnn in your research, please cite our preprint:

Zhang, Q., Zhang, Z., Lin, K., Wang, J., Deng, K., Xiang, X., Xu, W., & Hu, X. (2026). PlantGeneAnn: a strand-specific genome foundation model for ab initio gene structure annotation of plant genomes. *bioRxiv*. https://doi.org/10.64898/2026.06.25.733695

```bibtex
@article{zhang2026plantgeneann,
  title={PlantGeneAnn: a strand-specific genome foundation model for ab initio gene structure annotation of plant genomes},
  author={Zhang, Qizhe and Zhang, Zhengyang and Lin, Kepeng and Wang, Jing and Deng, Kaixuan and Xiang, Xianglei and Xu, Wei and Hu, Xuehai},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.06.25.733695},
  url={[https://doi.org/10.64898/2026.06.25.733695](https://doi.org/10.64898/2026.06.25.733695)}
}
```
## 📜 License
See the LICENSE file for details.
## 📧 Contact
Feel free to contact qzzhang@webmail.hzau.edu.cn if you have any questions or suggestions regarding the code and models.

