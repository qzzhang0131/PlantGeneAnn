# PlantGeneAnn: Plant Gene Annotator

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

> **Note**: **The latest annotation pipeline only supports the PlantGeneAnn-v1.5-flower-plants model.** If you need to use the PlantGeneAnn-v1.0-model-plants or PlantGeneAnn-v1.0-multi-species models, please download the corresponding previous version of the annotation pipeline from the [Releases](https://github.com/qzzhang0131/PlantGeneAnn/releases) page.

## 📁 Repository Structure
* `run_annotator.py`: one-step annotation script.
* `run_model_prediction.py`: two-step annotation script for model inference.
* `run_prediction_decoding.py`: two-step annotation script for HMM decoding.
* `annotator.py`: Core inference script utilizing [accelerate](https://github.com/huggingface/accelerate) library (bf16 precision).
* `src/`: Functional modules for sequence processing, sequence tokenization, and HMM decoding.

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

You can use PlantGeneAnn in two ways: directly using the [transformers](https://github.com/huggingface/transformers) library for model inference and obtaining embeddings, or running the complete annotation pipeline script to generate standard GFF/GFF3 annotation files.

### 1. Direct Model Inference
You can retrieve both genomic feature probabilities and sequence embeddings using the following snippet:

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
repo_id = "qzzhang/PlantGeneAnn-v1.5-flower-plants"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# The number of DNA tokens (excluding the [CLS] and [SEP] token) needs to be divisible by 16 
# as required by the U-Net downsampling blocks. 
sequences = ["AATTCCGGAA"*4096,"AAAATTTTCC"*4096] 

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
    outs = model(input_ids=tokenized_sequences.to("cuda"), return_dict=True)

# Obtain the logits over the genomic features
# Shape: [batch, sequence_length - 2 * flank_length, 2 * num_features]
logits = outs.logits

# Get the sequence embeddings
# Shape: [batch, sequence_length - 2 * flank_length, 512]
hidden_states = outs.hidden_states.detach()
```

### 2. Complete Annotation Pipeline (Recommended)
The PlantGeneAnn annotation pipeline integrates genome foundation model inference with phase-aware HMM decoding, enabling ab initio genome annotation through the following two approaches:
### 2.1 One-step Annotation:
* `-i`: The genome FA/FASTA/FNA file to be annotated.
* `-m`: Specify the path to the PlantGeneAnn model (downloaded weights from HuggingFace above).
* `-o`: Specify the output GFF/GFF3 file.
* `--batch_size`: Batch size depending on GPU memory (default: 16).
* `--num_tokenize_threads`: Number of CPU threads used to tokenize the sequences (default: 8).
* `--num_hmm_threads`: Number of CPU threads for HMM parallel decoding. (default: 8).
* `--min_chromosome_size`: Minimum chromosome length (bp). Shorter records are skipped (default: 500,000).

```bash
python run_annotator.py \
    -i ./examples/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa \
    -m ./PlantGeneAnn-v1.5-flower-plants \
    -o ./examples/Alyrata_GeneAnn.gff3 \
```

### 2.2 Two-step Annotation:
**2.2.1 Step-1 PlantGeneAnn Model Inference:**
* `-i`: The genome FA/FASTA/FNA file to be annotated.
* `-m`: Specify the path to the PlantGeneAnn model (downloaded weights from HuggingFace above).
* `--cache_path`: Specify the path to use for both the cache and the chromosome level HDF5 file.
* `--batch_size`: Batch size depending on GPU memory (default: 16).
* `--num_tokenize_threads`: Number of CPU threads used to tokenize the sequences (default: 8).
* `--min_chromosome_size`: Minimum chromosome length (bp). Shorter records are skipped (default: 500,000).
```bash
python run_model_prediction.py \
    -i ./examples/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa \
    -m ./PlantGeneAnn-v1.5-flower-plants \
    --cache_path ./examples/Alyrata_cache \
```

**2.2.2 Step-2 HMM Decoding:**
* `-i`: The genome FA/FASTA/FNA file to be annotated.
* `--chromosome_h5`: Path to chromosome level PlantGeneAnn probability HDF5.
* `-o`: Specify the output GFF/GFF3 file.
* `--num_hmm_threads`: Number of CPU threads for HMM parallel decoding. (default: 8).
```bash
python run_prediction_decoding.py \
    -i ./examples/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa \
    --chromosome_h5 ./examples/Alyrata_cache/chromosome_predictions.h5 \
    -o ./examples/Alyrata_GeneAnn.gff3 \
```

*For a full list of parameters, simply run `python run_annotator.py --help`, `python run_model_prediction.py --help` or `python run_prediction_decoding.py --help`.*

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

