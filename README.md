[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2603.07119)
[![Demo](https://img.shields.io/badge/Demo-blue?style=flat-square)](demo/)

# TIQA: Human-Aligned Perceptual Text Quality Assessment in Generated Images
*Kirill Koltsov, Aleksandr Gushchin, Dmitriy Vatolin, Anastasia Antsiferova* 
**TD;DR** A small model that takes horizontal text crops as input and produces scores for their visual quality in terms of generative artifacts on a scale from 0 to 5

**Abstract**

Recent text-to-image models have improved global realism, but text rendering remains a persistent failure mode: images may look convincing overall, yet local typography often contains malformed glyphs, broken strokes, irregular spacing, and other artifacts that heavily penalize humans. We formulate Text-in-Image Quality Assessment (TIQA), a no-reference task that estimates a human-aligned perceptual quality score for detected text regions while disentangling visual text quality from semantic correctness. To support this setting, we introduce two datasets. TIQA-Crops contains 120k text crops from 36k AI-generated images produced by 12 generators, with 10k mean-opinion-score (MOS) labels and 110k proxy labels for pretraining TIQA-Images contains 1,500 text-heavy images from 10 recent generators, including proprietary systems, with paired overall-quality and text-quality subjective scores. We also propose ANTIQA, a lightweight predictor with text-specific inductive biases. Across crop-level and image-level evaluations, ANTIQA achieves the best alignment with human judgments, reaching PLCC/SROCC of 0.942/0.935 on TIQA-Crops and 0.842/0.837 for text-quality MOS on unseen generators in TIQA-Images. In best-of-5 AI-generated image ranking, ANTIQA improves the text quality of the selected image by 0.36 MOS (14%), demonstrating utility for benchmarking, filtering, and generation-time selection. Together, these findings establish perceptual text quality as a distinct evaluation target for modern text-to-image generation.

![Example](example.png)

---

## TODO:
- [x] Release inference code
- [ ] Release training code
- [ ] Release datasets

## Setup

### Clone the repository

```bash
git clone https://github.com/<user>/antiqa_inference.git
cd antiqa_inference
```

### Install dependencies

Using Conda (recommended):

```bash
conda env create -f environment.yml
conda activate antiqa
```

Or using pip:

```bash
pip install -r requirements.txt
```

> **Requirements:** CUDA-compatible GPU, Python 3.12+, PaddlePaddle GPU, PyTorch.

---

## Usage

### 1. Compute ANTIQA scores for images

```bash
python antiqa_infer.py \
    --gpu 0 \
    --antiqa_ckpt antiqa.ckpt \
    --input path/to/image_or_folder \
    --out_path results.tsv
```

| Argument | Description |
|---|---|
| `--gpu` | GPU index (e.g. `0`) |
| `--antiqa_ckpt` | Path to the ANTIQA checkpoint (`antiqa.ckpt`) |
| `--input` | Single image file or directory of images |
| `--out_path` | Output TSV file (omit to print to stdout) |
| `--paddle_det_ckpt` | Optional: path to local PP-OCRv5 detector model directory |

**Output format** (one line per image):

```
<image_path>\tmean=<m>\tnum_crops=<N>\tpoly=<p1>\tscore=<s1>\t...
```

### 2. Visualize results

```bash
python codebase/visualizer.py \
    --results results.tsv \
    --out_dir vis_output
```

| Argument | Description |
|---|---|
| `--results` | TSV output from `antiqa_infer.py` |
| `--out_dir` | Directory for annotated images |
| `--thickness` | Line thickness (default: 1) |
| `--font_scale` | Font scale for labels (default: 0.3) |

### Quick demo

```bash
# Step 1 — obtain scores for demo images
python antiqa_infer.py --gpu 0 --antiqa_ckpt antiqa.ckpt --input demo/imgs --out_path demo/out.tsv

# Step 2 — generate visualizations
python codebase/visualizer.py --results demo/out.tsv --out_dir demo/vis
```
