[![Paper](https://img.shields.io/badge/Paper-red?style=flat-square)]()
[![Demo](https://img.shields.io/badge/Demo-blue?style=flat-square)](demo/)

# TIQA: HUMAN-ALIGNED PERCEPTUAL TEXT QUALITY ASSESSMENT IN GENERATED IMAGES

**ABSTRACT**

RECENT TEXT-TO-IMAGE MODELS HAVE IMPROVED GLOBAL REALISM, BUT TEXT RENDERING REMAINS A PERSISTENT FAILURE MODE: IMAGES MAY LOOK CONVINCING OVERALL, YET LOCAL TYPOGRAPHY OFTEN CONTAINS MALFORMED GLYPHS, BROKEN STROKES, IRREGULAR SPACING, AND OTHER ARTIFACTS THAT HEAVILY PENALIZE HUMANS. WE FORMULATE TEXT-IN-IMAGE QUALITY ASSESSMENT (TIQA), A NO-REFERENCE TASK THAT ESTIMATES A HUMAN-ALIGNED PERCEPTUAL QUALITY SCORE FOR DETECTED TEXT REGIONS WHILE DISENTANGLING VISUAL TEXT QUALITY FROM SEMANTIC CORRECTNESS. TO SUPPORT THIS SETTING, WE INTRODUCE TWO DATASETS. TIQA-CROPS CONTAINS 120K TEXT CROPS FROM 36K AI-GENERATED IMAGES PRODUCED BY 12 GENERATORS, WITH 10K MEAN-OPINION-SCORE (MOS) LABELS AND 110K PROXY LABELS FOR PRETRAINING. TIQA-IMAGES CONTAINS 1,500 TEXT-HEAVY IMAGES FROM 10 RECENT GENERATORS, INCLUDING PROPRIETARY SYSTEMS, WITH PAIRED OVERALL-QUALITY AND TEXT-QUALITY SUBJECTIVE SCORES. WE ALSO PROPOSE ANTIQA, A LIGHTWEIGHT PREDICTOR WITH TEXT-SPECIFIC INDUCTIVE BIASES. ACROSS CROP-LEVEL AND IMAGE-LEVEL EVALUATIONS, ANTIQA ACHIEVES THE BEST ALIGNMENT WITH HUMAN JUDGMENTS, REACHING PLCC/SROCC OF 0.942/0.935 ON TIQA-CROPS AND 0.842/0.837 FOR TEXT-QUALITY MOS ON UNSEEN GENERATORS IN TIQA-IMAGES. IN BEST-OF-5 AI-GENERATED IMAGE RANKING, ANTIQA IMPROVES THE TEXT QUALITY OF THE SELECTED IMAGE BY 0.36 MOS (14%), DEMONSTRATING UTILITY FOR BENCHMARKING, FILTERING, AND GENERATION-TIME SELECTION. TOGETHER, THESE FINDINGS ESTABLISH PERCEPTUAL TEXT QUALITY AS A DISTINCT EVALUATION TARGET FOR MODERN TEXT-TO-IMAGE GENERATION.

![Example](example.png)

---

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
