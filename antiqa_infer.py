"""
Standalone ANTIQA inference over PaddleOCR-detected text regions.

Given an image (or a folder of images), this script:
  1. Reads each image.
  2. Runs the PaddleOCR text detector to get oriented quadrilateral crops.
  3. Rectifies each quadrilateral to a horizontal crop in memory
     (no files are written to disk for the crops).
  4. Feeds every crop through ANTIQA and records its quality score.
  5. Either prints per-image summaries to stdout, or writes them to
     a single output file, depending on whether --out_path is provided.

CLI
---
    python antiqa_infer.py \
        --gpu 0 \
        --antiqa_ckpt ./antiqa.ckpt \
        --input path/to/image_or_folder \
        [--paddle_det_ckpt ./PP-OCRv5_server_det_infer] \
        [--out_path results.tsv]

Output format (per image):
    <image_path>\tmean=<float>\t\
    poly=[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]\tscore=<float>\t...
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

import cv2
import numpy as np
import torch

from paddleocr import TextDetection

# Project-local imports — these must be importable from the script's working dir.
from codebase.antiqa import ANTIQA                                # the trained checkpoint class
from codebase.dataloader_and_aug import Sobel                     # fixed Sobel operator
# Geometry / cropping helpers already defined in the project.
from codebase.utils import (  # noqa: F401
    crop_and_make_horizontal,
    read_img_names_from_folder,
)


# --------------------------------------------------------------------------- #
#                               ANTIQA helpers                                #
# --------------------------------------------------------------------------- #

def _prepare_input_for_model(imgs: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert an RGB batch in [0, 1] to the 2-channel (gray, sobel) input that
    ANTIQA expects."""
    imgs = imgs.to(device=device, dtype=torch.float32)
    weights = torch.tensor(
        [0.2989, 0.5870, 0.1140], device=device, dtype=imgs.dtype
    ).view(1, 3, 1, 1)
    gray = (imgs[:, :3, :, :] * weights).sum(dim=1, keepdim=True)   # [N, 1, H, W]
    sobel = Sobel().to(device=device, dtype=imgs.dtype)
    edge = sobel(gray)                                              # [N, 1, H, W]
    return torch.cat([gray, edge], dim=1)                           # [N, 2, H, W]


@torch.no_grad()
def antiqa_score_single(model: torch.nn.Module,
                        crop_bgr: np.ndarray,
                        device: torch.device) -> float:
    """Score a single BGR crop (uint8 HxWx3) with ANTIQA."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    inp = _prepare_input_for_model(t, device)
    out = model(inp).view(-1)
    return float(out.item())


# --------------------------------------------------------------------------- #
#                          PaddleOCR + ANTIQA pipeline                        #
# --------------------------------------------------------------------------- #

# Upper bound on the number of detected regions we keep per image.
MAX_NUM_CROPS: int = 50
# Upscale factor applied during perspective rectification.
FIRST_SCALE: float = 1.9


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def iter_input_images(input_path: Path) -> List[Path]:
    """Return a list of image paths from a file or a directory."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return read_img_names_from_folder(str(input_path))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def score_image(img: np.ndarray,
                det_model: TextDetection,
                antiqa_model: torch.nn.Module,
                device: torch.device) -> List[Tuple[np.ndarray, float]]:
    """
    Run detection + rectification + ANTIQA scoring on a single image.

    Every detected polygon that rectifies to a non-empty crop is scored;

    Returns a list of (polygon, antiqa_score) tuples, one per kept crop.
    Polygons are 4x2 float arrays in the original image coordinate system.
    """
    det_out = det_model.predict(img, batch_size=1)[0]
    polys = det_out.get("dt_polys")
    if polys is None:
        polys = []

    results: List[Tuple[np.ndarray, float]] = []

    for poly in np.asarray(polys[:MAX_NUM_CROPS], dtype=np.float32):
        # Rectify the quadrilateral to a horizontal crop (in-memory only).
        try:
            crop = crop_and_make_horizontal(img, poly, upscale=FIRST_SCALE)
        except Exception:
            continue

        if crop is None or crop.size == 0:
            continue

        score = antiqa_score_single(antiqa_model, crop, device)
        results.append((np.asarray(poly, dtype=np.float32), score))

    return results


# --------------------------------------------------------------------------- #
#                              output formatting                              #
# --------------------------------------------------------------------------- #

def _poly_to_str(poly: np.ndarray) -> str:
    """Format a 4x2 polygon as '(x1,y1);(x2,y2);(x3,y3);(x4,y4)'."""
    return ";".join(f"({float(p[0]):.2f},{float(p[1]):.2f})" for p in poly)


def format_image_record(image_path: Path,
                        records: List[Tuple[np.ndarray, float]]) -> str:
    """
    One line per image. Fields are tab-separated:
        image_path \t mean=<m> \t poly=<p1> \t score=<s1> \t poly=<p2> \t ...
    """
    if records:
        mean = float(np.mean([s for _, s in records]))
    else:
        mean = float("nan")

    parts = [str(image_path), f"mean={mean:.6f}", f"num_crops={len(records)}"]
    for poly, score in records:
        parts.append(f"poly={_poly_to_str(poly)}")
        parts.append(f"score={score:.6f}")
    return "\t".join(parts)


def print_image_record_console(image_path: Path,
                               records: List[Tuple[np.ndarray, float]]) -> None:
    """Pretty console output: mean first, then per-crop polygon and score."""
    if records:
        mean = float(np.mean([s for _, s in records]))
    else:
        mean = float("nan")

    print(f"=== {image_path} ===")
    print(f"  mean ANTIQA score over {len(records)} crop(s): {mean:.4f}")
    for i, (poly, score) in enumerate(records, start=1):
        poly_str = _poly_to_str(poly)
        print(f"  crop {i:02d}: score={score:.4f}  poly={poly_str}")
    print()


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ANTIQA over PaddleOCR-detected crops without saving them."
    )
    p.add_argument("--gpu", type=str, required=True,
                   help="GPU index passed to both PaddleOCR (gpu:N) and PyTorch (cuda:N).")
    p.add_argument("--antiqa_ckpt", type=str, required=True,
                   help="Path to the ANTIQA Lightning checkpoint.")
    p.add_argument("--paddle_det_ckpt", type=str, default=None,
                   help="Optional path to a local PP-OCRv5 detector model directory. "
                        "If omitted, the PaddleOCR default auto-download is used.")
    p.add_argument("--input", type=str, required=True,
                   help="Path to a single image file or a directory of images.")
    p.add_argument("--out_path", type=str, default=None,
                   help="Optional output file. If omitted, results are printed to stdout.")
    return p.parse_args()


def build_paddle_detector(gpu: str, det_ckpt: Optional[str]) -> TextDetection:
    device = f"gpu:{gpu}"
    det_kwargs = {"model_name": "PP-OCRv5_server_det", "device": device}
    if det_ckpt:
        det_kwargs["model_dir"] = det_ckpt
    return TextDetection(**det_kwargs)


def build_antiqa(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load an ANTIQA Lightning checkpoint and prepare it for inference."""
    model = ANTIQA.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    det_model = build_paddle_detector(args.gpu, args.paddle_det_ckpt)
    antiqa_model = build_antiqa(args.antiqa_ckpt, device)

    image_paths = iter_input_images(Path(args.input))
    if not image_paths:
        print(f"No images found under {args.input}", file=sys.stderr)
        sys.exit(1)

    out_file = None
    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w", encoding="utf-8")

    try:
        for img_path in tqdm(image_paths):
            try:
                img = read_image(img_path)
            except Exception as e:
                print(f"[warn] cannot read {img_path}: {e}", file=sys.stderr)
                continue

            try:
                records = score_image(img, det_model, antiqa_model, device)
            except Exception as e:
                print(f"[warn] failed on {img_path}: {e}", file=sys.stderr)
                continue

            if out_file is not None:
                out_file.write(format_image_record(img_path, records) + "\n")
                out_file.flush()
            else:
                print_image_record_console(img_path, records)
    finally:
        if out_file is not None:
            out_file.close()


if __name__ == "__main__":
    main()