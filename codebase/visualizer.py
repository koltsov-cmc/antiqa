"""
Visualize ANTIQA inference results.

Takes the tab-separated output file produced by ``antiqa_infer.py`` and, for
every image listed in it, overlays the detected oriented quadrilaterals and
the corresponding ANTIQA scores (truncated to two decimals) onto the original
image, then writes the rendered copy into a target folder.

CLI
---
    python antiqa_visualize.py \
        --results results.tsv \
        --out_dir ./vis_out \
        [--thickness 2] [--font_scale 0.6]

Input line format (produced by antiqa_infer.py):
    <image_path>\tmean=<m>\tnum_crops=<N>\t
    poly=(x1,y1);(x2,y2);(x3,y3);(x4,y4)\tscore=<s>\t...
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

import cv2
import numpy as np


ParsedRecord = Tuple[np.ndarray, float]
ParsedLine = Tuple[Path, float, List[ParsedRecord]]


# --------------------------------------------------------------------------- #
#                                  parsing                                    #
# --------------------------------------------------------------------------- #

_POINT_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")


def _parse_poly(field_value: str) -> np.ndarray:
    """Parse ``(x1,y1);(x2,y2);(x3,y3);(x4,y4)`` into a 4x2 float array."""
    points = _POINT_RE.findall(field_value)
    if len(points) < 3:
        raise ValueError(f"Malformed poly field: {field_value!r}")
    return np.asarray([[float(x), float(y)] for x, y in points], dtype=np.float32)


def parse_line(line: str) -> Optional[ParsedLine]:
    """Parse one TSV line produced by antiqa_infer.py."""
    line = line.rstrip("\n")
    if not line.strip():
        return None

    parts = line.split("\t")
    if len(parts) < 2:
        return None

    image_path = Path(parts[0])
    mean = float("nan")
    records: List[ParsedRecord] = []

    pending_poly: Optional[np.ndarray] = None
    for field in parts[1:]:
        if "=" not in field:
            continue
        key, _, value = field.partition("=")
        key = key.strip()
        value = value.strip()

        if key == "mean":
            try:
                mean = float(value)
            except ValueError:
                mean = float("nan")
        elif key == "num_crops":
            continue
        elif key == "poly":
            try:
                pending_poly = _parse_poly(value)
            except ValueError as e:
                print(f"[warn] {image_path}: {e}", file=sys.stderr)
                pending_poly = None
        elif key == "score":
            if pending_poly is None:
                continue
            try:
                score = float(value)
            except ValueError:
                pending_poly = None
                continue
            records.append((pending_poly, score))
            pending_poly = None

    return image_path, mean, records


# --------------------------------------------------------------------------- #
#                                 rendering                                   #
# --------------------------------------------------------------------------- #

def _score_to_color(score: float) -> Tuple[int, int, int]:
    """Map an ANTIQA score in [0, 5] to a BGR color (red -> yellow -> green)."""
    if np.isnan(score):
        return (180, 180, 180)
    t = float(np.clip(score / 5.0, 0.0, 1.0))
    if t < 0.5:
        # red -> yellow
        r = 255
        g = int(round(255 * (t / 0.5)))
        b = 0
    else:
        # yellow -> green
        r = int(round(255 * (1.0 - (t - 0.5) / 0.5)))
        g = 255
        b = 0
    return (b, g, r)  # OpenCV is BGR


def _draw_label(img: np.ndarray,
                text: str,
                anchor: Tuple[int, int],
                color: Tuple[int, int, int],
                font_scale: float,
                thickness: int) -> None:
    """Draw a filled label box with `text` just above/right of `anchor`."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = max(2, thickness + 1)

    h, w = img.shape[:2]
    x0, y0 = int(anchor[0]), int(anchor[1])

    # Place the label above the anchor; if it goes off-screen, put it below.
    top_y = y0 - th - 2 * pad
    if top_y < 0:
        top_y = y0 + 2
    bottom_y = top_y + th + 2 * pad
    left_x = max(0, min(w - tw - 2 * pad, x0))
    right_x = left_x + tw + 2 * pad

    cv2.rectangle(img, (left_x, top_y), (right_x, bottom_y), color, thickness=-1)
    cv2.putText(
        img, text,
        (left_x + pad, bottom_y - pad - baseline // 2),
        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
    )


def render_image(image_path: Path,
                 mean: float,
                 records: List[ParsedRecord],
                 thickness: int,
                 font_scale: float) -> Optional[np.ndarray]:
    """Return a copy of the image with polygons and scores drawn on top."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[warn] cannot read image {image_path}", file=sys.stderr)
        return None

    vis = img.copy()

    for poly, score in records:
        poly_int = np.round(poly).astype(np.int32)
        color = _score_to_color(score)

        cv2.polylines(
            vis, [poly_int], isClosed=True,
            color=color, thickness=thickness, lineType=cv2.LINE_AA,
        )

        # Anchor the label at the top-left corner of the polygon.
        anchor_idx = int(np.argmin(poly_int[:, 0] + poly_int[:, 1]))
        anchor = tuple(poly_int[anchor_idx].tolist())
        _draw_label(vis, f"{score:.2f}", anchor, color, font_scale, thickness)

    # Small caption in the top-left with the per-image mean score.
    caption = f"mean={mean:.2f}  crops={len(records)}"
    _draw_label(
        vis, caption, (10, 10 + int(30 * font_scale)),
        color=(255, 255, 255), font_scale=font_scale, thickness=thickness,
    )

    return vis


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize ANTIQA inference results produced by antiqa_infer.py.",
    )
    p.add_argument("--results", type=str, required=True,
                   help="Path to the TSV output file of antiqa_infer.py.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory where annotated images will be written.")
    p.add_argument("--thickness", type=int, default=1,
                   help="Line thickness for polygons and label text (default: 2).")
    p.add_argument("--font_scale", type=float, default=0.3,
                   help="Font scale for the score labels (default: 0.6).")
    return p.parse_args()


def _unique_output_path(out_dir: Path, image_path: Path) -> Path:
    """Pick a non-colliding output filename for the rendered copy."""
    stem = image_path.stem
    suffix = image_path.suffix if image_path.suffix else ".png"
    candidate = out_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = out_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.is_file():
        print(f"Results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    total = 0
    rendered = 0

    with results_path.open("r", encoding="utf-8") as f:
        for raw in tqdm(f):
            parsed = parse_line(raw)
            if parsed is None:
                continue
            total += 1

            image_path, mean, records = parsed
            vis = render_image(
                image_path, mean, records,
                thickness=args.thickness,
                font_scale=args.font_scale,
            )
            if vis is None:
                continue

            dst = _unique_output_path(out_dir, image_path)
            if not cv2.imwrite(str(dst), vis):
                print(f"[warn] failed to write {dst}", file=sys.stderr)
                continue
            rendered += 1

    print(f"Rendered {rendered} / {total} images into {out_dir}")


if __name__ == "__main__":
    main()