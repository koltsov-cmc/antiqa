"""
Visualize ANTIQA inference results.

Takes the tab-separated output file produced by ``antiqa_infer.py`` and, for
every image listed in it, overlays the detected oriented quadrilaterals and
the corresponding ANTIQA scores (truncated to two decimals) onto the original
image, then writes the rendered copy into a target folder.

CLI
---
    python codebase/visualizer.py \
        --results results.tsv \
        --out_dir ./vis_out \
        [--thickness 2] [--poly_thickness 2] [--font_scale 0.7]

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

def _adjust_bgr(bgr: Tuple[int, int, int],
                darken: float = 0.0,
                saturation: float = 1.0) -> Tuple[int, int, int]:
    """Boost saturation and darken value of a BGR color, working in HSV.

    Scaling all RGB channels toward black keeps the hue but reads as a dull,
    muddy tone. Instead this pushes saturation up (`saturation` > 1 = more
    vivid) and drops only the value (`darken` in [0, 1], 1 = black), so a
    darker color stays a deep, clean version of the same hue.
    """
    px = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[0, 0]
    s = float(np.clip(s * max(0.0, saturation), 0.0, 255.0))
    v = float(np.clip(v * (1.0 - float(np.clip(darken, 0.0, 1.0))), 0.0, 255.0))
    hsv[0, 0] = (h, s, v)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    b, g, r = out[0, 0]
    return (int(b), int(g), int(r))


def _score_to_color(score: float,
                    darken: float = 0.0,
                    saturation: float = 1.0) -> Tuple[int, int, int]:
    """Map an ANTIQA score in [0, 5] to a BGR color (red -> yellow -> green).

    `darken` and `saturation` are forwarded to `_adjust_bgr` so the polygons
    and label boxes render darker but stay vivid rather than muddy.
    """
    if np.isnan(score):
        b, g, r = 180, 180, 180
    else:
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
    return _adjust_bgr((b, g, r), darken, saturation)  # OpenCV is BGR


Rect = Tuple[int, int, int, int]  # (left, top, right, bottom)


def _label_geometry(text: str,
                    font_scale: float,
                    thickness: int) -> Tuple[int, int, int, int]:
    """Return (box_w, box_h, text_height, pad) for a label holding `text`."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    # Padding scales with the font so the text keeps an even margin at any size.
    pad = max(3, int(round(4 * font_scale)))
    box_w = tw + 2 * pad
    # Box tall enough for the ascenders (th) *and* the descenders (baseline).
    box_h = th + baseline + 2 * pad
    return box_w, box_h, th, pad


def _rects_overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _closest_on_segment(p, a, b) -> Tuple[float, float]:
    """Point on segment a-b closest to p."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return ax, ay
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    return ax + t * dx, ay + t * dy


def _nearest_on_poly(pt, poly) -> Tuple[int, int]:
    """Point on the polygon outline closest to `pt` (lands on the drawn frame)."""
    best = None
    best_d = None
    n = len(poly)
    for i in range(n):
        q = _closest_on_segment(pt, poly[i], poly[(i + 1) % n])
        d = (q[0] - pt[0]) ** 2 + (q[1] - pt[1]) ** 2
        if best_d is None or d < best_d:
            best_d, best = d, q
    return int(round(best[0])), int(round(best[1]))


def _nearest_on_rect(pt, rect: Rect) -> Tuple[int, int]:
    """Point on `rect`'s border closest to `pt`."""
    x = min(max(int(pt[0]), rect[0]), rect[2])
    y = min(max(int(pt[1]), rect[1]), rect[3])
    return x, y


def _place_label(target: Rect,
                 box_w: int,
                 box_h: int,
                 labels: List[Rect],
                 boxes: List[Rect],
                 img_h: int,
                 img_w: int) -> Rect:
    """Find a rectangle for the label as close to `target` as possible.

    The label is packed against the text box like a brick: among all positions
    that overlap nothing, the one nearest the box wins (touching preferred, then
    smallest centre offset), so labels sit snug on whichever side has room and
    the leader line meets the frame at whatever point is closest. The search
    window grows only if the immediate neighbourhood is full. Two passes: first
    avoid both the other `labels` and every text `box`; if nothing fits, relax
    to labels only (may sit on a frame, but never on another label).
    """
    tx0, ty0, tx1, ty1 = target
    tcx, tcy = (tx0 + tx1) // 2, (ty0 + ty1) // 2
    step = max(3, min(box_w, box_h) // 4)   # fine grid -> tight packing

    def cost(rect: Rect) -> int:
        # Gap distance to the box dominates (touching == 0); centre offset breaks
        # ties so the label hugs the box instead of drifting along a side.
        dx = max(tx0 - rect[2], rect[0] - tx1, 0)
        dy = max(ty0 - rect[3], rect[1] - ty1, 0)
        ccx = (rect[0] + rect[2]) // 2 - tcx
        ccy = (rect[1] + rect[3]) // 2 - tcy
        return (dx * dx + dy * dy) * 100000 + (ccx * ccx + ccy * ccy)

    def best_in_window(radius: int, avoid: List[Rect]) -> Optional[Rect]:
        x_lo = max(0, tx0 - box_w - radius)
        x_hi = min(img_w - box_w, tx1 + radius)
        y_lo = max(0, ty0 - box_h - radius)
        y_hi = min(img_h - box_h, ty1 + radius)
        best: Optional[Rect] = None
        best_c: Optional[int] = None
        y = y_lo
        while y <= y_hi:
            x = x_lo
            while x <= x_hi:
                rect = (x, y, x + box_w, y + box_h)
                if not any(_rects_overlap(rect, o) for o in avoid):
                    c = cost(rect)
                    if best_c is None or c < best_c:
                        best_c, best = c, rect
                x += step
            y += step
        return best

    r0 = max(box_w, box_h)
    for avoid in (labels + boxes, labels):
        radius = r0
        while radius <= 2 * max(img_w, img_h):
            rect = best_in_window(radius, avoid)
            if rect is not None:
                return rect
            radius *= 2

    left_x = max(0, min(img_w - box_w, tx1 + step))
    top_y = max(0, min(img_h - box_h, ty0))
    return (left_x, top_y, left_x + box_w, top_y + box_h)


def _draw_label(img: np.ndarray,
                text: str,
                rect: Rect,
                th: int,
                pad: int,
                color: Tuple[int, int, int],
                font_scale: float,
                thickness: int) -> None:
    """Draw the filled label box `rect` with `text` inside it."""
    left_x, top_y, right_x, bottom_y = rect
    cv2.rectangle(img, (left_x, top_y), (right_x, bottom_y), color, thickness=-1)
    # Baseline sits `pad` below the top plus the ascender band; the descenders
    # then occupy `baseline`, leaving `pad` down to the bottom edge.
    text_org = (left_x + pad, top_y + pad + th)
    cv2.putText(
        img, text, text_org,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
    )


def render_image(image_path: Path,
                 mean: float,
                 records: List[ParsedRecord],
                 thickness: int,
                 font_scale: float,
                 poly_thickness: int,
                 darken: float = 0.0,
                 saturation: float = 1.0) -> Optional[np.ndarray]:
    """Return a copy of the image with polygons and scores drawn on top."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[warn] cannot read image {image_path}", file=sys.stderr)
        return None

    vis = img.copy()
    h, w = vis.shape[:2]

    # Draw every text polygon first, and remember its points + bounding box so
    # each label can be wired back to its own frame with a leader line.
    items = []  # (bbox, poly_int, score, color)
    for poly, score in records:
        poly_int = np.round(poly).astype(np.int32)
        color = _score_to_color(score, darken, saturation)

        cv2.polylines(
            vis, [poly_int], isClosed=True,
            color=color, thickness=poly_thickness, lineType=cv2.LINE_AA,
        )

        xs, ys = poly_int[:, 0], poly_int[:, 1]
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        items.append((bbox, poly_int, score, color))

    # Place labels top-to-bottom, then left-to-right, so nearby ones settle into
    # the closest free slot deterministically.
    items.sort(key=lambda it: (it[0][1], it[0][0]))

    # Every text frame, inflated by a small clearance so a leader line always
    # has room, is something labels try to avoid.
    clr = 6
    boxes = [(b[0] - clr, b[1] - clr, b[2] + clr, b[3] + clr) for b, _p, _s, _c in items]

    placed: List[Rect] = []
    to_draw = []  # (poly_int, rect, text, th, pad, color)
    for bbox, poly_int, score, color in items:
        text = f"{score:.2f}"
        box_w, box_h, th, pad = _label_geometry(text, font_scale, thickness)
        # Prefer a slot clear of both other labels and all frames; if none
        # exists, fall back to overlapping a frame (never another label).
        rect = _place_label(bbox, box_w, box_h, placed, boxes, h, w)
        placed.append(rect)
        to_draw.append((poly_int, rect, text, th, pad, color))

    # Leader lines first, so the label boxes are painted over their ends. The
    # line runs from the label edge to the nearest point *on the polygon* (not
    # its axis-aligned bbox), so it always meets the drawn frame, and it uses
    # the same thickness as the frame itself.
    lead_thickness = max(1, poly_thickness)
    for poly_int, rect, _text, _th, _pad, color in to_draw:
        lc = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
        p_box = _nearest_on_poly(lc, poly_int)
        p_label = _nearest_on_rect(p_box, rect)
        cv2.line(vis, p_box, p_label, color, lead_thickness, cv2.LINE_AA)

    # Then the label boxes with the score text on top.
    for _poly, rect, text, th, pad, color in to_draw:
        _draw_label(vis, text, rect, th, pad, color, font_scale, thickness)

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
                   help="Stroke thickness of the score numbers only (default: 1).")
    p.add_argument("--poly_thickness", type=int, default=2,
                   help="Line thickness of the detection polygons/frames (default: 2).")
    p.add_argument("--font_scale", type=float, default=0.7,
                   help="Font scale for the score labels (default: 0.7).")
    p.add_argument("--darken", type=float, default=0.0,
                   help="Darken the polygon/label colors toward black in [0, 1] "
                        "(0 = original, 1 = black; default: 0.0).")
    p.add_argument("--saturation", type=float, default=1.0,
                   help="Saturation multiplier for the polygon/label colors "
                        "(>1 = more vivid, keeps dark colors clean; default: 1.0).")
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
                poly_thickness=args.poly_thickness,
                darken=args.darken,
                saturation=args.saturation,
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