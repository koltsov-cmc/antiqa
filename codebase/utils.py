import numpy as np
from paddleocr import TextDetection
from paddleocr import TextRecognition
from paddleocr import PaddleOCR  
from pathlib import Path
from typing import Dict, List, Tuple, Union
import csv
import numpy as np
import cv2
from itertools import combinations
import string
from math import isnan

def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4,2)

    x_sorted = pts[np.argsort(pts[:, 0])]
    left = x_sorted[:2]
    right = x_sorted[2:]

    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def find_right_angle_triplet(pts, cos_tol=0.05): 
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] != 4 or pts.shape[1] != 2:
        raise ValueError("pts must be 4x2")
    for j in range(4):
        others = [x for x in range(4) if x != j]
        for i, k in combinations(others, 2):
            v1 = pts[i] - pts[j]
            v2 = pts[k] - pts[j]
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cos = np.dot(v1, v2) / (n1 * n2)
            if abs(cos) <= cos_tol:
                return (i, j, k)

def is_rectangle_strict(rect, angle_tol_deg=10.0, rel_tol=1e-2, abs_tol=1e-6):
    rect = np.asarray(rect, dtype=float).reshape(4, 2)
    vs = [rect[(i+1) % 4] - rect[i] for i in range(4)]
    norms = [np.linalg.norm(v) for v in vs]

    for i in range(4):
        a, b = vs[i], vs[(i+1) % 4]
        na, nb = norms[i], norms[(i+1) % 4]
        if na < 1e-9 or nb < 1e-9:  
            return False
        cos = np.dot(a, b) / (na * nb)
        cos = float(np.clip(cos, -1.0, 1.0))
        angle_dev = abs(90.0 - np.degrees(np.arccos(cos)))

        if angle_dev > angle_tol_deg:
            return False

    return True

def find_coordinate(p1, p2, known_coord, find_x=True):
    y_1, x_1 = p1
    y_2, x_2 = p2

    if find_x:
        y_0 = known_coord
        return -((y_0 - y_1)*(y_2 - y_1)) / (x_2 - x_1) + x_1

    x_0 = known_coord
    return -((x_0 - x_1)*(x_2 - x_1)) / (y_2 - y_1) + y_1

def get_trapezoid_area(p1, p2, p3, p4):
    return abs(0.5 * np.linalg.norm(p1 - p2) * np.linalg.norm(p3 - p4) * np.linalg.norm(p4 - p2))

def fourth_corner_from_three(pts):
    pts = np.asarray(pts, float).reshape(3,2)
    i = min(range(3), key=lambda k: abs(np.dot(pts[(k+1)%3]-pts[k], pts[(k+2)%3]-pts[k])))
    return pts[(i+1)%3] + pts[(i+2)%3] - pts[i]

def check_if_vertical(tr, br, angle_tol_deg=5.0):
    vec = tr - br
    dx, dy = vec[0], vec[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return False

    angle_deg = abs(np.degrees(np.arctan2(dy, dx)))  
    angle_dev = abs(90.0 - angle_deg) 
    return angle_dev <= angle_tol_deg

def crop_rect_and_make_horizontal(img, rect, upscale=1.0, interp=cv2.INTER_CUBIC):
    pts = order_points(rect) 

    widthA = np.linalg.norm(pts[2] - pts[3])
    widthB = np.linalg.norm(pts[1] - pts[0])
    maxW = max(3, int(min(widthA, widthB)))

    heightA = np.linalg.norm(pts[1] - pts[2])
    heightB = np.linalg.norm(pts[0] - pts[3])
    maxH = max(3, int(min(heightA, heightB)))

    dst_w = max(3, int(maxW * upscale))
    dst_h = max(3, int(maxH * upscale))

    dst = np.array([[0,0],[dst_w-1,0],[dst_w-1,dst_h-1],[0,dst_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (dst_w, dst_h), flags=interp, borderMode=cv2.BORDER_REPLICATE)

    return warped

def three_indices_with_center(pts, chosen, angle_tol_deg=30.0):
    """
    pts: array-like (4,2)
    chosen: index (0..3) or point-like (x,y) present in pts
    Returns: (edge_idx1, center_idx, edge_idx2) -- indices into original pts
    """
    pts = np.asarray(pts, dtype=float).reshape(4,2)

    if isinstance(chosen, int):
        ci = int(chosen)
    else:
        arr = np.asarray(chosen, dtype=float).reshape(2)
        ci = int(np.where(np.all(np.isclose(pts, arr), axis=1))[0][0])

    rem = [i for i in range(4) if i != ci]
    P = pts[rem]  

    # find which of the 3 is the center: the one whose vectors to the other two are closest to 90 deg
    def angle_dev_from_90(a,b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 1e9
        cos = np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        return abs(ang - 90.0)

    devs = []
    for j in range(3):
        others = [k for k in range(3) if k != j]
        a = P[others[0]] - P[j]
        b = P[others[1]] - P[j]
        devs.append(angle_dev_from_90(a,b))

    center_local = int(np.argmin(devs))  # 0..2
    # if you want to enforce tolerance, you can check devs[center_local] <= angle_tol_deg

    # prepare output: (edge1_idx, center_idx, edge2_idx) as indices in original pts
    edge1_idx = rem[(center_local+1) % 3]
    center_idx = rem[center_local]
    edge2_idx = rem[(center_local+2) % 3]
    return (edge1_idx, center_idx, edge2_idx)

def find_border_point_idx(pts, w, h):
    for i, p in enumerate(pts):
        if np.isclose(p[1], 0.): # верхняя
            return i, "up" 
        elif np.isclose(p[0], w): # правая 
            return i, "right"
        elif np.isclose(p[1], h): # нижняя 
            return i, "down"
        elif np.isclose(p[0], 0.): # левая
            return i, "left"
    return 0, None

def crop_and_make_horizontal(img, poly, upscale=1.0, interp=cv2.INTER_CUBIC):
    pts = order_points(poly) 
    h, w = img.shape[:-1]

    if is_rectangle_strict(pts):
        return crop_rect_and_make_horizontal(img, pts, upscale=upscale, interp=interp)

    border_point_idx, border = find_border_point_idx(pts, w, h)
    if border is None:
        return crop_rect_and_make_horizontal(img, pts, upscale=upscale, interp=interp)
    
    border_point = pts[border_point_idx]

    left_idx, center_idx, right_idx = three_indices_with_center(pts, border_point)
    left, center, right = pts[left_idx], pts[center_idx], pts[right_idx]

    if border == "up": 
        first_cand = [find_coordinate(left, center, border_point[1], find_x=False), border_point[1]]
        second_cand = [find_coordinate(center, right, border_point[1], find_x=False), border_point[1]]

    elif border == "right": 
        first_cand = [border_point[0], find_coordinate(left, center, border_point[0])]
        second_cand = [border_point[0], find_coordinate(center, right, border_point[0])]

    elif border == "down":
        first_cand = [find_coordinate(left, center, border_point[1], find_x=False), border_point[1]]
        second_cand = [find_coordinate(center, right, border_point[1], find_x=False), border_point[1]]

    else:
        first_cand = [border_point[0], find_coordinate(left, center, border_point[0])]
        second_cand = [border_point[0], find_coordinate(center, right, border_point[0])]

    corners = [fourth_corner_from_three([first_cand, left, center]), 
               fourth_corner_from_three([second_cand, right, center])]
    
    first_lost_area = get_trapezoid_area(first_cand, corners[0],second_cand, right)
    second_lost_area = get_trapezoid_area(second_cand, corners[1], first_cand, left)

    best_idx = np.argmin([first_lost_area, second_lost_area])
    best_cand = [first_cand, second_cand][best_idx]
    best_cand_intersection = corners[best_idx]
    remain_point = [left, right][best_idx]
    
    pts = [center, remain_point, best_cand, best_cand_intersection]

    return crop_rect_and_make_horizontal(img, pts, upscale=upscale, interp=interp)


def read_img_names_from_folder(folder_path):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Wrong directory: {folder_path}")

    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files) 

def read_images_from_folder(names, recursive=False):
    images = []
    for p in names:
        img = cv2.imread(str(p))  # BGR, uint8
        if img is None:
            print(f"Warning: can't read image {p!s}, skipping.")
            continue
        images.append((str(p), img))

    return images

def largest_crop_index(crops):
    arr = np.asarray(crops, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (4, 2):
        raise ValueError(f"Expected (N, 4, 2). Got: {arr.shape}")
    x = arr[:, :, 0]   # shape (N, 4)
    y = arr[:, :, 1]   # shape (N, 4)

    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)

    signed = np.sum(x * y_next - x_next * y, axis=1)
    areas = 0.5 * np.abs(signed)

    idx = int(np.argmax(areas))   
    return idx, float(areas[idx])

def read_img(img_path):
    img = cv2.imread(img_path)
    return img

def process_outputs(imgs, det_output, det_model, ocr_model, double_process=False):
    MAX_NUM_CROPS = 50
    FIRST_SCALE = 1.9
    SECOND_SCALE = 1.1
    REC_SCORE_TRESHOLD = 0.2
    HEIGHT_TRESHOLD = 20
    ALLOWED = set(string.ascii_letters + string.digits + string.punctuation + " \t\n\rı·!@#$%^&*()_+\"?:><|")
    res = {}
    
    # for each image
    for img, det_out in zip(imgs, det_output):
        input_path = Path(det_out["input_path"]).stem
        dt_polys = det_out["dt_polys"]
        dt_scores = [float(el) for el in det_out["dt_scores"]]

        # for each crop in image
        horizontal_crops_and_texts = []
        for idx, poly in enumerate(np.asarray(dt_polys[:MAX_NUM_CROPS], dtype=np.float32)):
            crop = crop_and_make_horizontal(img, poly, upscale=FIRST_SCALE)

            if crop.shape[0] <= HEIGHT_TRESHOLD:
                continue

            # crop processing
            if double_process:
                new_det_out = det_model.predict(crop, batch_size=1)[0] # second det call
                new_dt_polys = new_det_out["dt_polys"]

                if type(new_det_out["dt_scores"]) is list and len(new_det_out["dt_scores"]) != 0:
                    largest_crop_idx = largest_crop_index(new_dt_polys)
                    crop = crop_and_make_horizontal(crop, new_dt_polys[largest_crop_idx], upscale=SECOND_SCALE)

            ocr_out = ocr_model.predict(crop, batch_size=1)
            rec_text = ocr_out[0]["rec_text"]
            rec_score = float(ocr_out[0]["rec_score"])

            if isnan(rec_score): # nothing recognized
                continue
            elif rec_score < REC_SCORE_TRESHOLD:
                continue
            elif not any(char in ALLOWED for char in rec_text): # recognized chineze
                continue

            horizontal_crops_and_texts.append((crop, rec_text, rec_score))

        res[input_path] = horizontal_crops_and_texts
    return res
        

def save_crops_and_texts(
    crops_dict: Dict[str, List[Tuple[np.ndarray, str]]],
    root_folder: Union[str, Path] = "crops",
    img_ext: str = ".png",
    crop_index_pad: int = 5,
) -> Dict[str, dict]:
    root = Path(root_folder)
    root.mkdir(parents=True, exist_ok=True)

    summary = {}

    for image_name, items in crops_dict.items():
        image_name_clean = Path(image_name).stem
        subdir = root / image_name_clean
        subdir.mkdir(parents=True, exist_ok=True)

        n = len(items)
        pad = max(crop_index_pad, len(str(max(1, n))))

        saved_paths = []
        ocr_info_path = subdir / "ocr_info.tsv"

        # Open TSV and write header then rows
        with ocr_info_path.open("w", encoding="utf-8", newline="") as tsvf:
            writer = csv.writer(tsvf, delimiter="\t", lineterminator="\n")
            # header as specified: crop_name, ocr_text, ocr_conf, antiqa_score
            writer.writerow(["crop_name", "ocr_text", "ocr_conf", "antiqa_score"])

            for idx, item in enumerate(items, start=1):
                # allow tuples of length 2..4: (crop,text), (crop,text,rec), (crop,text,rec,antiqa)
                if not isinstance(item, (list, tuple)):
                    raise TypeError(f"Item for {image_name_clean} at idx={idx} must be tuple/list")
                if len(item) < 2:
                    raise ValueError(f"Item for {image_name_clean} at idx={idx} must contain at least (crop, text)")

                crop_arr = item[0]
                text = item[1]
                rec_score = item[2] if len(item) >= 3 else ""
                antiqa_score = item[3] if len(item) >= 4 else ""

                if not isinstance(crop_arr, np.ndarray):
                    raise TypeError(f"Crop must be numpy.ndarray, got {type(crop_arr)} for {image_name_clean} idx={idx}")

                fname = f"{image_name_clean}_crop_{idx:0{pad}d}{img_ext}"
                out_path = subdir / fname

                # attempt save
                ok = False
                try:
                    ok = cv2.imwrite(str(out_path), crop_arr)
                except Exception:
                    ok = False

                if not ok:
                    # try float->[0,255] conversion
                    try:
                        arr = crop_arr
                        if np.issubdtype(arr.dtype, np.floating):
                            arr_u8 = (np.clip(arr, 0, 1) * 255).astype("uint8")
                        else:
                            arr_u8 = arr.astype("uint8")
                        ok = cv2.imwrite(str(out_path), arr_u8)
                    except Exception as e:
                        ok = False

                if not ok:
                    # skip this crop but continue processing others
                    # you may want to log/warn here
                    continue

                saved_paths.append(out_path)

                # sanitize text fields (no tabs/newlines)
                text_s = "" if text is None else str(text).replace("\t", " ").replace("\n", " ").strip()
                rec_s = "" if rec_score is None else str(rec_score)
                antiqa_s = "" if antiqa_score is None else str(antiqa_score)

                writer.writerow([fname, text_s, rec_s, antiqa_s])

        summary[image_name_clean] = {
            "dir": subdir,
            "saved_images": saved_paths,
            "ocr_info": ocr_info_path,
        }

    return summary

def get_crops_dict_from_image(img_path, det_model, ocr_model, double_process=False):
    img = read_img(img_path)
    det_output = det_model.predict(img, batch_size=1)
    det_output[0]["input_path"] = img_path

    return process_outputs([img], det_output, det_model, ocr_model, double_process=double_process)

def get_crops_dict_from_batch(batch, det_model, ocr_model, double_process=False):
    pairs = read_images_from_folder(batch)

    imgs = [pair[1] for pair in pairs]
    det_output = det_model.predict(imgs, batch_size=len(batch))

    for idx, pair in enumerate(pairs):
        det_output[idx]["input_path"] = pair[0]

    crops_dict = process_outputs(imgs, det_output, det_model, ocr_model, double_process=double_process)
    return crops_dict

def save_crops_from_folder(in_folder, out_folder, det_model, ocr_model, batch_size=2, 
                                double_process=False, max_images=None):
    names = read_img_names_from_folder(in_folder)

    if max_images is not None:
        names = names[:max_images]

    batches = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]

    for i, batch in enumerate(batches):
        
        crops_dict = get_crops_dict_from_batch(batch, det_model, ocr_model, double_process=double_process)

        print(f"processed batch {i+1} out of {len(batches)}")

        save_crops_and_texts(crops_dict, out_folder)


import numpy as np
import torch


def _to_1d_numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().float().reshape(-1).numpy()
    else:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x


def _pearson_corr(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return float("nan")

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return float("nan")

    return float((x * y).sum() / denom)


def _rankdata_avg(a):
    """
    Average ranks for ties, 1-based ranks.
    Equivalent to scipy.stats.rankdata(method="average"), but without scipy.
    """
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))

    sorted_a = a[sorter]
    ranks = np.empty(len(a), dtype=np.float64)

    i = 0
    while i < len(sorted_a):
        j = i + 1
        while j < len(sorted_a) and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0  # 1-based average rank
        ranks[i:j] = avg_rank
        i = j

    return ranks[inv]


def _spearman_corr(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return float("nan")

    rx = _rankdata_avg(x)
    ry = _rankdata_avg(y)
    return _pearson_corr(rx, ry)


def correlations(targets, preds):
    """
    targets, preds: torch.Tensor / np.ndarray / list
    Returns:
        {
            "pearson": float,
            "spearman": float,
        }
    """
    t = _to_1d_numpy(targets)
    p = _to_1d_numpy(preds)

    n = min(len(t), len(p))
    t = t[:n]
    p = p[:n]

    return {
        "pearson": _pearson_corr(t, p),
        "spearman": _spearman_corr(t, p),
    }