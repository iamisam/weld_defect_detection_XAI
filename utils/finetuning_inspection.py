import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE_DIR = r"D:\weld_defect_project\datasets\segmentation_finetuning_dataset"
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

# ----------------------------------------

def get_dataset(name):
    if name.startswith("gdx"):
        return "gdxray"
    elif name.startswith("gokul"):
        return "gokul"
    return "unknown"

def analyze_mask(mask):
    if mask is None:
        return True, 0.0

    total_pixels = mask.size
    non_zero = np.count_nonzero(mask)

    is_empty = (non_zero == 0)
    coverage = non_zero / total_pixels

    return is_empty, coverage

# ---------------- ANALYSIS ----------------

stats = defaultdict(lambda: {
    "total": 0,
    "positive": 0,
    "negative": 0,
    "coverage_values": [],
    "shapes": set()
})

files = os.listdir(IMG_DIR)

for f in tqdm(files, desc="Analyzing finetuning dataset"):

    img_path = os.path.join(IMG_DIR, f)
    mask_path = os.path.join(MASK_DIR, f)

    if not os.path.exists(mask_path):
        continue

    dataset = get_dataset(f)

    stats[dataset]["total"] += 1

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    empty, coverage = analyze_mask(mask)

    if empty:
        stats[dataset]["negative"] += 1
    else:
        stats[dataset]["positive"] += 1
        stats[dataset]["coverage_values"].append(coverage)

    # sample shapes
    if len(stats[dataset]["shapes"]) < 5:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stats[dataset]["shapes"].add(img.shape)

# ---------------- PRINT ----------------

print("\n📊 FINETUNING DATASET ANALYSIS\n")

for dataset, data in stats.items():

    total = data["total"]
    pos = data["positive"]
    neg = data["negative"]

    pos_pct = (pos / total * 100) if total else 0
    neg_pct = (neg / total * 100) if total else 0

    if data["coverage_values"]:
        avg_cov = np.mean(data["coverage_values"])
        min_cov = np.min(data["coverage_values"])
        max_cov = np.max(data["coverage_values"])
    else:
        avg_cov = min_cov = max_cov = 0

    print(f"🔹 {dataset.upper()}")
    print(f"  Total: {total}")
    print(f"  Positive: {pos} ({pos_pct:.2f}%)")
    print(f"  Negative: {neg} ({neg_pct:.2f}%)")
    print(f"  Coverage (defect area):")
    print(f"    avg: {avg_cov:.6f}")
    print(f"    min: {min_cov:.6f}")
    print(f"    max: {max_cov:.6f}")
    print(f"  Sample shapes: {list(data['shapes'])}")
    print("-" * 60)