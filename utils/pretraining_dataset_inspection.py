import os
import cv2
from collections import defaultdict
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE_DIR = r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset"
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

# ----------------------------------------

def get_dataset(name):
    if name.startswith("neu"):
        return "neu"
    elif name.startswith("dagm"):
        return "dagm"
    elif name.startswith("sev"):
        return "sev"
    return "unknown"

def is_empty_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return True
    return mask.sum() == 0

# ---------------- ANALYSIS ----------------

stats = defaultdict(lambda: {
    "total": 0,
    "positive": 0,
    "negative": 0,
    "shapes": set()
})

files = os.listdir(IMG_DIR)

for f in tqdm(files, desc="Analyzing dataset"):

    img_path = os.path.join(IMG_DIR, f)
    mask_path = os.path.join(MASK_DIR, f)

    if not os.path.exists(mask_path):
        continue

    dataset = get_dataset(f)

    stats[dataset]["total"] += 1

    # mask check
    if is_empty_mask(mask_path):
        stats[dataset]["negative"] += 1
    else:
        stats[dataset]["positive"] += 1

    # shape check (only sample few to avoid memory)
    if len(stats[dataset]["shapes"]) < 5:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stats[dataset]["shapes"].add(img.shape)

# ---------------- PRINT ----------------

print("\n📊 DATASET ANALYSIS\n")

for dataset, data in stats.items():

    total = data["total"]
    pos = data["positive"]
    neg = data["negative"]

    pos_pct = (pos / total * 100) if total else 0
    neg_pct = (neg / total * 100) if total else 0

    print(f"🔹 {dataset.upper()}")
    print(f"  Total: {total}")
    print(f"  Positive: {pos} ({pos_pct:.2f}%)")
    print(f"  Negative: {neg} ({neg_pct:.2f}%)")
    print(f"  Sample shapes: {list(data['shapes'])}")
    print("-" * 50)