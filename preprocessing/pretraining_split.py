import os
import cv2
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE_DIR = r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset"
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

OUTPUT_DIR = os.path.join(BASE_DIR, "split")

RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}

random.seed(42)

# ---------------- HELPERS ----------------

def get_dataset(name):
    if name.startswith("neu"):
        return "neu"
    elif name.startswith("dagm"):
        return "dagm"
    elif name.startswith("sev"):
        return "sev"
    return "unknown"

def is_empty(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask is None or mask.sum() == 0

# ---------------- BUCKETING ----------------

buckets = defaultdict(list)

files = os.listdir(IMG_DIR)

for f in tqdm(files, desc="Bucketing"):

    img_path = os.path.join(IMG_DIR, f)
    mask_path = os.path.join(MASK_DIR, f)

    if not os.path.exists(mask_path):
        continue

    dataset = get_dataset(f)
    empty = is_empty(mask_path)

    key = f"{dataset}_{'neg' if empty else 'pos'}"
    buckets[key].append(f)

# ---------------- SPLIT ----------------

splits = {"train": [], "val": [], "test": []}

for key, items in buckets.items():

    random.shuffle(items)

    total = len(items)
    n_train = int(total * RATIOS["train"])
    n_val = int(total * RATIOS["val"])

    splits["train"].extend(items[:n_train])
    splits["val"].extend(items[n_train:n_train+n_val])
    splits["test"].extend(items[n_train+n_val:])

# ---------------- BALANCE TRAIN ----------------

train_files = splits["train"]

pos_files = []
neg_files = []

for f in train_files:
    mask_path = os.path.join(MASK_DIR, f)
    if is_empty(mask_path):
        neg_files.append(f)
    else:
        pos_files.append(f)

# cap negatives
max_neg = int(len(pos_files) * 1.2)

if len(neg_files) > max_neg:
    neg_files = random.sample(neg_files, max_neg)

balanced_train = pos_files + neg_files
random.shuffle(balanced_train)

splits["train"] = balanced_train

# ---------------- SAVE ----------------

for split in splits:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "masks"), exist_ok=True)

    for f in tqdm(splits[split], desc=f"Saving {split}"):

        src_img = os.path.join(IMG_DIR, f)
        src_mask = os.path.join(MASK_DIR, f)

        dst_img = os.path.join(OUTPUT_DIR, split, "images", f)
        dst_mask = os.path.join(OUTPUT_DIR, split, "masks", f)

        shutil.copy(src_img, dst_img)
        shutil.copy(src_mask, dst_mask)

print("\nSplitting complete.")