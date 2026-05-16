import os, cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

BASE = "D:/weld_defect_project/datasets"
cls_root = f"{BASE}/unified_classification_standardized_dataset/"

rows = []

for split in ["train","val","test"]:
    root = os.path.join(cls_root, split)

    for cls in os.listdir(root):
        p = os.path.join(root, cls)
        if not os.path.isdir(p): continue

        count = len(os.listdir(p))
        rows.append([split, cls, count])

df = pd.DataFrame(rows, columns=["split","class","count"])
df.to_csv("classification_distribution_all_splits.csv", index=False)

print("Done.")


# ---------- SEGMENTATION EMPTY/NON-EMPTY ----------
def calc_empty_ratio(root):
    empty = 0
    total = 0

    mask_dir = os.path.join(root, "masks")

    for name in tqdm(os.listdir(mask_dir)):
        m = cv2.imread(os.path.join(mask_dir, name), 0)
        if m is None: continue

        total += 1
        if np.sum(m) == 0:
            empty += 1

    return empty, total-empty, total


datasets = {
    "neu": "segmentation_pretraining_dataset/train/neu",
    "dagm": "segmentation_pretraining_dataset/train/dagm",
    "sev": "segmentation_pretraining_dataset/train/sev",
    "custom": "segmentation_finetuning_augmented_dataset/train/gokul",
    "gdx": "segmentation_finetuning_augmented_dataset/train/gdx",
    "neut": "segmentation_pretraining_dataset/test/neu",
    "dagmt": "segmentation_pretraining_dataset/test/dagm",
    "sevt": "segmentation_pretraining_dataset/test/sev",
    "customt": "segmentation_finetuning_augmented_dataset/test/gokul",
    "gdxt": "segmentation_finetuning_augmented_dataset/test/gdx",
    "neuv": "segmentation_pretraining_dataset/val/neu",
    "dagmv": "segmentation_pretraining_dataset/val/dagm",
    "sevv": "segmentation_pretraining_dataset/val/sev",
    "customv": "segmentation_finetuning_augmented_dataset/val/gokul",
    "gdxv": "segmentation_finetuning_augmented_dataset/val/gdx"
}

rows = []

for k,v in datasets.items():
    e, ne, t = calc_empty_ratio(os.path.join(BASE, v))

    # infer split from name
    if k.endswith("t"):
        split = "test"
        ds = k[:-1]
    elif k.endswith("v"):
        split = "val"
        ds = k[:-1]
    else:
        split = "train"
        ds = k

    rows.append([split, ds, e, ne, e/t, ne/t])

pd.DataFrame(
    rows,
    columns=["split","dataset","empty","non_empty","empty_ratio","non_empty_ratio"]
).to_csv("segmentation_empty_stats.csv", index=False)