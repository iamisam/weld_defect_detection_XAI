import os
import cv2
from tqdm import tqdm

# ---------------- CONFIG ----------------

DATASETS = {
    "pretraining": r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset",
    "finetuning": r"D:\weld_defect_project\datasets\segmentation_finetuning_augmented_dataset"
}

SPLITS = ["train", "val", "test"]

# ---------------- HELPERS ----------------

def is_empty_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return True
    return mask.sum() == 0

# ---------------- RUN ----------------

for dname, base_dir in DATASETS.items():

    print(f"\n================ {dname.upper()} ================\n")

    total_all = 0
    empty_all = 0

    for split in SPLITS:

        mask_dir = os.path.join(base_dir, split, "masks")

        if not os.path.exists(mask_dir):
            continue

        files = os.listdir(mask_dir)

        total = 0
        empty = 0

        for f in tqdm(files, desc=f"{dname}/{split}"):

            mask_path = os.path.join(mask_dir, f)

            if is_empty_mask(mask_path):
                empty += 1

            total += 1

        total_all += total
        empty_all += empty

        pct = (empty / total * 100) if total > 0 else 0

        print(f"\n--- {split} ---")
        print(f"Total masks: {total}")
        print(f"Empty masks: {empty}")
        print(f"Empty %: {pct:.2f}%")

    overall_pct = (empty_all / total_all * 100) if total_all > 0 else 0

    print("\n--- OVERALL ---")
    print(f"Total masks: {total_all}")
    print(f"Empty masks: {empty_all}")
    print(f"Empty %: {overall_pct:.2f}%")