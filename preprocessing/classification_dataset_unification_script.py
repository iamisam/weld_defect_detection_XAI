import os
import shutil
from tqdm import tqdm

# ---------------- CONFIG ----------------

RIAWELC_DIR = r"D:\weld_defect_project\datasets\RIAWELC_dataset"
GOKUL_DIR = r"D:\weld_defect_project\datasets\gokul_split"
OUTPUT_DIR = r"D:\weld_defect_project\datasets\unified_classification_dataset"

SPLITS = ["training", "validation", "testing"]

# ---------------- HELPER ----------------

def copy_dataset(src_root, split):

    src_split_path = os.path.join(src_root, split)

    if not os.path.exists(src_split_path):
        return

    classes = os.listdir(src_split_path)

    # count total files for tqdm
    total_files = 0
    for cls in classes:
        total_files += len(os.listdir(os.path.join(src_split_path, cls)))

    dataset_name = os.path.basename(src_root)

    with tqdm(total=total_files, desc=f"{dataset_name} | {split}", leave=True) as pbar:

        for cls in classes:

            src_class_path = os.path.join(src_split_path, cls)
            dst_class_path = os.path.join(OUTPUT_DIR, split, cls)

            os.makedirs(dst_class_path, exist_ok=True)

            files = os.listdir(src_class_path)

            for f in files:

                src = os.path.join(src_class_path, f)
                prefix = os.path.basename(src_root)
                dst_name = f"{prefix}_{f}"
                dst = os.path.join(dst_class_path, dst_name)

                shutil.copy(src, dst)

                pbar.update(1)

# ---------------- RUN ----------------

for split in tqdm(SPLITS, desc="Splits", leave=True):
    copy_dataset(RIAWELC_DIR, split)
    copy_dataset(GOKUL_DIR, split)

print("\nUnified dataset created successfully.")