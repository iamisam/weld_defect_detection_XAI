import os
from tqdm import tqdm

# ---------------- CONFIG ----------------

DATASETS = {
    "finetuning": r"D:\weld_defect_project\datasets\segmentation_finetuning_augmented_dataset",
    "classification": r"D:\weld_defect_project\datasets\unified_classification_standardized_dataset"
}

SPLITS = ["val", "test"]

# ---------------- RULES ----------------

def is_riawelc(name):
    return name.startswith("RIAWELC")

def is_aug_finetuning(name):
    return any(x in name for x in ["_flip", "_rot", "_both"])

def is_aug_classification(name):
    return "_aug" in name

# ---------------- RUN ----------------

for dtype, base_dir in DATASETS.items():

    print(f"\nProcessing: {dtype}")

    for split in SPLITS:

        split_path = os.path.join(base_dir, split)

        if not os.path.exists(split_path):
            continue

        for root, _, files in os.walk(split_path):

            for f in tqdm(files, desc=f"{dtype}/{split}"):

                # 🔒 NEVER TOUCH RIAWELC
                if is_riawelc(f):
                    continue

                path = os.path.join(root, f)

                if dtype == "finetuning":
                    if is_aug_finetuning(f):
                        os.remove(path)

                elif dtype == "classification":
                    if is_aug_classification(f):
                        os.remove(path)

print("\nCleanup complete.")