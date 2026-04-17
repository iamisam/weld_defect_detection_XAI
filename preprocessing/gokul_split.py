import os
import random
import shutil
from tqdm import tqdm
from collections import defaultdict

# ---------------- CONFIG ----------------

INPUT_DIR = r"D:\weld_defect_project\datasets\gokul_augmented"
OUTPUT_DIR = r"D:\weld_defect_project\datasets\gokul_split"

RATIOS = {
    "training": 0.65,
    "validation": 0.25,
    "testing": 0.10
}

random.seed(42)

# ---------------- PREP ----------------

classes = os.listdir(INPUT_DIR)

for split in RATIOS:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# ---------------- SPLIT ----------------

for cls in classes:

    class_path = os.path.join(INPUT_DIR, cls)
    images = os.listdir(class_path)

    # -------- GROUP FILES --------
    groups = defaultdict(list)

    for f in images:
        if "_orig" in f:
            base = f.split("_orig")[0]
        elif "_aug" in f:
            base = f.split("_aug")[0]
        else:
            base = f  # fallback

        groups[base].append(f)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    total = len(group_keys)

    n_train = int(total * RATIOS["training"])
    n_val = int(total * RATIOS["validation"])

    train_keys = group_keys[:n_train]
    val_keys = group_keys[n_train:n_train + n_val]
    test_keys = group_keys[n_train + n_val:]

    # ensure at least 1 group in small classes
    if len(val_keys) == 0 and total > 2:
        val_keys = group_keys[n_train:n_train+1]
    if len(test_keys) == 0 and total > 3:
        test_keys = group_keys[n_train+len(val_keys):n_train+len(val_keys)+1]

    split_map = {
        "training": train_keys,
        "validation": val_keys,
        "testing": test_keys
    }

    # -------- COPY FILES --------
    for split, keys in split_map.items():
        for key in keys:
            for f in groups[key]:
                src = os.path.join(class_path, f)
                dst = os.path.join(OUTPUT_DIR, split, cls, f)
                shutil.copy(src, dst)

    print(f"{cls}: groups={total}, train={len(train_keys)}, val={len(val_keys)}, test={len(test_keys)}")

print("\nSplitting complete.")