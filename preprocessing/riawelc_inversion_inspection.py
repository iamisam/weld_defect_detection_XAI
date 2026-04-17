import os
import cv2
import random

# ---------------- CONFIG ----------------

BASE_DIR = r"D:\weld_defect_project\datasets\unified_classification_dataset\training"
OUT_DIR  = r"D:\weld_defect_project\datasets\riawelc_inversion_samples"

CLASSES = ["crack", "porosity", "lack_of_penetration", "no_defect"]
SAMPLES_PER_CLASS = 10

random.seed(42)

# ---------------- SETUP ----------------

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- PROCESS ----------------

for cls in CLASSES:

    class_dir = os.path.join(BASE_DIR, cls)
    out_class_dir = os.path.join(OUT_DIR, cls)

    os.makedirs(out_class_dir, exist_ok=True)

    # filter RIAWELC images only
    files = [f for f in os.listdir(class_dir) if f.startswith("RIAWELC")]

    if len(files) == 0:
        print(f"No RIAWELC images found in {cls}")
        continue

    # sample 10
    sample_files = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))

    for i, f in enumerate(sample_files):

        img_path = os.path.join(class_dir, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # invert
        inv = 255 - img

        base_name = f"{cls}_{i}"

        # save original
        cv2.imwrite(os.path.join(out_class_dir, base_name + "_orig.png"), img)

        # save inverted
        cv2.imwrite(os.path.join(out_class_dir, base_name + "_inv.png"), inv)

    print(f"{cls}: saved {len(sample_files)} samples")

print("\nDone.")