import os
import cv2
import shutil
from tqdm import tqdm

# ---------------- CONFIG ----------------

INPUT_DIR  = r"D:\weld_defect_project\datasets\unified_classification_dataset"
OUTPUT_DIR = r"D:\weld_defect_project\datasets\unified_classification_standardized_dataset"

SPLITS = ["training", "validation", "testing"]  # adjust if needed

TARGET_SIZE = (256, 256)

# ---------------- HELPERS ----------------

def process_riawelc(img):
    # invert
    img = 255 - img
    # resize
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    return img

# ---------------- RUN ----------------

for split in SPLITS:

    split_in  = os.path.join(INPUT_DIR, split)
    split_out = os.path.join(OUTPUT_DIR, split)

    classes = os.listdir(split_in)

    for cls in classes:

        in_dir  = os.path.join(split_in, cls)
        out_dir = os.path.join(split_out, cls)

        os.makedirs(out_dir, exist_ok=True)

        files = os.listdir(in_dir)

        for f in tqdm(files, desc=f"{split}/{cls}"):

            src = os.path.join(in_dir, f)
            dst = os.path.join(out_dir, f)

            # identify RIAWELC
            if f.startswith("RIAWELC"):

                img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = process_riawelc(img)
                cv2.imwrite(dst, img)

            else:
                # gokul or others → copy as-is
                shutil.copy(src, dst)

print("Standardization complete.")