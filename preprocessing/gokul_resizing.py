import os
import cv2
import numpy as np
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE_DIR = r"D:\weld_defect_project\datasets\gokul_weld_defects"

OUTPUT_DIR = r"D:\weld_defect_project\datasets\segmentation_finetuning_dataset"
IMG_OUT = os.path.join(OUTPUT_DIR, "images")
MASK_OUT = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

TARGET_SIZE = (256, 256)

# ---------------- PROCESS ----------------

def process_split(split):
    print(f"\nProcessing {split}...")

    img_dir = os.path.join(BASE_DIR, "images", split)
    label_dir = os.path.join(BASE_DIR, "labels", split)

    images = os.listdir(img_dir)

    for img_name in tqdm(images):

        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        h, w = image.shape

        # ---- CREATE MASK ----
        mask = np.zeros((h, w), dtype=np.uint8)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                _, xc, yc, bw, bh = map(float, parts)

                # Convert to pixel coords
                xmin = int((xc - bw / 2) * w)
                xmax = int((xc + bw / 2) * w)
                ymin = int((yc - bh / 2) * h)
                ymax = int((yc + bh / 2) * h)

                # Clip bounds (IMPORTANT)
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)

                mask[ymin:ymax, xmin:xmax] = 1

        # ---- RESIZE ----
        image_resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        # ---- SAVE ----
        base = img_name.split(".")[0]
        filename = f"gokul_{split}_{base}.png"

        cv2.imwrite(os.path.join(IMG_OUT, filename), image_resized)
        cv2.imwrite(os.path.join(MASK_OUT, filename), mask_resized * 255)


# ---------------- RUN ----------------

process_split("train")
process_split("val")

print("\nGokul preprocessing complete.")