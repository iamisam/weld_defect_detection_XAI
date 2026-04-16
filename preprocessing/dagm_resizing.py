import os
import cv2
import numpy as np
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE_DIR = r"D:\weld_defect_project\datasets\DAGM_Dataset\Class10"

OUTPUT_DIR = r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset"
IMG_OUT = os.path.join(OUTPUT_DIR, "images")
MASK_OUT = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

TARGET_SIZE = (256, 256)

# ---------------- PROCESS ----------------

def process_split(split_name):
    print(f"\nProcessing {split_name}...")

    split_path = os.path.join(BASE_DIR, split_name)
    label_path = os.path.join(split_path, "Label")

    images = [f for f in os.listdir(split_path) if f.lower().endswith(".png")]

    for img_name in tqdm(images):

        img_path = os.path.join(split_path, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        base_name = img_name.split(".")[0]
        mask_name = f"{base_name}_label.PNG"
        mask_path = os.path.join(label_path, mask_name)

        # ---- LOAD OR CREATE MASK ----
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros_like(image, dtype=np.uint8)

        # ---- RESIZE ----
        image_resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        # ---- SAVE ----
        filename = f"dagm_{split_name.lower()}_{base_name}.png"

        cv2.imwrite(os.path.join(IMG_OUT, filename), image_resized)
        cv2.imwrite(os.path.join(MASK_OUT, filename), mask_resized * 255)


# ---------------- RUN ----------------

process_split("Train")
process_split("Test")

print("\nDAGM preprocessing complete.")