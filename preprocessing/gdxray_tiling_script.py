import os
import cv2
import numpy as np
from tqdm import tqdm
import random

# ---------------- CONFIG ----------------

BASE_DIR = r"D:\weld_defect_project\datasets\GDXray_Dataset\Welds"

IMG_DIR = os.path.join(BASE_DIR, "W0001")
MASK_DIR = os.path.join(BASE_DIR, "W0002")

OUTPUT_DIR = r"D:\weld_defect_project\datasets\segmentation_finetuning_dataset"

IMG_OUT = os.path.join(OUTPUT_DIR, "images")
MASK_OUT = os.path.join(OUTPUT_DIR, "masks")

TILE_SIZE = 256
EMPTY_KEEP_RATIO = 0.30

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

# ---------------- PROCESS ----------------

tile_count = 0

image_files = sorted(os.listdir(IMG_DIR))

for img_name in tqdm(image_files):

    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, img_name.replace("W0001", "W0002"))

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    # ---- Ensure binary mask ----
    mask = (mask > 0).astype(np.uint8)

    h, w = image.shape

    # ---- TILING (2D) ----
    for y in range(0, h, TILE_SIZE):
        for x in range(0, w, TILE_SIZE):

            y_end = y + TILE_SIZE
            x_end = x + TILE_SIZE

            if y_end > h:
                y = h - TILE_SIZE
                y_end = h

            if x_end > w:
                x = w - TILE_SIZE
                x_end = w

            img_tile = image[y:y_end, x:x_end]
            mask_tile = mask[y:y_end, x:x_end]

            if img_tile.shape != (TILE_SIZE, TILE_SIZE):
                continue

            # ---- EMPTY FILTER ----
            if np.sum(mask_tile) == 0:
                if random.random() > EMPTY_KEEP_RATIO:
                    continue

            # ---- SAVE ----
            filename = f"gdx_{img_name.split('.')[0]}_{y}_{x}.png"

            cv2.imwrite(os.path.join(IMG_OUT, filename), img_tile)
            cv2.imwrite(os.path.join(MASK_OUT, filename), mask_tile * 255)

            tile_count += 1

print(f"Total GDXray tiles saved: {tile_count}")