import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

# ---------------- CONFIG ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEV_IMG_DIR = os.path.join(BASE_DIR, "datasets", "severstal-steel-defect-detection-dataset", "train_images")
CSV_PATH = os.path.join(BASE_DIR, "datasets", "severstal-steel-defect-detection-dataset", "train.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "segmentation_pretraining_dataset")
IMG_OUT = os.path.join(OUTPUT_DIR, "images")
MASK_OUT = os.path.join(OUTPUT_DIR, "masks")

TILE_SIZE = 256
EMPTY_KEEP_RATIO = 0.15  # keep 25% empty tiles

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

# ---------------- RLE DECODE ----------------

def rle_decode(mask_rle, shape):
    s = list(map(int, mask_rle.split()))
    starts = np.array(s[0::2]) - 1
    lengths = np.array(s[1::2])
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for start, end in zip(starts, ends):
        img[start:end] = 1

    return img.reshape(shape, order='F')  # CRITICAL


# ---------------- LOAD CSV ----------------

df = pd.read_csv(CSV_PATH)

# group all rows per image
grouped = df.groupby("ImageId")

# ---------------- PROCESS ----------------

tile_count = 0

image_files = os.listdir(SEV_IMG_DIR)

for img_name in tqdm(image_files):

    img_path = os.path.join(SEV_IMG_DIR, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    h, w = image.shape

    # ---- CREATE MASK ----
    mask = np.zeros((h, w), dtype=np.uint8)

    if img_name in grouped.groups:
        rows = grouped.get_group(img_name)

        for _, row in rows.iterrows():
            if pd.isna(row["EncodedPixels"]):
                continue

            decoded = rle_decode(row["EncodedPixels"], (h, w))
            mask = np.maximum(mask, decoded)

    # ---- TILING ----
    for x in range(0, w, TILE_SIZE):

        x_end = x + TILE_SIZE

        if x_end > w:
            x = w - TILE_SIZE
            x_end = w

        img_tile = image[:, x:x_end]
        mask_tile = mask[:, x:x_end]

        # skip incorrect sizes
        if img_tile.shape != (TILE_SIZE, TILE_SIZE):
            continue

        # ---- EMPTY FILTER ----
        if np.sum(mask_tile) == 0:
            if random.random() > EMPTY_KEEP_RATIO:
                continue

        # ---- SAVE ----
        filename = f"sev_{img_name.split('.')[0]}_{x}.png"

        cv2.imwrite(os.path.join(IMG_OUT, filename), img_tile)
        cv2.imwrite(os.path.join(MASK_OUT, filename), mask_tile * 255)

        tile_count += 1

print(f"Total tiles saved: {tile_count}")