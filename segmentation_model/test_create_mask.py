import cv2
import os
import numpy as np
import pandas as pd


# ---- CONFIG ----

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "severstal-steel-defect-detection-dataset",
    "train_images",
    "0002cc93b.jpg",
)
CSV_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "severstal-steel-defect-detection-dataset",
    "train.csv"
)

# ----------------


def rle_decode(mask_rle, shape):
    s = list(map(int, mask_rle.split()))
    starts = np.array(s[0::2]) - 1
    lengths = np.array(s[1::2])

    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for start, end in zip(starts, ends):
        img[start:end] = 1

    # ✅ CRITICAL: column-major reshape
    return img.reshape(shape, order='F')


# ---- LOAD IMAGE ----
image = cv2.imread(IMAGE_PATH)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

h, w = image_gray.shape

# ---- LOAD CSV ----
df = pd.read_csv(CSV_PATH)

# Extract image id
image_id = IMAGE_PATH.split("\\")[-1]

print(df[df["EncodedPixels"].notna()].head())

# ---- GET MASKS FOR THIS IMAGE ----
rows = df[df["ImageId"] == image_id]

# Initialize empty mask
mask = np.zeros((h, w), dtype=np.uint8)

for _, row in rows.iterrows():
    if pd.isna(row["EncodedPixels"]):
        continue

    class_id = int(row["ClassId"])
    decoded = rle_decode(row["EncodedPixels"], (h, w))

    # Combine all classes into one mask
    mask = np.maximum(mask, decoded)

# ---- VISUALIZE ----
mask_255 = (mask * 255).astype(np.uint8)

# Overlay mask on image
overlay = image.copy()
overlay[mask == 1] = [0, 0, 255]  # red overlay

blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# ---- SHOW ----
cv2.imshow("Original", image)
cv2.imshow("Mask", mask_255)
cv2.imshow("Overlay", blended)

cv2.waitKey(0)
cv2.destroyAllWindows()