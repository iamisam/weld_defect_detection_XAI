import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

images_dir = Path(r"D:\weld_defect_project\datasets\segmentation_finetuning_augmented_dataset\train\gokul\images")
masks_dir  = Path(r"D:\weld_defect_project\datasets\segmentation_finetuning_augmented_dataset\train\gokul\masks")

def add_noise(img):
    strength = random.uniform(0.15, 0.30) * 255
    noise = np.random.normal(0, strength, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255)
    return img.astype(np.uint8)

def rotate(img, angle):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

def apply(img, mask, angle=None, flip=None):
    if angle:
        img = rotate(img, angle)
        mask = rotate(mask, angle)

    if flip == "h":
        img = cv2.flip(img, 1); mask = cv2.flip(mask, 1)
    elif flip == "v":
        img = cv2.flip(img, 0); mask = cv2.flip(mask, 0)
    elif flip == "hv":
        img = cv2.flip(img, -1); mask = cv2.flip(mask, -1)

    img = add_noise(img)
    return img, mask

combos = [
    (90,None),(180,None),(270,None),
    (None,"h"),(None,"v"),(None,"hv"),
    (90,"h"),(90,"v"),(90,"hv"),
    (180,"h"),(180,"v"),(180,"hv"),
    (270,"h"),(270,"v"),(270,"hv"),
    (45,"h"),(45,"v"),(45,"hv"),
    (135,"v"),(135,"h"),(135,"hv")
]

for mask_path in tqdm(list(masks_dir.glob("*.png")), desc="Aug empty"):
    mask = cv2.imread(str(mask_path), 0)
    if mask.max() != 0:
        continue  # only empty

    stem = mask_path.stem
    img = cv2.imread(str(images_dir / f"{stem}.png"), 0)

    for i, (angle, flip) in enumerate(combos):
        aug_img, aug_mask = apply(img.copy(), mask.copy(), angle, flip)
        name = f"{stem}_augE_{i}"

        cv2.imwrite(str(images_dir / f"{name}.png"), aug_img)
        cv2.imwrite(str(masks_dir  / f"{name}.png"), aug_mask)

print("Done")