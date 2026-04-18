import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

images_dir = Path(r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset\train\neu\images")
masks_dir  = Path(r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset\train\neu\masks")

def add_noise_blur(img):
    var = np.random.uniform(10, 50)
    noise = np.random.normal(0, var**0.5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    k = random.choice([3,5,7])
    return cv2.GaussianBlur(img, (k,k), 0)

def apply_combo(img, mask, rot):
    # Flip H+V
    img = cv2.flip(img, -1)
    mask = cv2.flip(mask, -1)

    # Rotation
    if rot == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
        mask = cv2.rotate(mask, cv2.ROTATE_180)
    elif rot == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # noise+blur (image only)
    img = add_noise_blur(img)

    return img, mask

rotations = [0, 90, 180, 270]

mask_paths = list(masks_dir.glob("*.png"))

for mask_path in tqdm(mask_paths, desc="Augmenting NEU"):
    stem = mask_path.stem
    img_path = images_dir / f"{stem}.png"

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    rot = random.choice(rotations)

    aug_img, aug_mask = apply_combo(img, mask, rot)

    new_name = f"{stem}_aug_r{rot}"
    cv2.imwrite(str(images_dir / f"{new_name}.png"), aug_img)
    cv2.imwrite(str(masks_dir  / f"{new_name}.png"), aug_mask)

print("Done")