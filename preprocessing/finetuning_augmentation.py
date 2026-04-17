import os
import cv2
import numpy as np
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE_DIR = r"D:\weld_defect_project\datasets\segmentation_finetuning_dataset\split"
OUT_DIR  = r"D:\weld_defect_project\datasets\segmentation_finetuning_augmented_dataset"

# match your split names exactly
SPLITS = ["train", "val", "test"]  # or ["training","validation","testing"]

ANGLES = [45, 90, 135, 180, 225, 270, 315]

# intensities
NOISE_BOTH = 0.08
BLUR_KSIZE = 3  # very mild

# ---------------- HELPERS ----------------

def rotate(img, angle, is_mask=False):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(
        img, M, (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

def flip(img):
    return cv2.flip(img, 1)

def add_noise(img, intensity):
    sigma = intensity * 255
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def add_blur(img):
    return cv2.GaussianBlur(img, (BLUR_KSIZE, BLUR_KSIZE), 0)

def save_pair(out_img_dir, out_mask_dir, name, img, mask):
    cv2.imwrite(os.path.join(out_img_dir, name + ".png"), img)
    cv2.imwrite(os.path.join(out_mask_dir, name + ".png"), mask)

def photometric_variants(img):
    # returns dict of 4 versions: base, noise, blur, both
    v = {}
    v["base"]  = img
    v["both"]  = add_blur(add_noise(img, NOISE_BOTH))
    return v

# ---------------- PROCESS ----------------

for split in SPLITS:

    in_img_dir  = os.path.join(BASE_DIR, split, "images")
    in_mask_dir = os.path.join(BASE_DIR, split, "masks")

    out_img_dir  = os.path.join(OUT_DIR, split, "images")
    out_mask_dir = os.path.join(OUT_DIR, split, "masks")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    files = os.listdir(in_img_dir)

    for f in tqdm(files, desc=f"Augmenting {split}"):

        img_path  = os.path.join(in_img_dir, f)
        mask_path = os.path.join(in_mask_dir, f)

        if not os.path.exists(mask_path):
            continue

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        base = os.path.splitext(f)[0]

        # -------- original --------
        pv = photometric_variants(img)
        save_pair(out_img_dir, out_mask_dir, f"{base}",           pv["base"],  mask)
        save_pair(out_img_dir, out_mask_dir, f"{base}_both",      pv["both"],  mask)

        # -------- flipped --------
        f_img  = flip(img)
        f_mask = flip(mask)

        pv = photometric_variants(f_img)
        save_pair(out_img_dir, out_mask_dir, f"{base}_flip",          pv["base"],  f_mask)
        save_pair(out_img_dir, out_mask_dir, f"{base}_flip_both",     pv["both"],  f_mask)

        # -------- rotations (no flip) --------
        for ang in ANGLES:
            r_img  = rotate(img,  ang, is_mask=False)
            r_mask = rotate(mask, ang, is_mask=True)

            pv = photometric_variants(r_img)
            tag = f"{base}_rot{ang}"
            save_pair(out_img_dir, out_mask_dir, f"{tag}",         pv["base"],  r_mask)
            save_pair(out_img_dir, out_mask_dir, f"{tag}_both",    pv["both"],  r_mask)

        # -------- rotations + flipped --------
        for ang in ANGLES:
            r_img  = rotate(img,  ang, is_mask=False)
            r_mask = rotate(mask, ang, is_mask=True)

            rf_img  = flip(r_img)
            rf_mask = flip(r_mask)

            pv = photometric_variants(rf_img)
            tag = f"{base}_rot{ang}_flip"
            save_pair(out_img_dir, out_mask_dir, f"{tag}",         pv["base"],  rf_mask)
            save_pair(out_img_dir, out_mask_dir, f"{tag}_both",    pv["both"],  rf_mask)

print("Augmentation complete.")