import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths
images_dir = Path(r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset\train\dagm\images")
masks_dir = Path(r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset\train\dagm\masks")

# --- noise + blur (replacement) ---
def add_noise_blur(img):
    var = np.random.uniform(10, 50)
    noise = np.random.normal(0, var**0.5, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    k = np.random.choice([3, 5, 7])
    blurred = cv2.GaussianBlur(noisy, (k, k), 0)

    return blurred

# --- transforms ---
def apply_combo(img, mask, combo_name):

    # flips
    if "FlipH" in combo_name and "FlipV" not in combo_name:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    elif "FlipV" in combo_name and "FlipH" not in combo_name:
        img = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)
    elif "FlipH" in combo_name and "FlipV" in combo_name:
        img = cv2.flip(img, -1)
        mask = cv2.flip(mask, -1)

    # rotations
    if "Rot90" in combo_name:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif "Rot180" in combo_name:
        img = cv2.rotate(img, cv2.ROTATE_180)
        mask = cv2.rotate(mask, cv2.ROTATE_180)
    elif "Rot270" in combo_name:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # noise + blur (image only)
    img = add_noise_blur(img)

    return img, mask

# combos
combos = [
    "FlipH","Rot90","Rot180","Rot270","FlipV",
    "Rot90_FlipH","Rot180_FlipH","Rot270_FlipH",
    "Rot90_FlipV","Rot180_FlipV","Rot270_FlipV",
    "Rot90_FlipH_FlipV","Rot180_FlipH_FlipV","Rot270_FlipH_FlipV"
]

# collect non-empty masks
non_empty_masks = []
for mask_path in sorted(masks_dir.glob("*.png")):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask.max() > 0:
        non_empty_masks.append(mask_path)

print(f"Found {len(non_empty_masks)} non-empty masks")
print(f"Generating {len(combos)} augs per mask = {len(non_empty_masks)*len(combos)}")

# augment loop
for mask_path in tqdm(non_empty_masks, desc="Augmenting"):
    stem = mask_path.stem
    img_path = images_dir / f"{stem}.png"

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    for combo in combos:
        aug_img, aug_mask = apply_combo(img.copy(), mask.copy(), combo)

        aug_stem = f"{stem}_aug_{combo}"
        cv2.imwrite(str(images_dir / f"{aug_stem}.png"), aug_img)
        cv2.imwrite(str(masks_dir / f"{aug_stem}.png"), aug_mask)

print("Done!")

# verify
total_imgs = len(list(images_dir.glob("*.png")))
total_masks = len(list(masks_dir.glob("*.png")))
empty_count = sum(
    1 for m in masks_dir.glob("*.png")
    if cv2.imread(str(m), cv2.IMREAD_GRAYSCALE).max() == 0
)

print("\nFinal stats:")
print(f"Total images: {total_imgs}")
print(f"Total masks: {total_masks}")
print(f"Empty masks: {empty_count} ({empty_count/total_masks*100:.2f}%)")