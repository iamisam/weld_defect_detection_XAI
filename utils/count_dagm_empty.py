import cv2
import numpy as np
from pathlib import Path

# Paths
images_dir = Path(r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset\train\dagm\images")
masks_dir = Path(r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset\train\dagm\masks")

# Count
empty_count = 0
total_count = 0

for mask_path in sorted(masks_dir.glob("*.png")):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    total_count += 1
    if mask.max() == 0:
        empty_count += 1

non_empty = total_count - empty_count
current_empty_pct = (empty_count / total_count * 100) if total_count > 0 else 0

print(f"DAGM Train Stats:")
print(f"  Total: {total_count}")
print(f"  Empty: {empty_count} ({current_empty_pct:.2f}%)")
print(f"  Non-empty: {non_empty}")

# Calculate augs needed to hit 32% empty
target_empty_pct = 32.0

if current_empty_pct > target_empty_pct:
    # Need to add non-empty images
    # Formula: empty / (empty + non_empty + X) = 0.32
    # Solve for X (new non-empty images needed)
    # empty = 0.32 * (empty + non_empty + X)
    # empty = 0.32*empty + 0.32*non_empty + 0.32*X
    # 0.68*empty = 0.32*non_empty + 0.32*X
    # X = (0.68*empty - 0.32*non_empty) / 0.32
    
    X = (empty_count / target_empty_pct * 100) - total_count
    
    print(f"\nTo reach {target_empty_pct}% empty:")
    print(f"  Need to add ~{int(np.ceil(X))} augmented non-empty images")
    print(f"  Aug per non-empty image: {X / non_empty:.2f}")
    print(f"  Suggested: {int(np.ceil(X / non_empty))} augs per non-empty image")
else:
    print(f"\nAlready below {target_empty_pct}% empty")