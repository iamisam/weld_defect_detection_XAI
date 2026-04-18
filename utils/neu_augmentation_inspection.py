import cv2
from pathlib import Path

images_dir = Path(r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset\train\neu\images")
masks_dir  = Path(r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset\train\neu\masks")

total = 0
empty = 0
non_empty = 0

for mask_path in masks_dir.glob("*.png"):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    total += 1
    
    if mask is None:
        continue
    
    if mask.max() == 0:
        empty += 1
    else:
        non_empty += 1

empty_pct = (empty / total) * 100 if total > 0 else 0

print(f"Total images: {total}")
print(f"Empty masks: {empty}")
print(f"Non-empty masks: {non_empty}")
print(f"Empty %: {empty_pct:.2f}%")