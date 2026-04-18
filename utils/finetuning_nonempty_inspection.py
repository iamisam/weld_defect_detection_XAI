import cv2
from pathlib import Path

datasets = {
    "gokul": {
        "images": Path(r"D:\weld_defect_project\datasets\segmentation_finetuning_augmented_dataset\train\gokul\images"),
        "masks":  Path(r"D:\weld_defect_project\datasets\segmentation_finetuning_augmented_dataset\train\gokul\masks")
    },
    "gdx": {
        "images": Path(r"D:\weld_defect_project\datasets\segmentation_finetuning_augmented_dataset\train\gdx\images"),
        "masks":  Path(r"D:\weld_defect_project\datasets\segmentation_finetuning_augmented_dataset\train\gdx\masks")
    }
}

for name, paths in datasets.items():
    masks_dir = paths["masks"]

    total = empty = non_empty = 0

    for mask_path in masks_dir.glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        total += 1
        if mask.max() == 0:
            empty += 1
        else:
            non_empty += 1

    pct = (empty / total * 100) if total > 0 else 0

    print(f"\n[{name}]")
    print(f"Total: {total}")
    print(f"Empty: {empty}")
    print(f"Non-empty: {non_empty}")
    print(f"Empty %: {pct:.2f}%")