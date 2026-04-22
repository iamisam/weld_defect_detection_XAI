import os
import cv2

ROOT = "D:/weld_defect_project/datasets/segmentation_finetuning_augmented_dataset"

for split in ["train", "val", "test"]:
    img_dir = os.path.join(ROOT, split, "gdx", "images")

    for name in os.listdir(img_dir):
        path = os.path.join(img_dir, name)

        img = cv2.imread(path)
        if img is None:
            continue

        inv = 255 - img
        cv2.imwrite(path, inv)

print("Done. GDX images inverted.")