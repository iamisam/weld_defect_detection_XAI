import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE_DIR = r"D:\weld_defect_project\datasets\NEU_Surface_Defect_Dataset\NEU-DET"

OUTPUT_DIR = r"D:\weld_defect_project\datasets\segmentation_pretraining_dataset"
IMG_OUT = os.path.join(OUTPUT_DIR, "images")
MASK_OUT = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(MASK_OUT, exist_ok=True)

TARGET_SIZE = (256, 256)

# ---------------- PROCESS ----------------

def process_split(split_name):
    print(f"\nProcessing {split_name}...")

    img_root = os.path.join(BASE_DIR, split_name, "images")
    ann_root = os.path.join(BASE_DIR, split_name, "annotations")

    classes = os.listdir(img_root)

    for cls in classes:
        cls_path = os.path.join(img_root, cls)
        images = os.listdir(cls_path)

        for img_name in tqdm(images, desc=f"{split_name}-{cls}"):

            img_path = os.path.join(cls_path, img_name)
            xml_path = os.path.join(ann_root, img_name.replace(".jpg", ".xml"))

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            h, w = image.shape

            # ---- CREATE MASK ----
            mask = np.zeros((h, w), dtype=np.uint8)

            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()

                for obj in root.findall("object"):
                    bndbox = obj.find("bndbox")

                    xmin = int(bndbox.find("xmin").text)
                    ymin = int(bndbox.find("ymin").text)
                    xmax = int(bndbox.find("xmax").text)
                    ymax = int(bndbox.find("ymax").text)

                    # fill rectangle
                    mask[ymin:ymax, xmin:xmax] = 1

            # ---- RESIZE ----
            image_resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

            # ---- SAVE ----
            base_name = img_name.split(".")[0]
            filename = f"neu_{split_name}_{cls}_{base_name}.png"

            cv2.imwrite(os.path.join(IMG_OUT, filename), image_resized)
            cv2.imwrite(os.path.join(MASK_OUT, filename), mask_resized * 255)


# ---------------- RUN ----------------

process_split("train")
process_split("validation")

print("\nNEU preprocessing complete.")