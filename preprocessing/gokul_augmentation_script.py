import os
import cv2
import numpy as np
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE_DIR = r"D:\weld_defect_project\datasets\gokul_weld_defects\images"
SPLITS = ["train", "val"]

OUTPUT_DIR = r"D:\weld_defect_project\datasets\gokul_augmented"
TARGET_SIZE = (256, 256)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- CLASS MAP ----------------

CLASS_MAP = {
    "air-hole1": "air_hole",
    "air-hole2": "air_hole",
    "slag-inclusion2": "slag_inclusion",
    "bite-edge2": "bite_edge",
    "broken-arc1": "broken_arc",
    "broken-arc2": "broken_arc",
    "crack": "crack",
    "overlap": "overlap",
    "unfused": "lack_of_fusion"
}

DIRECTIONAL = {"overlap", "lack_of_fusion", "bite_edge", "broken_arc"}
NON_DIRECTIONAL = {"air_hole", "slag_inclusion"}

# ---------------- HELPERS ----------------

def extract_class_name(filename):
    name = filename.split(".")[0]
    if "-" in name:
        return "-".join(name.split("-")[:-1])
    else:
        i = len(name)
        while i > 0 and name[i-1].isdigit():
            i -= 1
        return name[:i]

def add_grayscale_noise(img, sigma=8):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def clahe(img):
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return c.apply(img)

def mirror(img):
    return cv2.flip(img, 1)

def rot180(img):
    return cv2.rotate(img, cv2.ROTATE_180)

# ---------------- PROCESS ----------------

for split in SPLITS:
    split_dir = os.path.join(BASE_DIR, split)
    files = [f for f in os.listdir(split_dir) if f.endswith(".jpg")]

    for f in tqdm(files, desc=f"Processing {split}"):

        img_path = os.path.join(split_dir, f)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        base_class = extract_class_name(f)
        if base_class not in CLASS_MAP:
            continue

        final_class = CLASS_MAP[base_class]

        class_dir = os.path.join(OUTPUT_DIR, final_class)
        os.makedirs(class_dir, exist_ok=True)

        base_name = f"{split}_{f.split('.')[0]}"  # include split to avoid collisions

        x = cv2.resize(image, TARGET_SIZE)

        # save original
        cv2.imwrite(os.path.join(class_dir, base_name + "_orig.png"), x)

        # -------- crack: no augmentation --------
        if final_class == "crack":
            continue

        # -------- directional --------
        if final_class in DIRECTIONAL:
            variants = [
                mirror(x),
                rot180(x),
                mirror(rot180(x))
            ]

            for i, v in enumerate(variants):
                v = add_grayscale_noise(v, sigma=8)
                cv2.imwrite(os.path.join(class_dir, f"{base_name}_aug{i}.png"), v)

        # -------- non-directional --------
        elif final_class in NON_DIRECTIONAL:
            variants = [
                clahe(x),
                mirror(x),
                clahe(mirror(x)),
                rot180(x),
                mirror(rot180(x)),
                clahe(rot180(x)),
                clahe(mirror(rot180(x)))
            ]

            for i, v in enumerate(variants):
                cv2.imwrite(os.path.join(class_dir, f"{base_name}_aug{i}.png"), v)

print("Augmentation complete.")