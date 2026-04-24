import os, cv2, random
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE = "D:/weld_defect_project/datasets"

IMG_ROOT   = f"{BASE}/unified_classification_standardized_dataset/train"
MASK_ROOT1 = f"{BASE}/unified_classification_masks/train"
MASK_ROOT2 = f"{BASE}/unified_classification_masks_custom/train"

PLAN_CSV = "D:/weld_defect_project/classification_model/logs/augmentation_plan.csv"

EXCLUDE = {"crack","porosity","lack_of_penetration","no_defect"}

df = pd.read_csv(PLAN_CSV)

# ---------- GEOM (SYNCED) ----------
def apply_geom(img, m1, m2):
    if random.random() < 0.5:
        img = np.flip(img,1).copy()
        m1  = np.flip(m1,1).copy()
        m2  = np.flip(m2,1).copy()

    if random.random() < 0.5:
        k = random.choice([1,2,3])
        img = np.rot90(img,k).copy()
        m1  = np.rot90(m1,k).copy()
        m2  = np.rot90(m2,k).copy()

    return img, m1, m2

# ---------- INTENSITY (IMG ONLY) ----------
def apply_intensity(img):
    if random.random() < 0.5:
        img = np.clip(img * random.uniform(0.85,1.15), 0,255)

    if random.random() < 0.5:
        noise = np.random.normal(0,5,img.shape)
        img = np.clip(img + noise,0,255)

    if random.random() < 0.3:
        img = cv2.GaussianBlur(img,(3,3),0)

    return img

# ---------- MAIN ----------
for _, row in df.iterrows():
    cls = row["class"]
    aug_n = int(row["augmentations_per_image"])

    if cls in EXCLUDE or aug_n == 0:
        continue

    img_dir = os.path.join(IMG_ROOT, cls)
    m1_dir  = os.path.join(MASK_ROOT1, cls)
    m2_dir  = os.path.join(MASK_ROOT2, cls)

    files = [f for f in os.listdir(img_dir) if "augp" not in f]

    print(f"\n{cls} | {len(files)} imgs | {aug_n} aug/img")

    for f in tqdm(files):
        img = cv2.imread(os.path.join(img_dir,f),0)
        m1  = cv2.imread(os.path.join(m1_dir,f),0)
        m2  = cv2.imread(os.path.join(m2_dir,f),0)

        if img is None or m1 is None or m2 is None:
            continue

        base = f.split(".")[0]

        for i in range(aug_n):
            img_aug, m1_aug, m2_aug = apply_geom(img, m1, m2)
            img_aug = apply_intensity(img_aug)

            name = f"{base}_augp{i}.png"

            cv2.imwrite(os.path.join(img_dir,name), img_aug.astype(np.uint8))
            cv2.imwrite(os.path.join(m1_dir,name), m1_aug.astype(np.uint8))
            cv2.imwrite(os.path.join(m2_dir,name), m2_aug.astype(np.uint8))

print("DONE")