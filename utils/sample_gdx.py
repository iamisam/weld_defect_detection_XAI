import os, cv2
import numpy as np
from tqdm import tqdm

SRC = "D:/weld_defect_project/datasets/segmentation_finetuning_augmented_dataset/train/gdx/images"
DST = "D:/weld_defect_project/datasets/segmentation_finetuning_augmented_dataset/train/gdx/sample_fix"

os.makedirs(DST, exist_ok=True)

def get_mask():
    h = w = 256
    cx = cy = 128
    Y, X = np.ogrid[:h, :w]
    return (np.abs(X - cx) + np.abs(Y - cy)) > 128

mask = get_mask()

files = [f for f in os.listdir(SRC) if "rot" in f][:50]

for name in tqdm(files):
    path = os.path.join(SRC, name)
    img = cv2.imread(path)
    if img is None:
        continue

    # distance transform to get nearest valid pixel
    inv_mask = (~mask).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(inv_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)

    h, w = mask.shape
    coords = np.column_stack(np.where(mask))

    for y, x in coords:
        label = labels[y, x] - 1
        ny, nx = divmod(label, w)
        img[y, x] = img[ny, nx]

    cv2.imwrite(os.path.join(DST, name), img)

print("Sample fix done.")