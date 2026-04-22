import os, cv2, torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp

DEVICE = "cuda"
BASE = "D:/weld_defect_project"

IMG_ROOT = f"{BASE}/datasets/unified_classification_standardized_dataset"
SAVE_ROOT = f"{BASE}/datasets/unified_classification_masks_custom"

os.makedirs(SAVE_ROOT, exist_ok=True)

# ---------- LOAD MODEL ----------
model = smp.Unet("efficientnet-b2", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load(f"{BASE}/segmentation_model/ema/custom.pt"))
model = model.to(DEVICE).eval()

# ---------- PROCESS ----------
def process_split(split):
    in_root = os.path.join(IMG_ROOT, split)
    out_root = os.path.join(SAVE_ROOT, split)

    for cls in os.listdir(in_root):
        in_dir = os.path.join(in_root, cls)
        out_dir = os.path.join(out_root, cls)
        os.makedirs(out_dir, exist_ok=True)

        files = os.listdir(in_dir)

        for f in tqdm(files, desc=f"{split}/{cls}"):
            img_path = os.path.join(in_dir, f)
            save_path = os.path.join(out_dir, f)

            img = cv2.imread(img_path, 0)
            if img is None: continue

            img = img.astype(np.float32)/255.0
            img3 = np.stack([img,img,img], axis=0)

            x = torch.tensor(img3).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = torch.sigmoid(model(x))[0,0].cpu().numpy()

            # SAVE SOFT MASK (0–255)
            mask = (pred * 255).astype(np.uint8)

            cv2.imwrite(save_path, mask)

# ---------- RUN ----------
for split in ["train","val","test"]:
    process_split(split)

print("Masks generated.")