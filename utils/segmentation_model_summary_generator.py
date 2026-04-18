import cv2
from pathlib import Path
from collections import Counter

root = Path(r"D:\weld_defect_project\datasets")

datasets = {
    "pretraining": root / "segmentation_pretraining_dataset",
    "finetuning": root / "segmentation_finetuning_augmented_dataset"
}

def analyze_split(split_path):
    stats = {}

    for ds in split_path.iterdir():
        if not ds.is_dir(): continue

        images_dir = ds / "images"
        masks_dir  = ds / "masks"

        total = empty = non_empty = 0
        res_counter = Counter()

        for m in masks_dir.glob("*.png"):
            mask = cv2.imread(str(m), 0)
            if mask is None: continue

            h, w = mask.shape
            res_counter[(h,w)] += 1

            total += 1
            if mask.max() == 0:
                empty += 1
            else:
                non_empty += 1

        pct_empty = (empty/total*100) if total else 0
        pct_non   = (non_empty/total*100) if total else 0

        stats[ds.name] = {
            "total": total,
            "empty": empty,
            "non_empty": non_empty,
            "empty_pct": pct_empty,
            "non_empty_pct": pct_non,
            "resolutions": dict(res_counter)
        }

    return stats

full = {}

for k, base in datasets.items():
    full[k] = {}
    for split in ["train","val","test"]:
        split_path = base / split
        if split_path.exists():
            full[k][split] = analyze_split(split_path)

# -------- PRINT SUMMARY --------

print("\n===== DATASET CONTEXT SUMMARY =====\n")

for phase, splits in full.items():
    print(f"\n## {phase.upper()}\n")
    for split, dsets in splits.items():
        print(f"### {split}")
        for name, s in dsets.items():
            print(f"\n[{name}]")
            print(f"Total: {s['total']}")
            print(f"Empty: {s['empty']} ({s['empty_pct']:.2f}%)")
            print(f"Non-empty: {s['non_empty']} ({s['non_empty_pct']:.2f}%)")
            print(f"Resolutions: {list(s['resolutions'].keys())}")

# -------- TRAINING PLAN BLOCK --------

print("\n\n===== TRAINING PLAN =====\n")

print("""
Model:
- U-Net + EMA
- Encoder: EfficientNet-B2 (ImageNet pretrained)

Loss:
- Dice + BCE + Focal

Curriculum Learning Order:
1. NEU  (coarse defects)
2. DAGM (background / negatives)
3. SEV  (fine defects)
4. GOKUL (real coarse defects)
5. GDX   (fine sparse precision)

Key Rules:
- train: augmented
- val/test: clean
- maintain empty/non-empty distribution
- focus on structure, not intensity

Pipeline:
pretrain → finetune → evaluate (TTA ready)
""")