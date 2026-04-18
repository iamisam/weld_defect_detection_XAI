"""
Split segmentation datasets into per-source subfolders.

Output structure:
  split/source/images/
  split/source/masks/

Usage:
  python split_seg_datasets.py --pretrain_root <path> --finetune_root <path>
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

PRETRAIN_PREFIXES = ['neu_', 'dagm_', 'sev_']
FINETUNE_PREFIXES = ['gdx_', 'gokul_']
SPLITS = ['train', 'val', 'test']


def get_source(filename: str, prefixes: list):
    name = filename.lower()
    for p in prefixes:
        if name.startswith(p):
            return p.rstrip('_')
    return None


def split_dataset(root: Path, prefixes: list, label: str):
    print(f"\n{'='*55}")
    print(f"  Processing: {label} → {root}")
    print(f"{'='*55}")

    counts = {}

    for split in SPLITS:
        img_dir = root / split / 'images'
        msk_dir = root / split / 'masks'

        if not img_dir.exists():
            print(f"  SKIP {split}: images folder not found")
            continue

        images = sorted(img_dir.glob('*.png'))
        print(f"\n  [{split}] {len(images)} images found")

        for img_path in tqdm(images, desc=f"  {split}", unit='img'):
            source = get_source(img_path.name, prefixes)
            if source is None:
                print(f"    WARN: no prefix match → {img_path.name}")
                continue

            # Destination dirs
            dst_img = root / split / source / 'images'
            dst_msk = root / split / source / 'masks'
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_msk.mkdir(parents=True, exist_ok=True)

            # Copy image
            shutil.copy2(img_path, dst_img / img_path.name)

            # Copy mask if exists
            msk_path = msk_dir / img_path.name
            if msk_path.exists():
                shutil.copy2(msk_path, dst_msk / img_path.name)
            else:
                print(f"    WARN: mask missing for {img_path.name}")

            key = (split, source)
            counts[key] = counts.get(key, 0) + 1

    print(f"\n  SUMMARY for {label}:")
    for (split, source), cnt in sorted(counts.items()):
        print(f"    {split:6s} / {source:10s} → {cnt} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_root', required=True,
                        help='Path to segmentation_pretraining_dataset')
    parser.add_argument('--finetune_root', required=True,
                        help='Path to segmentation_finetuning_augmented_dataset')
    args = parser.parse_args()

    split_dataset(Path(args.pretrain_root), PRETRAIN_PREFIXES, 'PRETRAINING')
    split_dataset(Path(args.finetune_root), FINETUNE_PREFIXES, 'FINETUNING')

    print(f"\n{'='*55}")
    print("  Done. Original files kept. New subfolders created.")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()