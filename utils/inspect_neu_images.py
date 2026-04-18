"""
Analyze NEU masks: count images where defect is strictly in left or right half.
Usage: python analyze_neu_halves.py --mask_dir <path_to_neu_masks>
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def check_half(mask_arr):
    """
    Returns 'left', 'right', or None.
    'left'  → all nonzero pixels are in left half
    'right' → all nonzero pixels are in right half
    None    → spans both halves or empty mask
    """
    h, w = mask_arr.shape[:2]
    mid = w // 2

    nonzero = np.argwhere(mask_arr > 0)
    if len(nonzero) == 0:
        return None  # empty mask, skip

    xs = nonzero[:, 1]  # column indices
    if xs.max() < mid:
        return 'left'
    elif xs.min() >= mid:
        return 'right'
    return None  # spans both


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_dir', required=True, help='Path to NEU masks folder')
    parser.add_argument('--exts', default='.png,.jpg,.bmp', help='Mask file extensions')
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    exts = set(args.exts.split(','))

    left_files, right_files, both_files, empty_files = [], [], [], []

    for f in sorted(mask_dir.rglob('*')):
        if f.suffix.lower() not in exts:
            continue
        if not f.name.lower().startswith('neu_'):
            continue
        try:
            mask = np.array(Image.open(f).convert('L'))
        except Exception as e:
            print(f"SKIP {f.name}: {e}")
            continue

        result = check_half(mask)
        if result == 'left':
            left_files.append(f.name)
        elif result == 'right':
            right_files.append(f.name)
        elif np.any(mask > 0):
            both_files.append(f.name)
        else:
            empty_files.append(f.name)

    total_usable = len(left_files) + len(right_files)

    print(f"\n{'='*50}")
    print(f"  Mask-only LEFT half  : {len(left_files)}")
    print(f"  Mask-only RIGHT half : {len(right_files)}")
    print(f"  Spans BOTH halves    : {len(both_files)}")
    print(f"  Already EMPTY masks  : {len(empty_files)}")
    for e in empty_files:
        print(f.name)
    print(f"{'='*50}")
    print(f"  Usable for synthetic : {total_usable}")
    print(f"{'='*50}\n")

    if left_files:
        print("LEFT half files:")
        for n in left_files:
            print(f"  {n}")

    if right_files:
        print("\nRIGHT half files:")
        for n in right_files:
            print(f"  {n}")


if __name__ == '__main__':
    main()