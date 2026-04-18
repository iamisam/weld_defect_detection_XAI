"""
Analyze folder structure and image file types for all datasets.
Usage: python analyze_structure.py --root <path_to_datasets_root>
"""

import argparse
from pathlib import Path
from collections import defaultdict


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def analyze_dir(directory: Path):
    ext_counts = defaultdict(int)
    total = 0
    for f in directory.rglob('*'):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            ext_counts[f.suffix.lower()] += 1
            total += 1
    return total, dict(ext_counts)


def print_tree(root: Path, prefix='', max_depth=4, current_depth=0):
    if current_depth > max_depth:
        return
    children = sorted(root.iterdir()) if root.is_dir() else []
    dirs = [c for c in children if c.is_dir()]
    files = [c for c in children if c.is_file() and c.suffix.lower() in IMAGE_EXTS]

    for d in dirs:
        print(f"{prefix}📁 {d.name}/")
        print_tree(d, prefix + '    ', max_depth, current_depth + 1)

    if files:
        ext_summary = defaultdict(int)
        for f in files:
            ext_summary[f.suffix.lower()] += 1
        summary = ', '.join(f"{cnt}x {ext}" for ext, cnt in sorted(ext_summary.items()))
        print(f"{prefix}🖼  {len(files)} images [{summary}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Root path containing all datasets')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} does not exist.")
        return

    print(f"\n{'='*60}")
    print(f"  DATASET ROOT: {root}")
    print(f"{'='*60}\n")

    dataset_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    if not dataset_dirs:
        print("No subdirectories found. Analyzing root directly.\n")
        dataset_dirs = [root]

    for ds in dataset_dirs:
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds.name}")
        print(f"{'='*60}")
        print_tree(ds)
        total, exts = analyze_dir(ds)
        print(f"\n  TOTAL images: {total}")
        print(f"  Extensions  : {exts}")
        print()


if __name__ == '__main__':
    main()