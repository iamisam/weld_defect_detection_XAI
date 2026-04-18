"""
Delete synthetic NEU images and masks (neu_synthetic_* files).
Usage: python delete_neu_synthetics.py --image_dir <path> --mask_dir <path>
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--mask_dir', required=True)
    args = parser.parse_args()

    for folder in [Path(args.image_dir), Path(args.mask_dir)]:
        files = list(folder.glob('neu_synthetic_*.png'))
        print(f"Found {len(files)} synthetic files in {folder}")
        for f in files:
            f.unlink()
        print(f"Deleted {len(files)} files.")

if __name__ == '__main__':
    main()