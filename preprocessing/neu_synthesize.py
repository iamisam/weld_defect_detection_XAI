"""
Generate synthetic NEU negative samples — within-class cross-product only.

Logic:
  - Group left-clean and right-clean halves by defect class
  - Cross-product ONLY within same class
  - scratches capped at 16x16=256
  - Empty mask for all ~653 synthetics

Usage:
  python gen_neu_synthetics.py \
    --image_dir <path/to/train/neu/images> \
    --out_image_dir <path/to/train/neu/images> \
    --out_mask_dir  <path/to/train/neu/masks> \
    --seed 42
"""

import argparse
import random
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# right-defect files → clean LEFT half available
RIGHT_DEFECT = {
    'crazing':         ['neu_train_crazing_crazing_146.png','neu_train_crazing_crazing_164.png',
                        'neu_train_crazing_crazing_210.png','neu_train_crazing_crazing_46.png'],
    'inclusion':       ['neu_train_inclusion_inclusion_11.png','neu_train_inclusion_inclusion_123.png',
                        'neu_train_inclusion_inclusion_131.png','neu_train_inclusion_inclusion_134.png',
                        'neu_train_inclusion_inclusion_135.png','neu_train_inclusion_inclusion_148.png',
                        'neu_train_inclusion_inclusion_168.png','neu_train_inclusion_inclusion_174.png',
                        'neu_train_inclusion_inclusion_185.png','neu_train_inclusion_inclusion_192.png',
                        'neu_train_inclusion_inclusion_195.png','neu_train_inclusion_inclusion_203.png',
                        'neu_train_inclusion_inclusion_220.png','neu_train_inclusion_inclusion_225.png',
                        'neu_train_inclusion_inclusion_226.png','neu_train_inclusion_inclusion_229.png',
                        'neu_train_inclusion_inclusion_236.png','neu_train_inclusion_inclusion_3.png',
                        'neu_train_inclusion_inclusion_51.png','neu_train_inclusion_inclusion_53.png',
                        'neu_train_inclusion_inclusion_56.png','neu_train_inclusion_inclusion_90.png'],
    'patches':         ['neu_train_patches_patches_129.png','neu_train_patches_patches_173.png',
                        'neu_train_patches_patches_41.png'],
    'rolled-in_scale': ['neu_train_rolled-in_scale_rolled-in_scale_15.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_159.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_161.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_17.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_179.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_196.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_207.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_223.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_234.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_46.png'],
    'scratches':       ['neu_train_scratches_scratches_1.png','neu_train_scratches_scratches_11.png',
                        'neu_train_scratches_scratches_112.png','neu_train_scratches_scratches_134.png',
                        'neu_train_scratches_scratches_16.png','neu_train_scratches_scratches_173.png',
                        'neu_train_scratches_scratches_179.png','neu_train_scratches_scratches_182.png',
                        'neu_train_scratches_scratches_187.png','neu_train_scratches_scratches_191.png',
                        'neu_train_scratches_scratches_195.png','neu_train_scratches_scratches_20.png',
                        'neu_train_scratches_scratches_202.png','neu_train_scratches_scratches_203.png',
                        'neu_train_scratches_scratches_204.png','neu_train_scratches_scratches_207.png',
                        'neu_train_scratches_scratches_22.png','neu_train_scratches_scratches_226.png',
                        'neu_train_scratches_scratches_227.png','neu_train_scratches_scratches_24.png',
                        'neu_train_scratches_scratches_26.png','neu_train_scratches_scratches_39.png',
                        'neu_train_scratches_scratches_41.png','neu_train_scratches_scratches_54.png',
                        'neu_train_scratches_scratches_67.png'],
}

# left-defect files → clean RIGHT half available
LEFT_DEFECT = {
    'crazing':         ['neu_train_crazing_crazing_17.png','neu_train_crazing_crazing_80.png'],
    'inclusion':       ['neu_train_inclusion_inclusion_103.png','neu_train_inclusion_inclusion_144.png',
                        'neu_train_inclusion_inclusion_150.png','neu_train_inclusion_inclusion_173.png',
                        'neu_train_inclusion_inclusion_182.png','neu_train_inclusion_inclusion_19.png',
                        'neu_train_inclusion_inclusion_191.png','neu_train_inclusion_inclusion_193.png',
                        'neu_train_inclusion_inclusion_200.png','neu_train_inclusion_inclusion_237.png',
                        'neu_train_inclusion_inclusion_49.png','neu_train_inclusion_inclusion_55.png',
                        'neu_train_inclusion_inclusion_63.png','neu_train_inclusion_inclusion_83.png',
                        'neu_train_inclusion_inclusion_99.png','neu_train_inclusion_inclusion_173.png',
                        'neu_train_inclusion_inclusion_182.png'],
    'patches':         ['neu_train_patches_patches_159.png','neu_train_patches_patches_164.png'],
    'rolled-in_scale': ['neu_train_rolled-in_scale_rolled-in_scale_111.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_121.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_151.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_18.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_2.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_22.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_222.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_35.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_38.png',
                        'neu_train_rolled-in_scale_rolled-in_scale_41.png'],
    'scratches':       ['neu_train_scratches_scratches_127.png','neu_train_scratches_scratches_148.png',
                        'neu_train_scratches_scratches_150.png','neu_train_scratches_scratches_159.png',
                        'neu_train_scratches_scratches_17.png','neu_train_scratches_scratches_175.png',
                        'neu_train_scratches_scratches_176.png','neu_train_scratches_scratches_180.png',
                        'neu_train_scratches_scratches_184.png','neu_train_scratches_scratches_189.png',
                        'neu_train_scratches_scratches_19.png','neu_train_scratches_scratches_190.png',
                        'neu_train_scratches_scratches_192.png','neu_train_scratches_scratches_193.png',
                        'neu_train_scratches_scratches_200.png','neu_train_scratches_scratches_206.png',
                        'neu_train_scratches_scratches_208.png','neu_train_scratches_scratches_3.png',
                        'neu_train_scratches_scratches_31.png','neu_train_scratches_scratches_35.png',
                        'neu_train_scratches_scratches_40.png','neu_train_scratches_scratches_51.png'],
}

# Per-class caps: (left_n, right_n)
CLASS_CAPS = {
    'crazing':         (2,  2),
    'inclusion':       (17, 17),
    'patches':         (2,  2),
    'rolled-in_scale': (10, 10),
    'scratches':       (16, 16),
}


def load_half(image_dir, filename, side):
    img = np.array(Image.open(image_dir / filename).convert('L'))
    mid = img.shape[1] // 2
    return img[:, :mid] if side == 'left' else img[:, mid:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',     required=True, help='Source NEU train images')
    parser.add_argument('--out_image_dir', required=True, help='Output images folder')
    parser.add_argument('--out_mask_dir',  required=True, help='Output masks folder')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    image_dir = Path(args.image_dir)
    out_img   = Path(args.out_image_dir)
    out_msk   = Path(args.out_mask_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    total_count = 0

    for cls, (left_n, right_n) in CLASS_CAPS.items():
        if cls not in LEFT_DEFECT or cls not in RIGHT_DEFECT:
            print(f"  SKIP {cls}: missing in one pool")
            continue

        # L_clean = left half from right-defect images
        l_pool = list(dict.fromkeys(RIGHT_DEFECT[cls]))
        l_pick = random.sample(l_pool, min(left_n, len(l_pool)))

        # R_clean = right half from left-defect images
        r_pool = list(dict.fromkeys(LEFT_DEFECT[cls]))
        r_pick = random.sample(r_pool, min(right_n, len(r_pool)))

        L_halves = [(f, load_half(image_dir, f, 'left'))  for f in l_pick]
        R_halves = [(f, load_half(image_dir, f, 'right')) for f in r_pick]

        cls_count = 0
        for i, (_, l_half) in enumerate(tqdm(L_halves, desc=f"{cls}")):
            for j, (_, r_half) in enumerate(R_halves):
                synthetic  = np.concatenate([l_half, r_half], axis=1)
                empty_mask = np.zeros(synthetic.shape, dtype=np.uint8)
                out_name   = f"neu_synthetic_{cls}_L{i:02d}_R{j:02d}.png"
                Image.fromarray(synthetic).save(out_img / out_name)
                Image.fromarray(empty_mask).save(out_msk / out_name)
                cls_count += 1

        print(f"  {cls:20s} → {cls_count} synthetics")
        total_count += cls_count

    neu_total = 1259 + total_count
    pct = total_count / neu_total * 100
    print(f"\nTotal synthetics : {total_count}")
    print(f"NEU train total  : {neu_total}")
    print(f"Negative ratio   : {pct:.1f}%")


if __name__ == '__main__':
    main()