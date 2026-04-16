import cv2
import numpy as np
import os
from glob import glob

import cv2
import numpy as np

def remove_annotations_tophat(image):
    # 1. Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # 2. Slight smoothing
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 3. Stronger top-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (65, 65))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    # 4. Normalize response (critical)
    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)

    # 5. More aggressive threshold
    _, mask = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)

    # 6. Expand mask
    cleanup_kernel = np.ones((7, 7), np.uint8)
    mask_clean = cv2.dilate(mask, cleanup_kernel, iterations=2)
    mask_clean = cv2.GaussianBlur(mask_clean, (5, 5), 0)

    # 7. Inpaint
    cleaned = cv2.inpaint(image, mask_clean, 10, cv2.INPAINT_TELEA)

    return cleaned, mask_clean

def process_batch(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_paths = glob(os.path.join(input_folder, "*.jpg"))

    print(f"Found {len(image_paths)} images")

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)

        # Load image (grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Skipping {img_name} (failed to load)")
            continue

        # Process
        cleaned, mask = remove_annotations_tophat(img)

        # Save outputs
        base_name = os.path.splitext(img_name)[0]

        cleaned_path = os.path.join(output_folder, f"{base_name}_clean.png")
        mask_path = os.path.join(output_folder, f"{base_name}_mask.png")

        cv2.imwrite(cleaned_path, cleaned)
        cv2.imwrite(mask_path, mask)

        print(f"[{idx+1}] Processed {img_name}")

    print("Batch processing complete.")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    input_folder = os.path.join(BASE_DIR, "..", "images_new_set")
    output_folder = os.path.join(BASE_DIR, "..", "annotation_removal_results")

    process_batch(input_folder, output_folder)