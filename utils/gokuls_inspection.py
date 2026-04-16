import os
import cv2

# ---------------- CONFIG ----------------
BASE_DIR = r"D:\weld_defect_project\datasets\gokul_weld_defects"
# ----------------------------------------


def inspect_split(split):
    print(f"\n📂 Inspecting: {split}")
    print("-" * 60)

    img_dir = os.path.join(BASE_DIR, "images", split)
    label_dir = os.path.join(BASE_DIR, "labels", split)

    images = os.listdir(img_dir)

    print(f"Total images: {len(images)}")
    print("Sample images:", images[:5])

    # Check image shapes
    print("\nSample image shapes:")
    for img_name in images[:3]:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            print(f"{img_name}: {img.shape}")
        else:
            print(f"{img_name}: FAILED")

    # Check labels
    print("\nSample labels:")
    for img_name in images[:5]:
        base = img_name.split(".")[0]
        label_path = os.path.join(label_dir, base + ".txt")

        if not os.path.exists(label_path):
            print(f"{base}.txt: NOT FOUND")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        print(f"{base}.txt: {len(lines)} objects")

        for line in lines[:3]:
            print(" ", line.strip())

    print("-" * 60)


def main():
    inspect_split("train")
    inspect_split("val")


if __name__ == "__main__":
    main()