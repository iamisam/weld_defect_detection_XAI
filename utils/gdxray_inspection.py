import os
import cv2

# ---------------- CONFIG ----------------
BASE_DIR = r"D:\weld_defect_project\datasets\GDXray_Dataset\Welds"
# ----------------------------------------

def inspect_folder(folder_path):
    print(f"\n📂 Inspecting: {folder_path}")
    print("-" * 50)

    files = os.listdir(folder_path)

    images = [f for f in files if f.lower().endswith((".png", ".jpg", ".tif"))]
    txts = [f for f in files if f.lower().endswith(".txt")]

    print(f"Total files: {len(files)}")
    print(f"Images: {len(images)}")
    print(f"Text files: {len(txts)}")

    # Show sample files
    print("\nSample image files:")
    print(images[:5])

    print("\nSample txt files:")
    print(txts[:5])

    # Check image shapes
    print("\nImage shape samples:")
    for img_name in images[:3]:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            print(f"{img_name}: {img.shape}")
        else:
            print(f"{img_name}: FAILED TO LOAD")

    print("-" * 50)


def main():
    weld_sets = ["W0001", "W0002", "W0003", "W0004"]

    for w in weld_sets:
        path = os.path.join(BASE_DIR, w)

        if os.path.exists(path):
            inspect_folder(path)
        else:
            print(f"Missing folder: {path}")


if __name__ == "__main__":
    main()