import os
import cv2

# ---------------- CONFIG ----------------
BASE_DIR = r"D:\weld_defect_project\datasets\DAGM_Dataset"
# ----------------------------------------


def inspect_class(class_path):
    print(f"\n📂 Inspecting: {class_path}")
    print("-" * 60)

    train_path = os.path.join(class_path, "Train")
    test_path = os.path.join(class_path, "Test")

    for split_name, split_path in [("Train", train_path), ("Test", test_path)]:
        if not os.path.exists(split_path):
            continue

        print(f"\n🔹 {split_name}:")

        files = os.listdir(split_path)
        images = [f for f in files if f.lower().endswith(".png")]

        label_path = os.path.join(split_path, "Label")
        has_label_dir = os.path.exists(label_path)

        print(f"Total images: {len(images)}")
        print(f"Has Label folder: {has_label_dir}")

        if has_label_dir:
            label_files = os.listdir(label_path)
            print(f"Total masks: {len(label_files)}")
            print("Sample masks:", label_files[:5])

        print("Sample images:", images[:5])

        # Check image shape
        print("\nSample image shapes:")
        for img_name in images[:3]:
            img_path = os.path.join(split_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                print(f"{img_name}: {img.shape}")
            else:
                print(f"{img_name}: FAILED TO LOAD")

        # Check mask shape (if exists)
        if has_label_dir:
            print("\nSample mask shapes:")
            for mask_name in label_files[:3]:
                mask_path = os.path.join(label_path, mask_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if mask is not None:
                    print(f"{mask_name}: {mask.shape}")
                else:
                    print(f"{mask_name}: FAILED TO LOAD")

        print("-" * 60)


def main():
    class_dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

    print("📊 Found classes:", class_dirs)

    for cls in class_dirs:
        class_path = os.path.join(BASE_DIR, cls)
        inspect_class(class_path)


if __name__ == "__main__":
    main()