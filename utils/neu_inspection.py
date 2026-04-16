import os
import xml.etree.ElementTree as ET
import cv2

# ---------------- CONFIG ----------------
BASE_DIR = r"D:\weld_defect_project\datasets\NEU_Surface_Defect_Dataset\NEU-DET"
# ----------------------------------------


def inspect_split(split_path):
    print(f"\n📂 Inspecting: {split_path}")
    print("-" * 60)

    img_dir = os.path.join(split_path, "images")
    ann_dir = os.path.join(split_path, "annotations")

    classes = os.listdir(img_dir)

    print("Classes found:", classes)

    for cls in classes:
        cls_path = os.path.join(img_dir, cls)
        images = os.listdir(cls_path)

        print(f"\n🔹 Class: {cls}")
        print(f"Total images: {len(images)}")
        print("Sample images:", images[:3])

        # Check image shape
        for img_name in images[:2]:
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                print(f"{img_name}: shape={img.shape}")
            else:
                print(f"{img_name}: FAILED")

        # Check annotation
        print("\nSample annotations:")
        for img_name in images[:2]:
            xml_name = img_name.replace(".jpg", ".xml")
            xml_path = os.path.join(ann_dir, xml_name)

            if not os.path.exists(xml_path):
                print(f"{xml_name}: NOT FOUND")
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()

            objects = root.findall("object")

            print(f"{xml_name}: {len(objects)} objects")

            for obj in objects:
                name = obj.find("name").text

                bndbox = obj.find("bndbox")
                xmin = bndbox.find("xmin").text
                ymin = bndbox.find("ymin").text
                xmax = bndbox.find("xmax").text
                ymax = bndbox.find("ymax").text

                print(f"  class={name}, box=({xmin},{ymin},{xmax},{ymax})")

    print("-" * 60)


def main():
    train_path = os.path.join(BASE_DIR, "train")
    val_path = os.path.join(BASE_DIR, "validation")

    inspect_split(train_path)
    inspect_split(val_path)


if __name__ == "__main__":
    main()