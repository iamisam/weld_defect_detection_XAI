import torch
import cv2
import numpy as np
import os
from torchvision import models, transforms, datasets
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# --- CONFIG ---
MODEL_PATH = os.path.join("models", "best_optimized_resnet18.pth")
BASE_DIR = "RIAWELC_Dataset"
TEST_DIR = os.path.join(BASE_DIR, "testing")
TRAIN_DIR = os.path.join(BASE_DIR, "training") # Used to detect class names
OUTPUT_DIR = os.path.join("results_optimized", "gradcam_outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_batch():
    try:
        temp_dataset = datasets.ImageFolder(TRAIN_DIR)
        class_names = temp_dataset.classes
        print(f"Detected {len(class_names)} classes: {class_names}")
    except Exception as e:
        print(f"Error reading classes from {TRAIN_DIR}: {e}")
        return

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = models.resnet18(weights=None)
    
    # reconstruct the EXACT architecture used in training
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_ftrs, len(class_names))
    )
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("Error: Model file not found. Did you run train_pro.py?")
        return

    model.eval()
    model.to(DEVICE)

    # Setup GradCAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Scanning images in {TEST_DIR}...")
    count = 0

    # Process Images
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                
                img_path = os.path.join(root, file)
                
                # Create output folder structure (e.g. results/.../Crack/)
                folder_name = os.path.basename(root)
                save_folder = os.path.join(OUTPUT_DIR, folder_name)
                os.makedirs(save_folder, exist_ok=True)

                try:
                    # Prepare Image
                    raw_img = Image.open(img_path).convert('RGB')
                    raw_img = raw_img.resize((224, 224))
                    
                    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
                    rgb_img = np.float32(raw_img) / 255

                    # Prediction
                    output = model(input_tensor)
                    _, pred_idx = torch.max(output, 1)
                    predicted_label = class_names[pred_idx.item()]

                    # Heatmap
                    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
                    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    # Save
                    save_name = f"{os.path.splitext(file)[0]}_Pred_{predicted_label}.jpg"
                    save_path = os.path.join(save_folder, save_name)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                    count += 1
                    if count % 10 == 0:
                        print(f"Processed {count} images...")

                except Exception as e:
                    print(f"Failed to process {file}: {e}")

    print(f"\nAll Done! {count} images processed.")
    print(f"Check the results in: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_batch()