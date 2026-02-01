import torch
import cv2
import numpy as np
import os
import glob
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# --- CONFIG ---
MODEL_PATH = "models/best_weld_resnet18.pth"
TEST_DIR = os.path.join("RIAWELC_Dataset", "testing")
OUTPUT_DIR = os.path.join("results", "gradcam_outputs")

CLASS_NAMES = ['Crack', 'Lack of Penetration', 'No Defect', 'Porosity']

def process_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # 1. Load Model
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)

    # 2. Setup GradCAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # 3. Setup Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Scanning images in {TEST_DIR}...")

    # Walk through directories
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                
                img_path = os.path.join(root, file)
                
                class_folder = os.path.basename(root)
                
                save_folder = os.path.join(OUTPUT_DIR, class_folder)
                os.makedirs(save_folder, exist_ok=True)

                try:
                    # Process Image
                    raw_img = Image.open(img_path).convert('RGB')
                    raw_img = raw_img.resize((224, 224))
                    
                    input_tensor = transform(raw_img).unsqueeze(0).to(device)
                    rgb_img = np.float32(raw_img) / 255

                    # Inference
                    output = model(input_tensor)
                    _, pred_idx = torch.max(output, 1)
                    predicted_label = CLASS_NAMES[pred_idx.item()]

                    # Generate Heatmap
                    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
                    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    # Save Result: Filename includes prediction (e.g., image_Pred_Crack.jpg)
                    save_name = f"{os.path.splitext(file)[0]}_Pred_{predicted_label}.jpg"
                    save_path = os.path.join(save_folder, save_name)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                    print(f"Processed: {file} -> Pred: {predicted_label}")

                except Exception as e:
                    print(f"Failed to process {file}: {e}")

    print(f"\nAll Done! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    process_batch()