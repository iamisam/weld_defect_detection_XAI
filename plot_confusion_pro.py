
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np

# --- CONFIG ---
MODEL_PATH = os.path.join("models", "best_optimized_resnet18.pth")
BASE_DIR = "RIAWELC_Dataset"
VAL_DIR = os.path.join(BASE_DIR, "validation")
TRAIN_DIR = os.path.join(BASE_DIR, "training")
OUTPUT_IMAGE = os.path.join("results_optimized", "confusion_matrix.png")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_matrix():
    print(f"Loading model from {MODEL_PATH}...")
    
    # get class names
    try:
        temp_dataset = datasets.ImageFolder(TRAIN_DIR)
        class_names = temp_dataset.classes
        print(f"Classes: {class_names}")
    except Exception as e:
        print(f"Error finding classes: {e}")
        return

    # load the model structure
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_ftrs, len(class_names))
    )
    
    # load weights
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # prepare validation data
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Running inference on validation set...")
    
    # Collect predictions
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # plot and save
    print("Generating plot...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a nice plot
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Plot with a blue colormap
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    
    plt.title("Confusion Matrix (Best Optimized Model)")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Confusion Matrix saved to {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    generate_matrix()