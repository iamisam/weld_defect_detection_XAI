import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

MODEL_PATH = "models/best_optimized_resnet18.pth"
DATA_DIR = os.path.join("RIAWELC_Dataset", "testing") 
OUTPUT_DIR = "results_optimized"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_specificity(cm):
    # Specificity = TN / (TN + FP)
    specificities = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        if tn + fp == 0:
            spec = 0
        else:
            spec = tn / (tn + fp)
        specificities.append(spec)
    return specificities

def evaluate():
    print(f"--- STARTING EVALUATION ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Data:  {DATA_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        class_names = dataset.classes
        print(f"Classes Detected: {class_names}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Load Model
    print("Loading model architecture...")
    model = models.resnet18(weights=None)
    
    # Rebuild the final layer to match your training (4 classes)
    num_ftrs = model.fc.in_features
    
    try:
        # Try Optimized Architecture first (Linear + Dropout)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, len(class_names))
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded (Optimized Architecture detected)")
    except:
        # Fallback to Normal Architecture (Just Linear)
        print("Optimized load failed, trying Normal architecture...")
        model.fc = nn.Linear(num_ftrs, len(class_names))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded (Normal Architecture detected)")

    model.to(DEVICE)
    model.eval()

    # 3. Run Inference
    y_true = []
    y_pred = []
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 4. Generate Metrics
    print("Calculating metrics...")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-Class Metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)))
    specificity = calculate_specificity(cm)
    
    # Create a nice DataFrame for the report
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1-Score': f1,
        'Support (Count)': support
    })
    
    # Add Averages row
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOVERALL ACCURACY: {acc*100:.2f}%")

    # Save Metrics to CSV
    csv_path = os.path.join(OUTPUT_DIR, "detailed_test_metrics_pro_gradcam.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Detailed metrics saved to {csv_path}")

    # 5. Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix\nAccuracy: {acc*100:.2f}%')
    
    cm_path = os.path.join(OUTPUT_DIR, "test_confusion_matrix_pro_gradcam.png")
    plt.savefig(cm_path)
    print(f"Confusion Matrix saved to {cm_path}")
    plt.show()

if __name__ == "__main__":
    evaluate()