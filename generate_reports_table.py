import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# CONFIGURATION
BASE_DIR = "RIAWELC_Dataset"
TEST_DIR = os.path.join(BASE_DIR, "testing")
TRAIN_DIR = os.path.join(BASE_DIR, "training")

NORMAL_MODEL = os.path.join("models", "best_weld_resnet18.pth")
OPTIMIZED_MODEL = os.path.join("models", "best_optimized_resnet18.pth")

SAVE_DIR = "final_report_tables"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, num_classes, is_optimized=False):
    """Loads the correct architecture based on model type"""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    
    if is_optimized:
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    # Load weights safely
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    else:
        print(f"Warning: Model not found at {path}")
        return None

def generate_stats(model, loader, class_names, model_name):
    print(f"Evaluating {model_name}...")
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 1. Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. Calculate Standard Metrics (Precision/Recall)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)

    stats_data = []
    
    for i, class_label in enumerate(class_names):
        expected = np.sum(cm[i, :]) 
        output_count = np.sum(cm[:, i]) 
        
        # Diagonal = Correct
        correct = cm[i, i]
        wrong = expected - correct
        
        # Percentage (Sensitivity/Recall for this class)
        # Avoid division by zero
        perc = (correct / expected * 100) if expected > 0 else 0.0

        stats_data.append({
            "Class": class_label,
            "Expected (Count)": expected,
            "Output (Predicted)": output_count,
            "Correct": correct,
            "Wrong": wrong,
            "Accuracy (%)": round(perc, 2),
            "Precision": round(precision[i], 4),
            "Recall": round(recall[i], 4),
            "F1-Score": round(f1[i], 4)
        })

    # Create DataFrame
    df = pd.DataFrame(stats_data)
    
    # Add Averages Row
    avg_row = {
        "Class": "AVERAGE",
        "Expected (Count)": df["Expected (Count)"].sum(),
        "Output (Predicted)": df["Output (Predicted)"].sum(),
        "Correct": df["Correct"].sum(),
        "Wrong": df["Wrong"].sum(),
        "Accuracy (%)": round(df["Correct"].sum() / df["Expected (Count)"].sum() * 100, 2),
        "Precision": round(df["Precision"].mean(), 4),
        "Recall": round(df["Recall"].mean(), 4),
        "F1-Score": round(df["F1-Score"].mean(), 4)
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    return df

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Setup Data
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        temp_dataset = datasets.ImageFolder(TRAIN_DIR)
        class_names = temp_dataset.classes
        
        # Load Testing Data
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=data_transforms)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Evaluate Normal Model
    model_normal = load_model(NORMAL_MODEL, len(class_names), is_optimized=False)
    if model_normal:
        df_normal = generate_stats(model_normal, test_loader, class_names, "Baseline Model")
        print("\n--- BASELINE MODEL RESULTS ---")
        print(df_normal.to_string(index=False))
        df_normal.to_csv(os.path.join(SAVE_DIR, "baseline_model_table.csv"), index=False)

    # 3. Evaluate Optimized Model
    model_opt = load_model(OPTIMIZED_MODEL, len(class_names), is_optimized=True)
    if model_opt:
        df_opt = generate_stats(model_opt, test_loader, class_names, "Optimized Model")
        print("\n--- OPTIMIZED MODEL RESULTS ---")
        print(df_opt.to_string(index=False))
        df_opt.to_csv(os.path.join(SAVE_DIR, "optimized_model_table.csv"), index=False)
        
    print(f"\nTables saved to folder: {SAVE_DIR}/")

if __name__ == "__main__":
    main()