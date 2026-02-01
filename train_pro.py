import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import pandas as pd
import os
import copy
from collections import Counter

# --- CONFIGURATION ---
BASE_DIR = "RIAWELC_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "training")
VAL_DIR = os.path.join(BASE_DIR, "validation")
SAVE_DIR = "results_optimized"

BATCH_SIZE = 32
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_class_weights(dataset):
    count_dict = Counter(dataset.targets)
    weights = []
    for i in range(len(count_dict)):
        weights.append(1.0 / count_dict[i])
    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum() * len(count_dict)
    return weights.to(DEVICE)

def train_optimized():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 1. Transforms (Augmentation)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Load Data & Auto-detect Classes
    try:
        train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
        val_dataset = datasets.ImageFolder(VAL_DIR, transform=data_transforms['val'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if len(train_dataset) == 0:
        print(f"ERROR: No images found in {TRAIN_DIR}")
        return

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} Classes: {class_names}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model Setup
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Unfreeze layer4 and fc
    for name, child in model.named_children():
        if name in ['layer4', 'fc']:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(DEVICE)

    # 4. Training Setup
    class_weights = get_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    history = [] 

    print("Starting Optimized Training...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct.double() / total
        
        model.eval()
        val_running_loss = 0.0 # NEW: Track val loss
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
        
        val_loss = val_running_loss / val_total
        val_acc = val_correct.double() / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc.item(),
            'val_acc': val_acc.item(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join("models", "best_optimized_resnet18.pth"))

    # Save CSV
    df = pd.DataFrame(history)
    log_path = os.path.join(SAVE_DIR, "training_logs_optimized.csv")
    df.to_csv(log_path, index=False)
    print(f"Training Complete. Logs saved to {log_path}")

if __name__ == "__main__":
    train_optimized()