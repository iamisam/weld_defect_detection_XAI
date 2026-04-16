import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy

# --- CONFIGURATION ---
BASE_DIR = "RIAWELC_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "training")
VAL_DIR = os.path.join(BASE_DIR, "validation")
SAVE_DIR = "results"

BATCH_SIZE = 32
NUM_CLASSES = 4       
LEARNING_RATE = 0.0005
NUM_EPOCHS = 10  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print(f"Using device: {DEVICE}")

    # 1. Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Load Data
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=data_transforms['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")

    # 3. Model Setup
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # Logging Setup
    history = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print("Starting training...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            train_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct.double() / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct.double() / val_total

        # Calculate Detailed Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

        # Log History
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} F1: {f1:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc.item(),
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc.item(),
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        })

        # Deep Copy Best Model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join("models", "best_weld_resnet18.pth"))

    print("Training Complete. Saving logs and graphs...")
    
    # Save CSV
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(SAVE_DIR, "training_logs.csv"), index=False)

    # Plot & Save Confusion Matrix (for the best model state)
    model.load_state_dict(best_model_wts)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix (Best Validation Model)")
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
    print(f"Results saved to {SAVE_DIR}/")

if __name__ == "__main__":
    train()