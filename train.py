import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import multiprocessing

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "data"  # Ovo ostaje isto za početak
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LR = 3e-4
NUM_CLASSES = 4
DEVICE = "cpu"

# -------------------------
# DATA TRANSFORMS
# -------------------------
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    print("Using device:", DEVICE)
    
    # DATASETS
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tfms)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_tfms)
    
    # VAŽNO: num_workers=0 za Windows
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("Classes:", train_ds.classes)
    
    # MODEL
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_val_acc = 0
    best_path = "best_model.pth"
    
    # TRAINING LOOP
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0
    
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
    
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item() * x.size(0)
            preds = outputs.argmax(1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
    
        train_acc = train_correct / train_total
        train_loss /= train_total
    
        # VALIDATION
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
    
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                loss = criterion(outputs, y)
    
                val_loss += loss.item() * x.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
    
        val_acc = val_correct / val_total
        val_loss /= val_total
    
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print("Saved new best model")
    
    # TEST EVALUATION
    print("\n=== Testing Best Model ===")
    
    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            outputs = model(x)
            preds = outputs.argmax(1).cpu().numpy()
    
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=train_ds.classes))