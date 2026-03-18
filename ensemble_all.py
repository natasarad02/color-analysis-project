import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix

# ====================
# ENSEMBLE KLASA
# ====================
class EnsembleModel:
    def __init__(self, model_paths, device="cpu"):
        self.device = device
        self.models = []
        self.model_paths = model_paths  # Sačuvaj za prikaz
        
        for path in model_paths:
            print(f"Učitavanje modela: {path}")
            
            model = models.mobilenet_v3_large()
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 4)
            
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            model.eval()
            self.models.append(model)
    
    def predict(self, x):
        predictions = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(x.to(self.device))
                probs = torch.softmax(outputs, dim=1)
                predictions.append(probs)
        
        avg_pred = torch.mean(torch.stack(predictions), dim=0)
        return avg_pred
    
    def predict_class(self, x):
        probs = self.predict(x)
        return probs.argmax(dim=1)

# ====================
# DATASET
# ====================
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ====================
# GLAVNI DEO
# ====================
if __name__ == "__main__":
    DEVICE = "cpu"
    BATCH_SIZE = 16
    IMG_SIZE = 224
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    
    # Učitaj test slike
    print("Učitavanje test skupa...")
    test_ds = TestDataset("data_face/test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Testiranje na {len(test_ds)} slika")
    print("Klase:", test_ds.classes)
    
    # ====================
    # TESTIRAJ RAZLIČITE KOMBINACIJE
    # ====================
    
    # Opcija 1: Svi modeli
    print("\n" + "="*50)
    print("🎯 ENSEMBLE - SVI MODELI")
    print("="*50)
    
    ensemble_all = EnsembleModel([
        "best_model.pth",
        "best_model_skin.pth",
        "best_model_hair.pth",
        "best_model_left_eye.pth",
        "best_model_right_eye.pth",
    ], device=DEVICE)
    
    all_preds = []
    all_labels = []
    
    for images, labels in test_loader:
        preds = ensemble_all.predict_class(images)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    print("\nMatrica konfuzije:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nIzveštaj:")
    print(classification_report(all_labels, all_preds, target_names=test_ds.classes))
    
    # Opcija 2: Bez očiju (ako su loše)
    print("\n" + "="*50)
    print("🎯 ENSEMBLE - BEZ OČIJU")
    print("="*50)
    
    ensemble_no_eyes = EnsembleModel([
        "best_model.pth",
        "best_model_skin.pth",
        "best_model_hair.pth"
    ], device=DEVICE)
    
    all_preds = []
    
    for images, labels in test_loader:
        preds = ensemble_no_eyes.predict_class(images)
        all_preds.extend(preds.cpu().numpy())
    
    print("\nMatrica konfuzije:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nIzveštaj:")
    print(classification_report(all_labels, all_preds, target_names=test_ds.classes))