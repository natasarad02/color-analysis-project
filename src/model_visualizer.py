import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.manifold import TSNE
import seaborn as sns
from PIL import Image

class ModelVisualizer:
    #Klasa za vizuelizaciju šta je CNN model naučio
    
    def __init__(self, model, device='cpu', class_names=None):
        self.model = model
        self.device = device
        self.class_names = class_names if class_names else ['Spring', 'Summer', 'Autumn', 'Winter']
        self.model.eval()
        
        
    def visualize_activation_maps(self, image_tensor, target_class=None):
        #Vizuelizacija aktivacijskih mapa (CAM/Grad-CAM)
        #Pojednostavljena verzija - prikazuje aktivacije poslednjeg konvolucionog sloja
        self.model.eval() #prebacuje u evaluacioni mod
        
        if image_tensor.dim() == 3: #ako su tri dimenzije, dodaje jos jednu sa 1
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Pronađi poslednji konvolucioni sloj, pre klasifikacionog
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        
        # Hvatanje feature mapa (aktivacija)
        feature_maps = []
        def hook_fn(module, input, output):
            feature_maps.append(output.detach())
        
        if last_conv:
            handle = last_conv.register_forward_hook(hook_fn) #registruje hook na poslednji konv sloj
        
        with torch.no_grad():
            outputs = self.model(image_tensor) #propusta sliku kroz model
            
        if last_conv:
            handle.remove() #uklanja hook
        
        # Aktivacije
        if feature_maps:
            activations = feature_maps[0][0]  # [C, H, W]
            
            # Sumiraj aktivacije po kanalima
            activation_map = activations.mean(dim=0).cpu().numpy()
            
            # Normalizacija (sve su izmedju 0 i 1)
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
            
            # Originalna slika
            img = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # Prikaz
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img)
            axes[0].set_title('Originalna slika')
            axes[0].axis('off')
            
            axes[1].imshow(activation_map, cmap='hot')
            axes[1].set_title('Aktivaciona mapa')
            axes[1].axis('off')
            
            axes[2].imshow(img)
            axes[2].imshow(activation_map, cmap='hot', alpha=0.5)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.suptitle(f'Activations - Predicted: {self.class_names[outputs.argmax(1).item()]}')
            plt.tight_layout()
            plt.show()
        else:
            print("Nije pronađen konvolucioni sloj")
        
    def visualize_confusion_matrix(self, test_loader):

        #Vizuelizacija konfuzione matrice
        from sklearn.metrics import confusion_matrix
        
        y_true = []
        y_pred = []
        
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                outputs = self.model(x)
                preds = outputs.argmax(1).cpu().numpy()
                
                y_pred.extend(preds)
                y_true.extend(y.numpy())
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def visualize_misclassified(self, test_loader, num_examples=8):
        #Prikaz pogrešno klasifikovanih primera

        self.model.eval()
        
        misclassified = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                outputs = self.model(x)
                preds = outputs.argmax(1)
                
                # Pronađi pogrešne
                wrong_mask = (preds != y.to(self.device))
                if wrong_mask.any():
                    for i in range(len(x)):
                        if wrong_mask[i] and len(misclassified) < num_examples:
                            # Denormalizuj sliku
                            img = x[i].cpu().numpy().transpose(1, 2, 0)
                            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                            img = np.clip(img, 0, 1)
                            
                            misclassified.append({
                                'image': img,
                                'true': y[i].item(),
                                'pred': preds[i].item(),
                                'confidence': F.softmax(outputs[i], dim=0)[preds[i]].item()
                            })
                
                if len(misclassified) >= num_examples:
                    break
        
        if not misclassified:
            print("Nema pogrešno klasifikovanih primera!")
            return
        
        # Prikaz
        n_cols = min(4, num_examples)
        n_rows = (len(misclassified) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols > 1 else [axes]
        
        for i, sample in enumerate(misclassified):
            if i >= len(axes):
                break
            ax = axes[i]
            ax.imshow(sample['image'])
            ax.set_title(f"True: {self.class_names[sample['true']]}\n"
                        f"Pred: {self.class_names[sample['pred']]}\n"
                        f"Conf: {sample['confidence']:.2f}", 
                        color='red' if sample['true'] != sample['pred'] else 'green')
            ax.axis('off')
        
        # Sakrij prazne podgrafikone
        for i in range(len(misclassified), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Pogrešno klasifikovani primeri', fontsize=14)
        plt.tight_layout()
        plt.show()
        
    def visualize_class_averages(self, dataloader):
        #Prikaz prosečnih slika za svaku klasu

        self.model.eval()
        
        # Sakupi sve slike po klasama
        class_images = {i: [] for i in range(len(self.class_names))}
        
        with torch.no_grad():
            for x, y in dataloader:
                for i in range(len(x)):
                    img = x[i].cpu().numpy().transpose(1, 2, 0)
                    # Denormalizacija
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                    class_images[y[i].item()].append(img)
        
        # Izračunaj prosečnu sliku za svaku klasu
        avg_images = {}
        for class_idx, images in class_images.items():
            if images:
                avg_images[class_idx] = np.mean(images, axis=0)
        
        # Prikaz
        fig, axes = plt.subplots(1, len(avg_images), figsize=(5 * len(avg_images), 5))
        if len(avg_images) == 1:
            axes = [axes]
        
        for i, (class_idx, avg_img) in enumerate(avg_images.items()):
            axes[i].imshow(avg_img)
            axes[i].set_title(f'Prosečna slika\n{self.class_names[class_idx]}', fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('Prosečne slike po klasama', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return avg_images