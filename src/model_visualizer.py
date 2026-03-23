import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.manifold import TSNE
import seaborn as sns
from PIL import Image

class ModelVisualizer:
    """
    Klasa za vizuelizaciju šta je CNN model naučio
    """
    
    def __init__(self, model, device='cpu', class_names=None):
        """
        Args:
            model: trenirani CNN model
            device: 'cpu' ili 'cuda'
            class_names: liste naziva klasa ['spring', 'summer', 'autumn', 'winter']
        """
        self.model = model
        self.device = device
        self.class_names = class_names if class_names else ['Spring', 'Summer', 'Autumn', 'Winter']
        self.model.eval()
        
        
    def visualize_activation_maps(self, image_tensor, target_class=None):
        """
        Vizuelizacija aktivacijskih mapa (CAM/Grad-CAM)
        Pojednostavljena verzija - prikazuje aktivacije poslednjeg konvolucionog sloja
        
        Args:
            image_tensor: ulazna slika [C, H, W] ili [1, C, H, W]
            target_class: klasa za koju se računa aktivacija (None = max)
        """
        self.model.eval()
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Pronađi poslednji konvolucioni sloj
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        
        # Hvatanje feature mapa
        feature_maps = []
        def hook_fn(module, input, output):
            feature_maps.append(output.detach())
        
        if last_conv:
            handle = last_conv.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
        if last_conv:
            handle.remove()
        
        # Aktivacije
        if feature_maps:
            activations = feature_maps[0][0]  # [C, H, W]
            
            # Sumiraj aktivacije po kanalima
            activation_map = activations.mean(dim=0).cpu().numpy()
            
            # Normalizacija
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
            
            # Overlay
            axes[2].imshow(img)
            axes[2].imshow(activation_map, cmap='hot', alpha=0.5)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.suptitle(f'Activations - Predicted: {self.class_names[outputs.argmax(1).item()]}')
            plt.tight_layout()
            plt.show()
        else:
            print("Nije pronađen konvolucioni sloj")
    
    def visualize_tsne(self, dataloader, max_samples=500):
        """
        t-SNE vizuelizacija feature vektora (embeddinga) iz modela
        
        Args:
            dataloader: DataLoader sa podacima
            max_samples: maksimalan broj uzoraka za vizuelizaciju
        """
        self.model.eval()
        
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                
                # Ekstrakcija embeddinga (pre classifier sloja)
                # Za MobileNet, uzimamo globalni pooling sloj
                features = self.model.features(x)
                features = self.model.avgpool(features)
                features = torch.flatten(features, 1)
                
                embeddings.append(features.cpu())
                labels.append(y)
                
                if len(embeddings) * x.size(0) >= max_samples:
                    break
        
        embeddings = torch.cat(embeddings, dim=0)[:max_samples].numpy()
        labels = torch.cat(labels, dim=0)[:max_samples].numpy()
        
        # t-SNE redukcija dimenzija
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Vizuelizacija
        plt.figure(figsize=(12, 10))
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       label=class_name, alpha=0.7, s=50)
        
        plt.title('t-SNE vizuelizacija embeddinga slika', fontsize=14)
        plt.xlabel('t-SNE dimenzija 1')
        plt.ylabel('t-SNE dimenzija 2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        
    def visualize_confusion_matrix(self, test_loader):
        """
        Vizuelizacija konfuzione matrice
        """
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
        """
        Prikaz pogrešno klasifikovanih primera
        
        Args:
            test_loader: DataLoader sa test podacima
            num_examples: broj primera za prikaz
        """
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