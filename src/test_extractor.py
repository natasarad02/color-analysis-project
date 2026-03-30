# test_extractor.py
import os
import sys
import cv2
import random
from pathlib import Path

import numpy as np

# Dodavanje src foldera u putanju
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from face_extractor import FaceExtractor
import matplotlib.pyplot as plt

def test_random_images(data_dir="data", num_samples=3):
    """
    Testira nasumične slike iz train/test/val skupova
    
    Args:
        data_dir: Putanja do data foldera
        num_samples: Koliko slika prikazati po skupu
    """
    
    extractor = FaceExtractor()
    
    # Sezone koje imamo
    seasons = ['spring', 'summer', 'autumn', 'winter']
    splits = ['train', 'test', 'val']
    
    # Prolazimo kroz svaki skup (train, test, val)
    for split in splits:
        print(f"\n{'='*50}")
        print(f"TESTIRANJE: {split.upper()} SKUP")
        print(f"{'='*50}")
        
        split_path = os.path.join(data_dir, split)
        
        if not os.path.exists(split_path):
            print(f"Folder ne postoji: {split_path}")
            continue
            
        # Biramo nasumične slike iz ovog skupa
        all_images = []
        for season in seasons:
            season_path = os.path.join(split_path, season)
            if os.path.exists(season_path):
                images = [os.path.join(season_path, f) for f in os.listdir(season_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                all_images.extend([(img, season) for img in images])
        
        if len(all_images) == 0:
            print(f"Nema slika u {split} skupu!")
            continue
            
        # Biramo nasumične uzorke
        samples = random.sample(all_images, min(num_samples, len(all_images)))
        
        for img_path, season in samples:
            print(f"\nObrada: {os.path.basename(img_path)} ({season})")
            
            try:
                # Ekstrakcija
                result = extractor.extract_features(img_path, visualize=False)
                
                if result is None:
                    print(f"   Lice nije detektovano ili nema dovoljno piksela")
                    continue
                
                # Statistika
                print(f"   Uspešno!")
                print(f"     - Koža: {len(result['skin_pixels'])} piksela")
                print(f"     - Kosa: {len(result['hair_pixels'])} piksela")
                print(f"     - Levo oko: {result['left_eye'].shape}")
                print(f"     - Desno oko: {result['right_eye'].shape}")
                
                # Vizualizacija
                visualize_result(result, split, season)
                
            except Exception as e:
                print(f"   Greška: {str(e)}")
    
    extractor.close()
    print(f"\n{'='*50}")
    print("TESTIRANJE ZAVRŠENO!")
    print(f"{'='*50}")

def visualize_result(features, split, season):
    """
    Prikazuje rezultate ekstrakcije
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{split.upper()} - {season.upper()}: {os.path.basename(features["image_path"])}', 
                 fontsize=14, fontweight='bold')
    
    image = features['original_image']
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Maska kože
    axes[0, 1].imshow(features['skin_mask'], cmap='gray')
    axes[0, 1].set_title('Maska kože')
    axes[0, 1].axis('off')
    
    # Maska kose
    axes[0, 2].imshow(features['hair_mask'], cmap='gray')
    axes[0, 2].set_title('Maska kose')
    axes[0, 2].axis('off')
    
    # Izdvojena koža
    skin_only = cv2.bitwise_and(image, image, mask=features['skin_mask'])
    axes[1, 0].imshow(cv2.cvtColor(skin_only, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Izdvojena koža')
    axes[1, 0].axis('off')
    
    # Izdvojena kosa
    hair_only = cv2.bitwise_and(image, image, mask=features['hair_mask'])
    axes[1, 1].imshow(cv2.cvtColor(hair_only, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Izdvojena kosa')
    axes[1, 1].axis('off')
    
    # Oči
    if features['left_eye'].size > 0 and features['right_eye'].size > 0:
        # Resize da budu iste visine
        h = max(features['left_eye'].shape[0], features['right_eye'].shape[0])
        left_resized = cv2.resize(features['left_eye'], 
                                 (int(features['left_eye'].shape[1] * h / features['left_eye'].shape[0]), h))
        right_resized = cv2.resize(features['right_eye'], 
                                  (int(features['right_eye'].shape[1] * h / features['right_eye'].shape[0]), h))
        combined_eyes = np.hstack([left_resized, right_resized])
        axes[1, 2].imshow(cv2.cvtColor(combined_eyes, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Oči')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def test_specific_image(image_path):
    """
    Testira jednu specifičnu sliku sa detaljnim prikazom
    
    Args:
        image_path: Putanja do slike
    """
    if not os.path.exists(image_path):
        print(f"Slika ne postoji: {image_path}")
        return
        
    extractor = FaceExtractor()
    
    print(f"Obrada slike: {image_path}")
    result = extractor.extract_features(image_path, visualize=False)
    
    if result:
        # Detaljan prikaz
        fig = plt.figure(figsize=(20, 12))
        
        # Original sa landmarkovima
        img_with_landmarks = result['original_image'].copy()
        h, w = img_with_landmarks.shape[:2]
        
        # Crtanje landmarkova
        for (x, y, z) in result['landmarks']:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(img_with_landmarks, (cx, cy), 1, (0, 255, 0), -1)
        
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(img_with_landmarks, cv2.COLOR_BGR2RGB))
        plt.title('Original sa landmarkovima (468 tačaka)')
        plt.axis('off')
        
        # Maska kože
        plt.subplot(2, 4, 2)
        plt.imshow(result['skin_mask'], cmap='gray')
        plt.title('Maska kože')
        plt.axis('off')
        
        # Maska kose
        plt.subplot(2, 4, 3)
        plt.imshow(result['hair_mask'], cmap='gray')
        plt.title('Maska kose')
        plt.axis('off')
        
        # Koža
        skin_img = cv2.bitwise_and(result['original_image'], result['original_image'], 
                                   mask=result['skin_mask'])
        plt.subplot(2, 4, 5)
        plt.imshow(cv2.cvtColor(skin_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Koža ({len(result["skin_pixels"])} piksela)')
        plt.axis('off')
        
        # Kosa
        hair_img = cv2.bitwise_and(result['original_image'], result['original_image'], 
                                   mask=result['hair_mask'])
        plt.subplot(2, 4, 6)
        plt.imshow(cv2.cvtColor(hair_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Kosa ({len(result["hair_pixels"])} piksela)')
        plt.axis('off')
        
        # Levo oko
        plt.subplot(2, 4, 7)
        plt.imshow(cv2.cvtColor(result['left_eye'], cv2.COLOR_BGR2RGB))
        plt.title('Levo oko')
        plt.axis('off')
        
        # Desno oko
        plt.subplot(2, 4, 8)
        plt.imshow(cv2.cvtColor(result['right_eye'], cv2.COLOR_BGR2RGB))
        plt.title('Desno oko')
        plt.axis('off')
        
        # Histogram boja kože
        plt.subplot(2, 4, 4)
        skin_rgb = cv2.cvtColor(result['skin_pixels'].reshape(-1, 1, 3), cv2.COLOR_BGR2RGB)
        colors = ('r', 'g', 'b')
        for i, col in enumerate(colors):
            histr = cv2.calcHist([skin_rgb], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.title('Histogram boja kože')
        plt.xlabel('Intenzitet')
        plt.ylabel('Frekvencija')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nDetalji:")
        print(f"  Putanja: {result['image_path']}")
        print(f"  Dimenzije: {result['original_image'].shape}")
        print(f"  Broj landmarkova: {len(result['landmarks'])}")
        print(f"  Koža - min/max B: {result['skin_pixels'][:,0].min()}/{result['skin_pixels'][:,0].max()}")
        print(f"  Koža - min/max G: {result['skin_pixels'][:,1].min()}/{result['skin_pixels'][:,1].max()}")
        print(f"  Koža - min/max R: {result['skin_pixels'][:,2].min()}/{result['skin_pixels'][:,2].max()}")
    else:
        print("Nije uspelo izdvajanje!")
    
    extractor.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Testiranje FaceExtractor-a')
    parser.add_argument('--mode', choices=['random', 'single'], default='random',
                       help='random: testira nasumične slike iz svih skupova, single: testira jednu sliku')
    parser.add_argument('--image', type=str, help='Putanja do slike (za single mode)')
    parser.add_argument('--data', type=str, default='data', help='Putanja do data foldera')
    parser.add_argument('--samples', type=int, default=2, help='Broj uzoraka po skupu (za random mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'random':
        test_random_images(args.data, args.samples)
    else:
        if not args.image:
            print("Greška: Za single mode moraš specificirati --image")
        else:
            test_specific_image(args.image)