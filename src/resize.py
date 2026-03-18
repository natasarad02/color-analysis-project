import os
import cv2
from pathlib import Path

def process_images():
    """
    Proverava raw folder, obradjuje nove slike i stavlja ih u processed
    Format: 001_ime_slike.jpg, 002_ime_slike.jpg, ...
    """
    seasons = ['spring', 'summer', 'autumn', 'winter']
    target_size = (224, 224)
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    for season in seasons:
        raw_path = Path(f'data/raw/{season}')
        processed_path = Path(f'data/train/{season}')
        
        processed_path.mkdir(parents=True, exist_ok=True)
        
        # Pronadji sve slike u raw
        raw_images = []
        for ext in valid_extensions:
            raw_images.extend(raw_path.glob(f'*{ext}'))
            raw_images.extend(raw_path.glob(f'*{ext.upper()}'))
        
        raw_images = sorted(raw_images)
        
        print(f"\n--- {season.upper()} ---")
        print(f"Pronadjeno u raw: {len(raw_images)}")
        
        # UZMI SVE POSTOJECE FAJLOVE (osvezi svaki put)
        existing_files = list(processed_path.glob('*.jpg'))
        print(f"Već u processed: {len(existing_files)}")
        
        # Nadji sledeci slobodan broj
        if existing_files:
            numbers = []
            for f in existing_files:
                try:
                    num = int(f.stem.split('_')[0])
                    numbers.append(num)
                except:
                    pass
            next_number = max(numbers) + 1 if numbers else 1
        else:
            next_number = 1
        
        processed_count = 0
        
        for img_path in raw_images:
            # Proveri da li vec postoji (po originalnom imenu)
            already_exists = False
            for existing in existing_files:
                # Izvuci originalno ime iz "001_ana.jpg" -> "ana"
                parts = existing.stem.split('_', 1)
                if len(parts) > 1:
                    original_name = parts[1]
                    if original_name == img_path.stem:
                        already_exists = True
                        break
            
            if already_exists:
                print(f"  Preskačem (postoji): {img_path.name}")
                continue
            
            # Napravi novo ime
            new_filename = f"{next_number:03d}_{img_path.stem}.jpg"
            processed_file = processed_path / new_filename
            
            print(f"  Obradjujem: {img_path.name} -> {new_filename}")
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f" Greška pri učitavanju!")
                continue
            
            img_resized = cv2.resize(img, target_size)
            cv2.imwrite(str(processed_file), img_resized, 
                       [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            print(f"  Sačuvano")
            
            # Dodaje u listu postojecih da ne duplira u istom run-u
            existing_files.append(processed_file)
            next_number += 1
            processed_count += 1
        
        print(f"Obradjeno novo: {processed_count}")

if __name__ == "__main__":
    process_images()
    print("\n Sve obradjeno!")