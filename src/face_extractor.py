import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision #Novi API za vizuelne zadatke (Face Landmarker)
from typing import Dict, Optional
import os

class FaceExtractor:
    """
    Klasa za izdvajanje ključnih regiona lica koristeći MediaPipe Face Landmarker 
    """
    
    def __init__(self, model_path='face_landmarker.task'):
        """
        Inicijalizacija Face Landmarker-a
        
        Args:
            model_path: Putanja do .task model fajla
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model fajl nije pronađen: {model_path}\n"
                f"Preuzmi ga sa: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
        #Kreira osnovne opcije
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False, #ne trebaju nam izrazi lica i matrice transformacije
            output_facial_transformation_matrixes=False
        )
        #kreira detektor sa onim sto smo podesili
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Indeksi landmarkova 
        #Definišu se indeksi landmark tačaka za različite delove lica
        # MediaPipe detektuje 468 tačaka na licu
        # Svaki indeks odgovara specifičnoj tački (npr. vrh nosa, uglovi očiju, itd.)
        
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 
                        386, 385, 384, 398]
        
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 
                         159, 160, 161, 246]
        
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 
                    269, 267, 0, 37, 39, 40, 185]
        
    def detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detekcija lica i ekstrakcija 468 landmarkova
        
        Args:
            image: BGR slika (OpenCV format)
            
        Returns:
            landmarks: Numpy array sa 468 landmarkova [x, y, z] ili None
        """
        # BGR -> RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Konverzija u MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detekcija
        results = self.detector.detect(mp_image)
        
        if not results.face_landmarks:
            return None
            
        # Konverzija u numpy array
        # vraca niz od 468 tačaka, svaka sa [x, y, z] koordinatama, ili None ako nije detektovano lice
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks[0]])
        return landmarks
    
    def get_skin_mask(self, image: np.ndarray, landmarks: np.ndarray, 
                 expansion_pixels: int = 15) -> np.ndarray:
    
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Kreiranje maske za lice (oval) 
        face_points = np.array([
            [int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)] 
            for idx in self.FACE_OVAL
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [face_points], 255)
        mask = cv2.erode(mask, np.ones((5,5),np.uint8), iterations=2)
        
        # Isključivanje usana 
        lips_points = np.array([
            [int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)] 
            for idx in self.LIPS if idx < len(landmarks)
        ], dtype=np.int32)
        
        if len(lips_points) > 0:
            lips_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(lips_mask, [lips_points], 255)
            lips_mask = cv2.dilate(lips_mask, np.ones((5,5),np.uint8), iterations=3)
            mask = cv2.subtract(mask, lips_mask)
        
        # Isključivanje očiju preko elipsi
        def draw_eye_ellipse(eye_points, scale_factor=1.8):
            """Crta elipsu oko oka na maski"""
            # Pronađe centar oka
            center_x = int(np.mean(eye_points[:, 0]) * w)
            center_y = int(np.mean(eye_points[:, 1]) * h)
            
            # Izračunava širinu i visinu oka
            eye_width = int((np.max(eye_points[:, 0]) - np.min(eye_points[:, 0])) * w * scale_factor)
            eye_height = int((np.max(eye_points[:, 1]) - np.min(eye_points[:, 1])) * h * scale_factor * 1.2)  # Visina malo veća
            
            # Osiguraj minimalne dimenzije
            eye_width = max(eye_width, 20)
            eye_height = max(eye_height, 15)
            
            # Nacrtaj belu elipsu na privremenoj maski
            eye_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(eye_mask, (center_x, center_y), (eye_width//2, eye_height//2), 0, 0, 360, 255, -1)
            
            return eye_mask
        
        # Levo oko
        left_eye_points_norm = np.array([landmarks[idx] for idx in self.LEFT_EYE])
        left_eye_mask = draw_eye_ellipse(left_eye_points_norm, scale_factor=1.8)
        
        # Desno oko
        right_eye_points_norm = np.array([landmarks[idx] for idx in self.RIGHT_EYE])
        right_eye_mask = draw_eye_ellipse(right_eye_points_norm, scale_factor=1.8)
        
        # Kombinuj maske očiju
        eyes_mask = cv2.bitwise_or(left_eye_mask, right_eye_mask)
        
        # Blago proširenje
        eyes_mask = cv2.dilate(eyes_mask, np.ones((5,5),np.uint8), iterations=2)
        
        # Isključi oči iz maske kože
        mask = cv2.subtract(mask, eyes_mask)
        
        # Blago proširenje maske kože
        mask = cv2.dilate(mask, np.ones((5,5),np.uint8), iterations=1)
        
        return mask
    
    
    def get_eye_regions(self, image: np.ndarray, landmarks: np.ndarray, 
                       padding: int = 5) -> Dict[str, np.ndarray]:
        """
        Izdvajanje regiona očiju
        """
        h, w = image.shape[:2]
        eyes = {}
        
        # Levo oko
        left_eye_points = np.array([
            [int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)] 
            for idx in self.LEFT_EYE
        ])
        x, y, w_eye, h_eye = cv2.boundingRect(left_eye_points) #pronalazi najmanji pravougaonik koji obuhvata sve tacke lica
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w, x + w_eye + padding), min(h, y + h_eye + padding)
        eyes['left_eye'] = image[y1:y2, x1:x2]
        
        # Desno oko
        right_eye_points = np.array([
            [int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)] 
            for idx in self.RIGHT_EYE
        ])
        x, y, w_eye, h_eye = cv2.boundingRect(right_eye_points)
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w, x + w_eye + padding), min(h, y + h_eye + padding)
        eyes['right_eye'] = image[y1:y2, x1:x2] #sece deo sa ocima i cuva
        
        return eyes
    
    def get_hair_region(self, image: np.ndarray, landmarks: np.ndarray,
                   forehead_expansion: float = 0.35) -> np.ndarray:
        """
        Kombinovani pristup za detekciju kose
        """
        h, w = image.shape[:2]
        
        # Geometrijska maska (prati oblik glave)
        forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
        forehead_points = np.array([
            [int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)] 
            for idx in forehead_indices if idx < len(landmarks)
        ], dtype=np.int32)
        
        if len(forehead_points) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        
        # Pronađi granice glave
        min_y_forehead = np.min(forehead_points[:, 1])
        max_y_forehead = np.max(forehead_points[:, 1])
        min_x_forehead = np.min(forehead_points[:, 0])
        max_x_forehead = np.max(forehead_points[:, 0])
        
        # Visina regiona kose - od vrha do čela
        hair_height = int((min_y_forehead) * forehead_expansion)
        top_y = max(0, min_y_forehead - hair_height)
        
        # Poboljšana geometrijska maska - prati oblik glave, nije pravougaonik
        geo_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Kreiraj poligon koji prati oblik glave
        # Uzmi tačke sa gornje strane glave (od leve slepoočnice do desne)
        left_temple_idx = 234  # leva slepoočnica
        right_temple_idx = 454  # desna slepoočnica
        
        left_temple = [int(landmarks[left_temple_idx][0] * w), int(landmarks[left_temple_idx][1] * h)]
        right_temple = [int(landmarks[right_temple_idx][0] * w), int(landmarks[right_temple_idx][1] * h)]
        
        # Proširi malo u stranu (podesavati)
        left_temple[0] = max(0, left_temple[0] - 22)
        right_temple[0] = min(w, right_temple[0] + 22)
        
        # Kreiraj poligon za kosu
        hair_polygon = np.array([
            [left_temple[0], top_y],  # gore levo
            [right_temple[0], top_y],  # gore desno
            [right_temple[0], right_temple[1]],  # desna slepoočnica
            [left_temple[0], left_temple[1]],  # leva slepoočnica
        ], dtype=np.int32)
        
        cv2.fillPoly(geo_mask, [hair_polygon], 255)
        
        #  Segmentacija po boji (ako imamo dobar uzorak)
        color_mask = np.zeros((h, w), dtype=np.uint8)
        hair_mask = geo_mask.copy()  # početna maska je geometrijska
        
        # Uzmi uzorak kose sa vrha glave
        sample_y = top_y + int((min_y_forehead - top_y) * 0.3)  # Malo iznad čela
        sample_x = w // 2
        sample_size = 20
        
        if (sample_y > sample_size and sample_y + sample_size < h and
            sample_x > sample_size and sample_x + sample_size < w):
            
            hair_sample = image[sample_y-sample_size:sample_y+sample_size,
                            sample_x-sample_size:sample_x+sample_size]
            
            # Proveri da li je uzorak dobar (da nije previše taman ili svetao)
            if np.mean(hair_sample) > 20 and np.mean(hair_sample) < 235:
                
                # Konvertuj u YCrCb za bolju segmentaciju boje kože/kose
                ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                ycrcb_sample = cv2.cvtColor(hair_sample, cv2.COLOR_BGR2YCrCb)
                
                # Izračunaj statističke parametre
                mean = np.mean(ycrcb_sample, axis=(0, 1))
                std = np.std(ycrcb_sample, axis=(0, 1))
                
                # Kreiraj masku na osnovu sličnosti boji (sa većom tolerancijom)
                lower = mean - 2.0 * std
                upper = mean + 2.0 * std
                
                # Ograniči vrednosti
                lower = np.maximum(lower, [0, 0, 0])
                upper = np.minimum(upper, [255, 255, 255])
                
                color_mask = cv2.inRange(ycrcb_image, lower, upper)
                
                # Kombinuj maske - uzmi presek geometrijske i color maske
                hair_mask = cv2.bitwise_and(geo_mask, color_mask)
                
                # Ako je color_mask previše prazna, vrati se na geometrijsku
                if cv2.countNonZero(hair_mask) < 500:
                    hair_mask = geo_mask.copy()
        
        #  Oduzmi lice (uvek radimo)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        face_points = np.array([
            [int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)] 
            for idx in self.FACE_OVAL
        ], dtype=np.int32)
        cv2.fillPoly(face_mask, [face_points], 255)
        
        # Veća dilatacija lica
        kernel_face = np.ones((15, 15), np.uint8)
        face_mask = cv2.dilate(face_mask, kernel_face, iterations=2)
        
        hair_mask = cv2.subtract(hair_mask, face_mask)
        
        #  Očisti masku
        kernel = np.ones((5, 5), np.uint8)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
        
        # 5. Ukloni male regione (samo ako imamo dobru masku)
        if cv2.countNonZero(hair_mask) > 1000:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_mask, connectivity=8)
            if num_labels > 1:
                # Pronađi najveći region
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                
                # Zadrži samo najveći region
                final_mask = np.zeros_like(hair_mask)
                final_mask[labels == largest_label] = 255
                hair_mask = final_mask
        
        return hair_mask
    
    def extract_features(self, image_path: str, visualize: bool = False) -> Optional[Dict]:
        """
        Glavna metoda za ekstrakciju svih regiona
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Greška pri učitavanju slike: {image_path}")
            return None
        
        landmarks = self.detect_face(image)
        if landmarks is None:
            print(f"Lice nije detektovano na slici: {image_path}")
            return None
        
        skin_mask = self.get_skin_mask(image, landmarks)
        hair_mask = self.get_hair_region(image, landmarks)
        eye_regions = self.get_eye_regions(image, landmarks)
        
        skin_pixels = image[skin_mask > 0]
        hair_pixels = image[hair_mask > 0]
        
        if len(skin_pixels) == 0 or len(hair_pixels) == 0:
            print(f"Nedovoljno piksela za analizu: {image_path}")
            return None
        
        result = {
            'image_path': image_path,
            'original_image': image,
            'landmarks': landmarks,
            'skin_mask': skin_mask,
            'hair_mask': hair_mask,
            'skin_pixels': skin_pixels,
            'hair_pixels': hair_pixels,
            'left_eye': eye_regions['left_eye'],
            'right_eye': eye_regions['right_eye']
        }
        
        if visualize:
            self.visualize_extraction(image, result) #kreira recnik sa izdvojenim podacima
        
        return result
    
    def visualize_extraction(self, image: np.ndarray, features: Dict):
        """
        Vizualizacija izdvojenih regiona
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(features['skin_mask'], cmap='gray')
        axes[0, 1].set_title('Maska kože')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(features['hair_mask'], cmap='gray')
        axes[0, 2].set_title('Maska kose')
        axes[0, 2].axis('off')
        
        skin_only = cv2.bitwise_and(image, image, mask=features['skin_mask'])
        axes[1, 0].imshow(cv2.cvtColor(skin_only, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Izdvojena koža')
        axes[1, 0].axis('off')
        
        hair_only = cv2.bitwise_and(image, image, mask=features['hair_mask'])
        axes[1, 1].imshow(cv2.cvtColor(hair_only, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Izdvojena kosa')
        axes[1, 1].axis('off')
        
        if features['left_eye'].size > 0 and features['right_eye'].size > 0:
            combined_eyes = np.hstack([features['left_eye'], features['right_eye']])
            axes[1, 2].imshow(cv2.cvtColor(combined_eyes, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title('Oči')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def close(self):
        """
        Zatvara detector i oslobađa resurse
        """
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()



if __name__ == "__main__":
    # Test
    extractor = FaceExtractor()
    print("FaceExtractor uspešno kreiran!")
    print("Spreman za upotrebu.")