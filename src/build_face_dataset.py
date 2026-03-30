import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from face_extractor import FaceExtractor

SRC_DIR = Path("data")          
DST_DIR = Path("data_face")     
IMG_SIZE = 224

SPLITS = ["train", "val", "test"]
SEASONS = ["spring", "summer", "autumn", "winter"]

def crop_face_from_landmarks(img_bgr: np.ndarray, landmarks: np.ndarray, pad: float = 0.20):
    """
    landmarks: np.ndarray shape (468,3), normalized [x,y,z]
    returns: face crop resized to IMG_SIZE x IMG_SIZE (BGR) or None
    """
    h, w = img_bgr.shape[:2]
    xs = landmarks[:, 0] * w
    ys = landmarks[:, 1] * h

    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None

    # Expand bbox
    x1 = int(max(0, x1 - pad * bw))
    x2 = int(min(w - 1, x2 + pad * bw))
    y1 = int(max(0, y1 - pad * bh))
    y2 = int(min(h - 1, y2 + pad * bh))

    if x2 <= x1 or y2 <= y1:
        return None

    face = img_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return face

def main():
    DST_DIR.mkdir(exist_ok=True)

    extractor = FaceExtractor()

    total_in = 0
    total_ok = 0
    total_fail = 0

    for split in SPLITS:
        for season in SEASONS:
            in_dir = SRC_DIR / split / season
            out_dir = DST_DIR / split / season
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                continue

            images = [p for p in in_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            for img_path in tqdm(images, desc=f"{split}/{season}"):
                total_in += 1

                res = extractor.extract_features(str(img_path), visualize=False)
                if res is None:
                    total_fail += 1
                    continue

                face = crop_face_from_landmarks(res["original_image"], res["landmarks"], pad=0.20)
                if face is None:
                    total_fail += 1
                    continue

                out_path = out_dir / img_path.name
                cv2.imwrite(str(out_path), face)
                total_ok += 1

    extractor.close()

    print("\n=== DONE ===")
    print("Input images:", total_in)
    print("Saved crops :", total_ok)
    print("Failed      :", total_fail)
    print("Output dir  :", DST_DIR.resolve())

if __name__ == "__main__":
    main()