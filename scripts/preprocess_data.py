# scripts/preprocess_data.py
import os
import cv2
from tqdm import tqdm

raw_data_dir = 'data/raw/'
processed_data_dir = 'data/processed/'

def preprocess():
    os.makedirs(processed_data_dir, exist_ok=True)
    images = os.listdir(raw_data_dir)
    for img_name in tqdm(images):
        img_path = os.path.join(raw_data_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Perform preprocessing steps (e.g., normalization, resizing)
        image = cv2.resize(image, (256, 256))
        cv2.imwrite(os.path.join(processed_data_dir, img_name), image)

if __name__ == "__main__":
    preprocess()
