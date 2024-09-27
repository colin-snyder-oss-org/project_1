# src/data_loader.py
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch

class MedicalImageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images_list = os.listdir(images_dir)
    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        img_name = self.images_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def get_dataloader(batch_size, images_dir, masks_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = MedicalImageDataset(images_dir, masks_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
