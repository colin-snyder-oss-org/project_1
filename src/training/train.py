# src/training/train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from src.utils.metrics import dice_coefficient

def train_model(model, dataloader, epochs, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
