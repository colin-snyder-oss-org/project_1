# src/models/cnn_model.py
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Assuming grayscale input
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
            # Add more layers as needed
        )
        self.decoder = nn.Sequential(
            # Corresponding decoder layers
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
