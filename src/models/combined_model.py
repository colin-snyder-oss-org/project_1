# src/models/combined_model.py
import torch
import torch.nn as nn
from .cnn_model import CNNModel
from .transformer_model import TransformerModel

class CombinedModel(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(CombinedModel, self).__init__()
        self.cnn = CNNModel()
        self.transformer = TransformerModel(img_size // 2, patch_size, num_classes, dim, depth, heads, mlp_dim)
        self.final_conv = nn.Conv2d(dim, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = self.cnn.encoder(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).view(x.size(0), -1, x.size(1) // x.size(0), x.size(1) // x.size(0))
        x = self.final_conv(x)
        return x
