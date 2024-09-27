# src/models/transformer_model.py
import torch
import torch.nn as nn
from einops import rearrange

class TransformerModel(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(TransformerModel, self).__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size

        self.patch_embeddings = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.transformer = nn.Transformer(dim, heads, depth)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        x = self.patch_embeddings(x)  # (batch_size, dim, num_patches^(1/2), num_patches^(1/2))
        x = x.flatten(2)  # Flatten height and width
        x = x.transpose(1, 2)  # (batch_size, num_patches, dim)
        x += self.position_embeddings
        x = self.transformer(x, x)
        x = self.mlp_head(x)
        return x
