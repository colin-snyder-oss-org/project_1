# scripts/train_model.py
import torch
from src.data_loader import get_dataloader
from src.models.combined_model import CombinedModel
from src.training.train import train_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataloader(
        batch_size=8,
        images_dir='data/processed/images/',
        masks_dir='data/processed/masks/'
    )
    model = CombinedModel(
        img_size=256,
        patch_size=16,
        num_classes=2,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024
    )
    train_model(model, dataloader, epochs=25, device=device)

if __name__ == "__main__":
    main()
