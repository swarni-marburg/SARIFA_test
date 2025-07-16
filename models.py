# models.py

from vit_model import VisionTransformer  # or your own implementation
from torchvision import models
import torch.nn as nn
from torchvision import transforms


def get_vitB_model(device):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=9,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        dropout=0.1
    ).to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    return model, transform

def get_resnet50_model(device):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 9)  # Adjust for 9 classes
    model = model.to(device)

    # Standard normalization for ResNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, transform

def get_resnet34_model(device):
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 9)  # Adjust for 9 classes
    model = model.to(device)

    # Standard normalization for ResNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model , transform