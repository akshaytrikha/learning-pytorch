import torch
from torch import nn
from pathlib import Path


def save_model(model: nn.Module, model_name: str):
    # PyTorch expected file extension
    model_name += ".pth"
    model_path = Path(f"./models/{model_name}")

    torch.save(model.state_dict(), model_path)


def load_model(model_name: str) -> nn.Module:
    # PyTorch expected file extension
    model_name += ".pth"
    model_path = Path(f"./models/{model_name}")

    return torch.load(f=model_path)
