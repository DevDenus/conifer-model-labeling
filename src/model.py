import torch
from torch import nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights
)

def build_resnet_binary(
    model_name = "resnet50",
    freeze_backbone: bool = True,
    weights = None
) -> nn.Module:
    if model_name == "resnet50":
        weights = weights or ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
    elif model_name == "resnet34":
        weights = weights or ResNet34_Weights.IMAGENET1K_V1
        model = resnet34(weights=weights)
    else:
        weights = weights or ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 1)
    )

    if freeze_backbone:
        for p in model.fc.parameters():
            p.requires_grad = True

    return model

def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def unfreeze_for_finetune(model: nn.Module, unfreeze_all : bool = False, unfreeze_layer4 : bool = True):
    set_requires_grad(model.fc, True)
    if unfreeze_all:
        set_requires_grad(model, True)
    elif unfreeze_layer4 and hasattr(model, "layer4"):
        set_requires_grad(model.layer4, True)
