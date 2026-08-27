"""
Shared model definition (Person 1 owns tuning; everyone imports build_model).
We fine-tune EfficientNet-B0 — a strong image model that already understands
pictures — and just swap its final layer for our real-vs-fake question.
"""
import torch
import torch.nn as nn
from torchvision import models
import config


def build_model(pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, config.NUM_CLASSES)
    return model


def load_model(device="cpu"):
    """Load trained weights if they exist; otherwise return a fresh model so
    Person 3 (eval) and Person 4 (demo) can build/test their code on day 1."""
    import os
    model = build_model(pretrained=True)
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        print(f"[model] loaded trained weights from {config.MODEL_PATH}")
    else:
        print("[model] no model.pth yet — using an UNTRAINED model (predictions are random).")
    return model.to(device).eval()
