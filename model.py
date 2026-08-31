"""
Shared model definition (Person 1 owns tuning; everyone imports build_model).
We fine-tune EfficientNet-B0 — a strong image model that already understands
pictures — and just swap its final layer for our real-vs-fake question.
"""
import torch
import torch.nn as nn
from torchvision import models
import config


class CLIPDetector(nn.Module):
    """Frozen CLIP image encoder + a trainable linear head.

    CLIP's features generalize across image generators much better than a
    from-scratch CNN, so this catches photorealistic fakes an EfficientNet misses.
    Only the head is trained; state_dict/load_state_dict handle ONLY the head, so
    the saved file is tiny and CLIP is reloaded fresh (pretrained) each time.
    """
    def __init__(self):
        super().__init__()
        import open_clip
        self.clip, _, _ = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        self.head = nn.Linear(self.clip.visual.output_dim, config.NUM_CLASSES)

    def forward(self, x):
        with torch.no_grad():
            feats = self.clip.encode_image(x).float()
        return self.head(feats)

    # Persist ONLY the trainable head (keeps the .pth tiny).
    def state_dict(self, *args, **kwargs):
        return self.head.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.head.load_state_dict(state_dict, *args, **kwargs)


def build_model(pretrained=True):
    if config.MODEL_ARCH == "clip":
        return CLIPDetector()
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
