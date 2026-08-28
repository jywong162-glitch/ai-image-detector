"""
PERSON 1 OWNS THIS FILE (needs the GPU).
Fine-tunes the model with augmentations ON and saves model.pth.
Run:  python train.py
"""
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from data import get_dataloaders
from model import build_model


def main():
    device = config.get_device()
    print(f"[train] device = {device}  (cuda=NVIDIA, mps=Apple Silicon, cpu=no GPU)")

    train_dl, test_dl = get_dataloaders()
    model = build_model(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(config.EPOCHS):
        model.train()
        running = 0.0
        for imgs, labels in tqdm(train_dl, desc=f"epoch {epoch+1}/{config.EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"  avg train loss: {running / max(1, len(train_dl)):.4f}")

        # quick validation accuracy on the clean test set
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"  clean test accuracy: {100*correct/max(1,total):.2f}%")

    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"[train] saved -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
