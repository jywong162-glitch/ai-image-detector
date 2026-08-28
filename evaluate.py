"""
PERSON 3 OWNS THIS FILE.
Produces the ROBUSTNESS TABLE — the centerpiece of the presentation.
Measures accuracy on clean images AND under each transformation.
Works even before a real model exists (uses a random model) so you can build
this on day 1.
Run:  python evaluate.py
"""
import torch
from torch.utils.data import DataLoader

import config
from data import ImageFolderDataset, eval_transform, get_corruptions
from model import load_model


def accuracy_under(model, device, corruption):
    ds = ImageFolderDataset(config.TEST_DIR, eval_transform(corruption))
    dl = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in dl:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / max(1, total)


def main():
    device = config.get_device()
    model = load_model(device)

    print("\n=== ROBUSTNESS TABLE ===")
    print(f"{'transformation':<16} | accuracy")
    print("-" * 30)
    for name, corruption in get_corruptions().items():
        acc = accuracy_under(model, device, corruption)
        print(f"{name:<16} | {acc:6.2f}%")


if __name__ == "__main__":
    main()
