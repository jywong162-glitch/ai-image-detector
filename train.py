"""
PERSON 1 OWNS THIS FILE. Fine-tunes the detector and saves model.pth.

Auto-uses the best hardware (CUDA GPU > Apple GPU > CPU) and turns on
mixed-precision (AMP) on NVIDIA GPUs for a big speedup. Saves the BEST model
seen so far every epoch, so an interrupted run still leaves a usable model.pth.

Run:                python train.py
Tune without edits: BATCH_SIZE=256 EPOCHS=5 NUM_WORKERS=8 python train.py
"""
import os

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from data import get_dataloaders
from model import build_model


def main():
    device = config.get_device()
    amp_device = "cuda" if device == "cuda" else "cpu"
    use_amp = (device == "cuda")

    print(f"[train] device = {device}  (cuda=NVIDIA, mps=Apple Silicon, cpu=no GPU)")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True                 # faster fixed-size convs
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}  |  mixed-precision: ON")

    train_dl, test_dl = get_dataloaders()
    print(f"[train] batch_size={config.BATCH_SIZE}  workers={config.get_num_workers()}  "
          f"batches/epoch={len(train_dl)}  epochs={config.EPOCHS}")

    model = build_model(pretrained=True).to(device)

    # Continue training from an existing checkpoint (fine-tuning) if RESUME is set.
    #   RESUME=model_all.pth ... python train.py
    resume = os.environ.get("RESUME")
    if resume:
        model.load_state_dict(torch.load(resume, map_location=device))
        print(f"[train] RESUMED from checkpoint: {resume}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        running = 0.0
        for imgs, labels in tqdm(train_dl, desc=f"epoch {epoch+1}/{config.EPOCHS}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device, enabled=use_amp):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        print(f"  avg train loss: {running / max(1, len(train_dl)):.4f}")

        # validation on the clean test set
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast(amp_device, enabled=use_amp):
                    preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / max(1, total)
        print(f"  clean test accuracy: {acc:.2f}%")

        if acc >= best_acc:                                    # keep the best model
            best_acc = acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  ** saved best -> {config.MODEL_PATH} ({best_acc:.2f}%)")

    print(f"[train] done. best clean accuracy: {best_acc:.2f}%  |  model: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
