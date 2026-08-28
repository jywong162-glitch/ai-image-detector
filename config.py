"""Shared settings. EVERYONE imports from here so our files stay compatible."""
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# --- Data layout (Person 2 fills these folders) ---
DATA_DIR  = os.path.join(BASE, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")   # has real/ and fake/ subfolders
TEST_DIR  = os.path.join(DATA_DIR, "test")    # has real/ and fake/ subfolders

# --- Model file (Person 1 produces this) ---
MODEL_PATH = os.path.join(BASE, "model.pth")

# --- Image / label conventions (agreed by whole team) ---
IMG_SIZE    = 224          # pretrained models expect ~224x224
NUM_CLASSES = 2
CLASS_NAMES = ["real", "fake"]   # index 0 = real, 1 = fake

# ImageNet normalization — pretrained models were trained with these numbers
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# --- Training knobs (Person 1 tunes these) ---
BATCH_SIZE  = 64
EPOCHS      = 5
LR          = 1e-4
NUM_WORKERS = 0   # 0 = load data in main process. Keep 0 on macOS/Windows to avoid
                  # DataLoader hangs. On Linux with a big dataset you can try 2-4.


def get_device():
    """Pick the best available accelerator: NVIDIA GPU > Apple Silicon GPU > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"                      # NVIDIA GPU (gaming laptop)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"                       # Apple Silicon GPU (M1/M2/M3/M4 Mac)
    return "cpu"                           # no GPU (slow but works)
