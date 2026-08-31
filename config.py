"""Shared settings. EVERYONE imports from here so our files stay compatible."""
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# --- Data layout (Person 2 fills these folders) ---
# Train on ONE or MANY datasets. Each dataset root must contain train/ and test/,
# and each of those a real/ (or REAL/) and fake/ (or FAKE/) subfolder.
# Combine datasets for better generalization (comma-separated, no spaces needed):
#   DATA_ROOTS="data,data_sid,data_wildfake" python train.py
def _roots():
    out = []
    for r in os.environ.get("DATA_ROOTS", "data").split(","):
        r = r.strip()
        if r:
            out.append(r if os.path.isabs(r) else os.path.join(BASE, r))
    return out

DATA_ROOTS = _roots()
TRAIN_DIRS = [os.path.join(r, "train") for r in DATA_ROOTS]
TEST_DIRS  = [os.path.join(r, "test")  for r in DATA_ROOTS]

# Back-compat single-folder aliases (first dataset)
DATA_DIR  = DATA_ROOTS[0]
TRAIN_DIR = TRAIN_DIRS[0]
TEST_DIR  = TEST_DIRS[0]

# --- Model file (Person 1 produces this) ---
# Override which weights to load/save, e.g.  MODEL_PATH=model_cifake.pth python evaluate.py
_mp = os.environ.get("MODEL_PATH")
MODEL_PATH = (_mp if _mp and os.path.isabs(_mp)
              else os.path.join(BASE, _mp) if _mp
              else os.path.join(BASE, "model_v3.pth"))

# --- Image / label conventions (agreed by whole team) ---
IMG_SIZE    = 224          # pretrained models expect ~224x224
NUM_CLASSES = 2
CLASS_NAMES = ["real", "fake"]   # index 0 = real, 1 = fake

# Decision threshold: flag an image as AI-generated when P(fake) >= THRESHOLD.
# Lower it (e.g. 0.3) to catch more fakes at the cost of more false positives.
#   e.g.  THRESHOLD=0.3 python app.py
THRESHOLD = float(os.environ.get("THRESHOLD", 0.5))

# ImageNet normalization — pretrained models were trained with these numbers
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# --- Model architecture ---
# "efficientnet" (default) or "clip" (frozen CLIP encoder + linear head — far
# better cross-generator generalization for photorealistic fakes).
#   MODEL_ARCH=clip ... python train.py     (use the SAME env at inference too!)
MODEL_ARCH      = os.environ.get("MODEL_ARCH", "efficientnet")
CLIP_MODEL      = os.environ.get("CLIP_MODEL", "ViT-L-14")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "openai")

# CLIP uses its own normalization (different from ImageNet).
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def get_norm():
    """Return the (mean, std) matching the current architecture."""
    return (CLIP_MEAN, CLIP_STD) if MODEL_ARCH == "clip" else (MEAN, STD)

# --- Training knobs (override via env vars, no need to edit this file) ---
#   e.g.  BATCH_SIZE=256 EPOCHS=5 python train.py
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
EPOCHS     = int(os.environ.get("EPOCHS", 5))
LR         = float(os.environ.get("LR", 1e-4))

# Cap images per class PER DATASET so no single dataset dominates (0 = no cap).
# Use this to BALANCE datasets for an all-rounder, e.g. MAX_PER_ROOT=15000.
MAX_PER_ROOT = int(os.environ.get("MAX_PER_ROOT", 0))


def get_device():
    """Pick the best available accelerator: NVIDIA GPU > Apple Silicon GPU > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"                      # NVIDIA GPU (RunPod 5090, gaming laptop)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"                       # Apple Silicon GPU (M1/M2/M3/M4 Mac)
    return "cpu"                           # no GPU (slow but works)


def get_num_workers():
    """DataLoader worker processes. Override with NUM_WORKERS env var.
    Default: 8 on a CUDA box (feed the GPU fast), 0 on Mac/Windows (avoids hangs)."""
    env = os.environ.get("NUM_WORKERS")
    if env is not None:
        return int(env)
    return 8 if get_device() == "cuda" else 0


def use_pin_memory():
    return get_device() == "cuda"
