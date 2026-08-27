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
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 1e-4
