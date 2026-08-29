"""
Download only a SUBSET of SID_Set by STREAMING it (never stores the whole
multi-GB dataset). Saves into data_sid/{train,test}/{real,fake} so it plugs
straight into DATA_ROOTS.

SID_Set labels (per the dataset card): 0 = real, 1 = fully synthetic,
2 = tampered (partially AI-edited). We map 0 -> real, and BOTH 1 and 2 -> fake,
so the model also learns to flag half-real / partially-edited images.

Usage:
  python -m pip install -U datasets
  python import_sid.py           # default 6000 images per class
  python import_sid.py 3000      # smaller, if disk/time is tight
"""
import os, sys
from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000   # images per class
if N <= 0:
    N = float("inf")                                  # pass 0 to import EVERYTHING
OUT = "data_sid"
EVERY_TEST = 10                                        # 1 in 10 -> test split


def save(img, split, cls, idx):
    d = os.path.join(OUT, split, cls)
    os.makedirs(d, exist_ok=True)
    img.convert("RGB").save(os.path.join(d, f"{idx}.jpg"), quality=95)


def main():
    stream = load_dataset("saberzl/SID_Set", split="train", streaming=True)
    counts = {"real": 0, "fake": 0}
    for n, ex in enumerate(stream):
        if n == 0:
            print("[import_sid] example keys:", list(ex.keys()))   # verify schema
        if "image" not in ex:
            print("[import_sid] ERROR: no 'image' key — paste the keys above to me.")
            return
        label = ex.get("label")
        # 0 = real, 1 = fully synthetic, 2 = tampered (partially AI-edited).
        # Treat BOTH 1 and 2 as "fake" so the model also learns to flag
        # half-real / partially-edited images, not just fully-generated ones.
        cls = "real" if label == 0 else "fake" if label in (1, 2) else None
        if cls is None or counts[cls] >= N:
            if counts["real"] >= N and counts["fake"] >= N:
                break
            continue
        split = "test" if counts[cls] % EVERY_TEST == 0 else "train"
        save(ex["image"], split, cls, counts[cls])
        counts[cls] += 1
        if sum(counts.values()) % 500 == 0:
            print(f"  saved real={counts['real']} fake={counts['fake']}")
    print(f"[import_sid] done: {counts}  ->  {OUT}/")


if __name__ == "__main__":
    main()
