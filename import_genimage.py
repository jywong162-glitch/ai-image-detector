"""
Import TheKernel01/Tiny-GenImage into data_genimage/{train,test}/{real,fake}.
Balanced real vs AI-generated (Midjourney, SD1.4/1.5, GLIDE, ADM, VQDM, Wukong,
BigGAN). Streams so it never caches the whole 8 GB at once.

Fields: 'image' (PIL), 'label' (0 = real, 1 = fake). HF splits: train, validation.

Usage:
  python -m pip install -U datasets
  python import_genimage.py            # import everything (~35k images)
  python import_genimage.py 8000       # cap N per class per split (optional)
"""
import os, sys
from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = all
if N <= 0:
    N = float("inf")

OUT = "data_genimage"
SPLITS = {"train": "train", "validation": "test"}   # HF split -> our folder


def save(img, split, cls, idx):
    d = os.path.join(OUT, split, cls)
    os.makedirs(d, exist_ok=True)
    img.convert("RGB").save(os.path.join(d, f"{idx}.jpg"), quality=95)


def main():
    for hf_split, out_split in SPLITS.items():
        stream = load_dataset("TheKernel01/Tiny-GenImage", split=hf_split, streaming=True)
        counts = {"real": 0, "fake": 0}
        for n, ex in enumerate(stream):
            if n == 0:
                print(f"[{hf_split}] example keys:", list(ex.keys()))
            if "image" not in ex:
                print("[import_genimage] ERROR: no 'image' key — paste the keys above to me.")
                return
            label = ex.get("label")
            cls = "real" if label == 0 else "fake" if label == 1 else None
            if cls is None or counts[cls] >= N:
                if counts["real"] >= N and counts["fake"] >= N:
                    break
                continue
            save(ex["image"], out_split, cls, counts[cls])
            counts[cls] += 1
            if sum(counts.values()) % 500 == 0:
                print(f"  [{out_split}] real={counts['real']} fake={counts['fake']}")
        print(f"[import_genimage] {hf_split} -> {out_split} done: {counts}")


if __name__ == "__main__":
    main()
