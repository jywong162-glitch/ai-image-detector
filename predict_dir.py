"""
REQUIRED DELIVERABLE (spec 5.5.2).
Takes an image DIRECTORY and writes a JSON file where each entry has:
  - "image_path": path to the image
  - "pred":       confidence in [0,1] that the image is AI-generated (AIGC)

Usage:
  python predict_dir.py <image_dir> [output.json]
Example:
  python predict_dir.py ./data/test/fake predictions.json
"""
import os, sys, json, glob
import torch

import config
from data import eval_transform
from model import load_model
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    if len(sys.argv) < 2:
        print("usage: python predict_dir.py <image_dir> [output.json]")
        sys.exit(1)
    image_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.json"

    device = config.get_device()
    model = load_model(device)
    transform = eval_transform()

    paths = [p for p in glob.glob(os.path.join(image_dir, "**", "*"), recursive=True)
             if p.lower().endswith(IMG_EXTS)]
    print(f"[predict] found {len(paths)} images under {image_dir}")

    results = []
    with torch.no_grad():
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"  skip {p}: {e}")
                continue
            x = transform(img).unsqueeze(0).to(device)
            prob_fake = torch.softmax(model(x), dim=1)[0, 1].item()  # index 1 = fake/AIGC
            results.append({"image_path": p, "pred": round(prob_fake, 6)})

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[predict] wrote {len(results)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
