# AGENTS.md — instructions for AI coding assistants (Codex, etc.)

You are helping build a **Track 5 hackathon** project: an image classifier that
detects **AI-generated vs real** images and stays accurate **after real-world
transformations** (JPEG compression, blur, resize, noise, color jitter, crop).

Read `README.md` for the full picture. This file is the quick operating manual.

## First thing, every session
Run `git pull` before starting — the repo may be ahead of the local clone.

## What already exists (don't rebuild it)
A complete, runnable scaffold is in place:
- `config.py` — shared settings + `get_device()` (auto-selects cuda / mps / cpu)
- `data.py` — dataset loader + augmentations (train) + single corruptions (eval).
  The corruptions match the spec table exactly: JPEG 90/70/50/30, blur σ 0.5/1/2,
  resize 0.5/0.25, noise 0.02/0.05/0.10, color jitter ±20%, center-crop 80%.
- `model.py` — EfficientNet-B0 (~5M params); `load_model()` loads `model.pth` or a fresh model
- `train.py` — fine-tunes and saves `model.pth`
- `evaluate.py` — prints the robustness table
- `predict_dir.py` — REQUIRED deliverable: image dir → JSON of `image_path` + `pred`
- `app.py` — Gradio demo

## Hard rules (do NOT violate)
1. **Labels are fixed:** `0 = real`, `1 = fake`. `pred` = probability of fake/AIGC.
2. **Images are 224×224.** Import sizes/paths from `config.py` — never hardcode.
3. **Model must stay < 2B parameters** (spec constraint). EfficientNet-B0 is fine.
4. **Never commit `data/` or `*.pth`** — they're gitignored on purpose.
5. **Never train on `data/validation/`** (COCO val2017 + DALL·E Advanced). It is a
   benchmark only and must not leak into training.
6. Always select the device via `config.get_device()`, not hardcoded `"cuda"`.

## Common commands
```bash
# setup (macOS/Linux)
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# setup (Windows): python -m venv .venv; .venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt

python data.py                         # sanity-check the dataset loads
python train.py                        # train -> model.pth
python evaluate.py                     # robustness table
python predict_dir.py <img_dir> out.json
python app.py                          # demo
```

## Current status / good next tasks
- Scaffold verified running (Gradio demo works with an untrained model).
- **Blocking task:** download CIFAKE into `data/train/{real,fake}` and
  `data/test/{real,fake}`, then run `python train.py`.
- High-value improvements if asked: train a **no-augmentation baseline** too and
  add its row to the robustness table (shows why augmentation matters); add an
  error-analysis script that saves example false positives/negatives.

## Style
Match the existing simple, well-commented style. Keep shared config in `config.py`.
Prefer small, readable functions over cleverness — this is a hackathon prototype.
