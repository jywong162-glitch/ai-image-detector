# AI-Generated Image Detector — Hackathon Track 5

Detect whether an image is a **real photo** or **AI-generated (AIGC)** — and stay
accurate even after the image is JPEG-compressed, blurred, resized, noised,
color-shifted, or cropped (the way images get mangled when shared online).

> **Just cloned an older copy?** Run `git pull` first — the code and docs were updated
> (Apple Silicon support + these instructions).

---

## The core idea (our whole approach in one line)
**We deliberately mangle training images (JPEG, blur, noise, crop, …) so the model
learns fingerprints that survive mangling.** That's why our detector stays robust
under real-world transformations instead of only working on clean lab images.

We fine-tune **EfficientNet-B0** (~5M params, well under the spec's 2B-param limit).

---

## How it flows
```
 images ──▶ data.py ──▶ model.py ──▶ train.py ──▶ model.pth
 (real/fake) (mangles &   (the brain)  (teaches it)  (the trained brain)
              feeds them)                                   │
                                                            ▼
                              ┌──────────────┬──────────────┬───────────────┐
                              ▼              ▼              ▼
                        evaluate.py     predict_dir.py     app.py
                        (robustness     (folder → JSON     (drag-drop
                         table)          of scores)         web demo)
```

## Project structure
| File | Purpose |
|---|---|
| `config.py` | Shared settings (paths, image size, labels) + `get_device()` (auto NVIDIA/Mac/CPU) |
| `data.py` | Loads images + applies the transformations (training augmentations + eval corruptions) |
| `model.py` | Defines EfficientNet-B0; `load_model()` loads `model.pth` or a fresh model |
| `train.py` | Fine-tunes the model, saves `model.pth` (run on a GPU/Mac machine) |
| `evaluate.py` | Prints the **robustness table**: accuracy clean vs each transformation |
| `predict_dir.py` | **Required deliverable**: image folder → JSON of `image_path` + `pred` |
| `app.py` | Gradio drag-and-drop web demo (for the video) |
| `requirements.txt` | Python dependencies |

---

## Setup

### Windows (PowerShell / VS Code terminal)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m ensurepip --upgrade        # only if you get "No module named pip"
python -m pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> Apple Silicon Macs (M1–M4) train on the GPU automatically via MPS — no setup needed.
> Look for `device = mps` when `train.py` starts.

### Smoke test (works with NO data / NO trained model)
```bash
python app.py      # open the printed URL, drag in any image
```
Predictions will be **random** until a model is trained — that's expected. It only
proves the pipeline is wired correctly.

---

## Get the dataset
Download **CIFAKE** (Kaggle: "CIFAKE: Real and AI-Generated Synthetic Images") and
arrange it like this (folder names `real`/`fake` or `REAL`/`FAKE` both work):
```
data/train/real/   data/train/fake/
data/test/real/    data/test/fake/
```
Check it loaded: `python data.py`  (prints how many images were found).

Other allowed datasets: **SID_Set**, **WildFake** (HuggingFace / ModelScope links in the spec).

### ⚠️ Do NOT train on the validation set
The COCO val2017 + DALL·E-Advanced (WildFake) subset is a **reference benchmark only**.
Keep it out of `data/train` and `data/test`. Put it in `data/validation/` if you use it.

---

## Run everything
```bash
python train.py                       # produces model.pth (needs GPU/Mac ideally)
python evaluate.py                    # robustness table (clean vs each transform)
python predict_dir.py <img_dir> out.json   # required JSON output
python app.py                         # web demo
```
`model.pth` is **gitignored** (too big) — share it via Drive/Discord, drop it in the
project root, and every other script picks it up automatically.

---

## Conventions (don't change without telling the team)
- Labels: **`0 = real`, `1 = fake`**. `pred` in the JSON = probability of **fake/AIGC**.
- Images resized to **224×224**.
- Model must stay **< 2B parameters** (spec rule).
- **Never commit** `data/` or `*.pth` (already in `.gitignore`).

---

## Deliverables checklist (spec 5.5)
- [ ] Devpost write-up (approach, tools, models, libraries, datasets)
- [x] Public GitHub repo
- [x] `predict_dir.py` → JSON of `image_path` + `pred`
- [x] README (overview, setup, reproduce steps) — add limitations + team contributions before submitting
- [ ] Demo video (YouTube, public, linked on Devpost)
- [ ] Robustness table (clean vs transforms) → `evaluate.py`
- [ ] Error-analysis note (sample false positives/negatives + trade-offs)

## Judging weights
Technical 35% · Innovation 20% · Impact 20% · Feasibility 15% · Presentation 10%.
**Translation:** make it run reliably, and tell the robustness story clearly.

## Team ownership
| Person | Owns |
|---|---|
| 1 (GPU/Mac) | `train.py`, producing `model.pth` |
| 2 | `data.py` (datasets + augmentation) |
| 3 | `evaluate.py` (robustness table) |
| 4 | `app.py` (demo) |
| 5 | coordination, README, slides, video |
