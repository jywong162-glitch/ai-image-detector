# AI-Generated Image Detector — Track 5

Detect whether an image is a **real photo** or **AI-generated (AIGC)** — and stay
accurate even after the image is JPEG-compressed, blurred, resized, noised,
color-shifted, or cropped (the way images get mangled when shared online).

---

## The core idea (our whole approach in one line)
**We deliberately mangle training images (JPEG, blur, noise, crop, …) so the model
learns fingerprints that survive mangling.** That is why our detector stays robust
under real-world transformations instead of only working on clean lab images.

We ship **two complementary detectors**:

| Model | What it is | Strength | Heatmap |
|---|---|---|---|
| **`model_v3.pth`** (default) | EfficientNet-B0 (~5M params) fine-tuned end-to-end | Fast, robust to transforms, strong on known generators | **Grad-CAM** |
| **`model_clip.pth`** | Frozen CLIP ViT-L/14 + a trained linear head | Best at **unseen / photorealistic** generators | **CLIP attention rollout** |

Both stay well under the spec's 2B-parameter limit. See **`MODELS.md`** for the full
advantages/disadvantages of every model we trained.

---

## How it flows
```
 images ──▶ data.py ──▶ model.py ──▶ train.py ──▶ model_v3.pth / model_clip.pth
 (real/fake) (mangles &   (the brain)  (teaches it)          │
              feeds them)                                     ▼
                              ┌──────────────┬──────────────┬───────────────┐
                              ▼              ▼              ▼
                        evaluate.py     predict_dir.py     app.py
                        (robustness     (folder → JSON     (drag-drop
                         table)          of scores)         web demo + heatmap)
```

## Project structure
| File | Purpose |
|---|---|
| `config.py` | Shared settings (paths, image size, labels, `MODEL_ARCH`) + `get_device()` (auto NVIDIA/Mac/CPU) |
| `data.py` | Loads images + applies transforms (training augmentations + eval corruptions) |
| `model.py` | Defines EfficientNet-B0 **and** the CLIP detector; `load_model()` loads the weights |
| `train.py` | Fine-tunes a model and saves it (run on a GPU/Mac machine) |
| `evaluate.py` | Prints the **robustness table**: accuracy clean vs each transformation |
| `error_analysis.py` | TP/TN/FP/FN, precision/recall, confusion matrix, example mistakes |
| `predict_dir.py` | **Required deliverable**: image folder → JSON of `image_path` + `pred` |
| `gradcam.py` | Grad-CAM (EfficientNet) + attention-rollout heatmap (CLIP) |
| `app.py` | Gradio drag-and-drop web demo with the heatmap |
| `import_sid.py`, `import_genimage.py` | Stream + subset the training datasets |
| `run_app.bat` / `run_clip.bat` | Double-click launchers (EfficientNet / CLIP) |

---

## Setup

### Windows (PowerShell / VS Code terminal)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> Apple Silicon Macs (M1–M4) train on the GPU automatically via MPS — look for
> `device = mps` when `train.py` starts.

---

## Datasets we used
All datasets are public. `data.py` expects each dataset root to contain
`train/` and `test/`, each with `real/` (or `REAL/`) and `fake/` (or `FAKE/`).
Combine several at once with `DATA_ROOTS="data_sid,data_genimage" python train.py`.

| Dataset | Source | Role | How to fetch |
|---|---|---|---|
| **SID_Set** | HuggingFace `saberzl/SID_Set` | Real + fully-synthetic + **tampered** (partially AI-edited) images. Labels 1 & 2 → *fake*. | `python import_sid.py` |
| **Tiny-GenImage** | HuggingFace `TheKernel01/Tiny-GenImage` | Balanced real vs AI across **Midjourney, SD1.4/1.5, GLIDE, ADM, VQDM, Wukong, BigGAN** | `python import_genimage.py` |
| **CIFAKE** | Kaggle `birdy654/cifake-real-and-ai-generated-synthetic-images` | Used for the **early** models and the augmentation ablation only | Kaggle download → `data/` |

- **Final `model_v3.pth`** is a balanced all-rounder trained on **SID_Set + Tiny-GenImage**
  (equal images per dataset via `MAX_PER_ROOT`, CIFAKE dropped because its 32×32 images
  are out-of-domain for real photos). **`model_clip.pth`** uses the same data with a frozen
  CLIP encoder + linear head.
- The import scripts **stream** the datasets, so they never download the full multi-GB archive.

> **Do NOT train on the validation benchmark.** The COCO val2017 + DALL·E-Advanced subset is a
> reference benchmark only — keep it out of `train/` and `test/`.

---

## Run everything
```bash
python train.py                            # train a model (needs GPU/Mac ideally)
python evaluate.py                         # robustness table (clean vs each transform)
python error_analysis.py                   # TP/TN/FP/FN, metrics, confusion matrix
python predict_dir.py <img_dir> out.json   # required JSON output
python app.py                              # web demo (model_v3 + Grad-CAM)
```

### Choosing the model at runtime
- **EfficientNet (default):** `python app.py` — or double-click **`run_app.bat`**.
  Loads `model_v3.pth` and shows the **Grad-CAM** heatmap.
- **CLIP:** double-click **`run_clip.bat`** — or set the env vars yourself:
  ```bash
  MODEL_ARCH=clip MODEL_PATH=model_clip.pth python app.py
  ```
  Loads `model_clip.pth` and shows the **CLIP attention-rollout** heatmap.
  (Requires `open_clip_torch`, in `requirements.txt`; the CLIP weights download on first run.)

The same `MODEL_ARCH`/`MODEL_PATH` env vars work for `evaluate.py`, `error_analysis.py`,
and `predict_dir.py`.

---

## Results & documentation
- **`RESULTS.md`** — the robustness table (augmented vs baseline) and the error analysis.
- **`MODELS.md`** — every model we trained, with advantages/disadvantages and why we shipped two.
- **`DEMO_SCRIPT.md`** — narration script for the demo video.
- **`RUNPOD.md`** — how to train on a rented GPU, step by step.

---

## Conventions
- Labels: **`0 = real`, `1 = fake`**. `pred` in the JSON = probability of **fake/AIGC**.
- Images resized to **224×224**.
- Model must stay **< 2B parameters** (spec rule).
- **Never commit** `data/` (dataset folders are gitignored). Trained `*.pth` are committed
  intentionally so the demo and grader work out of the box.

## Limitations (honest)
- **Cross-generator gap:** EfficientNet reliably flags generators similar to its training data
  but misses some brand-new photorealistic generators — this is exactly why we added the CLIP
  model, which generalizes far better to unseen generators.
- **Heavy center-crop** (~87%) is the weakest transform because crop was not in the training
  augmentations. Adding random-crop augmentation is the clearest next improvement.
