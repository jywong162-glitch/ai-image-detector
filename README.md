# AI-Generated Image Detector — Track 5

Detect whether an image is a **real photo** or **AI-generated (AIGC)** — and stay
accurate even after the image is JPEG-compressed, blurred, resized, noised,
color-shifted, or cropped (the way images get mangled when shared online).

---

## The core idea (our whole approach in one line)
**We deliberately mangle training images (JPEG, blur, noise, crop, …) so the model
learns fingerprints that survive mangling.** That is why our detector stays robust
under real-world transformations instead of only working on clean lab images.

## Our model

Our detector is **`model_v3.pth`** — an **EfficientNet-B0** (~5M params, well under the
spec's 2B limit) fine-tuned end-to-end on our data. It's fast, robust to real-world
transforms, and explainable via an attention heatmap. This is the default everywhere:
`app.py`, `predict_dir.py`, and `evaluate.py` all load it automatically.

> **Experimental — `model_clip.pth` (not the default).** We also built a research-y second
> detector: a **frozen CLIP ViT-L/14 encoder + a small trained linear head**. It generalizes
> better to *unseen* photorealistic generators, **but** we found it **over-flags real images as
> AI-generated** (too many false positives), so we did **not** ship it as the default. We keep it
> in the repo as a documented experiment — see `MODELS.md` for the full trade-off.

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
| `heatmap.py` | Attention heatmaps: gradient-based (EfficientNet) + attention rollout (CLIP) |
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
  are out-of-domain for real photos). The experimental `model_clip.pth` uses the same data
  with a frozen CLIP encoder + linear head.
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
python app.py                              # web demo (model_v3 + attention heatmap)
```

### Running the app
- **Default (our model):** `python app.py` — or double-click **`run_app.bat`**.
  Loads `model_v3.pth` and shows the attention heatmap. **This is what you should use.**
- **Experimental CLIP model (optional):** double-click **`run_clip.bat`** — or set the env vars
  yourself: `MODEL_ARCH=clip MODEL_PATH=model_clip.pth python app.py`. Note its known
  false-positive issue above; it's kept only for reference. (Requires `open_clip_torch`; the
  CLIP weights download on first run.)

---

## Results & documentation
- **`RESULTS.md`** — the robustness table (augmented vs baseline) and the error analysis.
- **`MODELS.md`** — every model we trained, with advantages/disadvantages and why we shipped `model_v3`.
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
  but can miss some brand-new photorealistic generators. We explored a CLIP-based model to close
  this gap; it generalized better to unseen generators **but over-flagged real images as fake**,
  so we kept the reliable EfficientNet as our shipped model and documented CLIP as an experiment.
- **Heavy center-crop** (~87%) is the weakest transform because crop was not in the training
  augmentations. Adding random-crop augmentation is the clearest next improvement.
