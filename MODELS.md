# Models — what we trained and the trade-offs

This project trained several detectors over the hackathon. Only **two** are shipped in
the repo today; the rest are recoverable from git history. This file explains what each
one was, and the **advantages / disadvantages** of each, so we can justify our final choice.

## TL;DR

| Model file | Arch | Trained on | In repo now? | Use it for |
|---|---|---|---|---|
| `model_baseline.pth` | EfficientNet-B0 | one small dataset | history only | first sanity check |
| `model.pth` | EfficientNet-B0 | CIFAKE + SID | history only | early "generalizes to real photos" model |
| `model_all.pth` | EfficientNet-B0 | all datasets combined | history only | broad but unbalanced |
| `model_v2.pth` | EfficientNet-B0 | SID+tampered / Tiny-GenImage / faces | history only | adds Midjourney + edited images |
| **`model_v3.pth`** | **EfficientNet-B0** | **balanced SID+GenImage+faces (no CIFAKE)** | **✅ default** | **the app default — fast + Grad-CAM** |
| **`model_clip.pth`** | **frozen CLIP ViT-L/14 + linear head** | **same balanced data as v3** | **✅** | **hard photorealistic fakes / unseen generators** |

> How to run each: default `python app.py` (or `run_app.bat`) loads **model_v3** with Grad-CAM.
> `run_clip.bat` loads **model_clip** (no Grad-CAM).

---

## The two models we ship

### `model_v3.pth` — EfficientNet-B0 (the app default)
Small CNN (~5M params) fine-tuned end-to-end on a **balanced** mix of SID + GenImage + face
datasets (CIFAKE dropped because it was too easy/synthetic and skewed results).

**Advantages**
- **Fast + tiny** (~16 MB, runs on CPU in real time) — great for a live demo.
- **Supports Grad-CAM** — it has conv feature maps, so the app can show a heatmap of where
  the model "looked." Good for explainability in the write-up.
- **Strong on generators it saw during training** and on real photos (low false positives).

**Disadvantages**
- **Weaker on photorealistic fakes from *unseen* generators.** A from-scratch CNN learns the
  fingerprints of the generators in its training set and doesn't transfer well to new ones —
  this is exactly the gap we saw on hard photorealistic images.

### `model_clip.pth` — frozen CLIP ViT-L/14 + linear head
CLIP's image encoder (~300M params) is **frozen**; we only train a small linear classifier
("linear probe") on top of its features. Trained on the **same balanced data** as v3 — so the
only difference vs v3 is the **architecture / features**, not the data.

**Advantages**
- **Best cross-generator generalization.** CLIP was pretrained on internet-scale images, so its
  features describe images in a general way rather than memorizing one generator's fingerprint.
  On our previously-missed photorealistic fakes, scores rose from ~0.005–0.36 (EfficientNet) to
  ~0.28–0.68 (CLIP) — one image flipped to correctly flagged, and the borderline ones would
  flag at a slightly lower threshold.
- **Tiny checkpoint** (~8 KB) — we only save the linear head; CLIP weights download separately.

**Disadvantages**
- **No Grad-CAM.** It's a transformer with no conv feature maps, so the heatmap is disabled.
- **Heavier to run.** Needs the `open_clip_torch` package and downloads the ~1.7 GB CLIP model
  on first run; ~15–20 s to load and slower per image than EfficientNet.
- Needs `MODEL_ARCH=clip` set at **both** train and inference time (that's what `run_clip.bat`
  handles) — forgetting it silently falls back to EfficientNet.

---

## Earlier models (git history only)

- **`model_baseline.pth`** — first EfficientNet trained on a single small dataset.
  *Advantage:* proved the pipeline works. *Disadvantage:* overfit; poor on anything else.
- **`model.pth`** — EfficientNet on **CIFAKE + SID**. First model that generalized to real
  photos. *Disadvantage:* CIFAKE is low-resolution/synthetic and inflated accuracy.
- **`model_all.pth`** — EfficientNet on **all datasets combined**. *Advantage:* broad coverage.
  *Disadvantage:* datasets were **unbalanced**, so large ones dominated and it skewed.
- **`model_v2.pth`** — EfficientNet fine-tuned to add **Tiny-GenImage (Midjourney)** and
  **tampered/edited** images. *Advantage:* catches partial AI edits + Midjourney.
  *Disadvantage:* still an EfficientNet — same unseen-generator gap as v3.

We introduced `MAX_PER_ROOT` (cap images per dataset) to fix the imbalance, which is how
`model_v3` became a cleaner, **balanced** all-rounder.

---

## Why we kept both v3 and CLIP

They're complementary:

- **model_v3** = fast, explainable (Grad-CAM), strong on known generators → best for the **demo**.
- **model_clip** = best at the **hard photorealistic / unseen-generator** cases the CNN misses →
  best for the **robustness story** the track is about.

A natural future step is an **ensemble** (average both scores) to get CLIP's generalization
*and* v3's speed/explainability, but each is usable on its own today.
