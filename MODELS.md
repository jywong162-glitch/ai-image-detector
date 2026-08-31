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
| **`model_v3.pth`** | **EfficientNet-B0** | **balanced SID+GenImage+faces (no CIFAKE)** | **✅ SHIPPED (default)** | **our final model — fast, robust, reliable on real images** |
| `model_clip.pth` | frozen CLIP ViT-L/14 + linear head | same balanced data as v3 | ⚠️ experiment only | generalizes to unseen generators BUT over-flags real images → not shipped |

> How to run: `python app.py` (or `run_app.bat`) loads our shipped **model_v3**.
> `run_clip.bat` loads the experimental **model_clip** (kept for reference only — see its
> false-positive limitation below).

---

## The two models we ship

### `model_v3.pth` — EfficientNet-B0 (the app default)
Small CNN (~5M params) fine-tuned end-to-end on a **balanced** mix of SID + GenImage + face
datasets (CIFAKE dropped because it was too easy/synthetic and skewed results).

**Advantages**
- **Fast + tiny** (~16 MB, runs on CPU in real time) — great for a live demo.
- **Supports a gradient-based attention heatmap** — it has conv feature maps, so the app can
  show a heatmap of where the model "looked." Good for explainability in the write-up.
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
- **Over-flags real images as AI-generated (too many false positives).** This is the deal-breaker:
  in our testing the CLIP model marked a number of genuine real photos as fake. A detector that
  cries "AI!" on real images isn't trustworthy, so we did **not** ship it as the default. This is
  why `model_v3` is our final model and `model_clip` is kept only as a documented experiment.
- **No gradient-based heatmap** (it's a transformer with no conv feature maps) — but it *does*
  get an attention heatmap via **attention rollout** (`clip_attention_rollout` in `heatmap.py`),
  which reads CLIP's own attention to show where it looked. Note this shows encoder attention,
  not decision-specific attribution.
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

## Which model we shipped, and why

**We ship `model_v3` (EfficientNet).** It's fast, explainable, robust to real-world transforms,
and — most importantly — **reliable on real images** (few false positives).

We explored **`model_clip`** to close the cross-generator gap, and it *did* generalize better to
unseen photorealistic generators. But it **over-flagged real images as AI-generated**, and a
detector that mislabels real photos isn't trustworthy — so we kept `model_v3` as the default and
retained `model_clip` in the repo only as a **documented experiment**.

**What we'd do with more time:** calibrate the CLIP head's decision threshold, add more real
images to its training, and/or **ensemble** the two models (average their scores) to get CLIP's
generalization without the false positives. We ran out of time to land this, so we shipped the
model we trust.
