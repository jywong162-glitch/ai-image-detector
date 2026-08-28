# Robustness Results & Error Analysis

**Task:** detect AI-generated vs real images, robustly under real-world transforms.
**Model:** EfficientNet-B0 (~5M params, well under the 2B limit), fine-tuned on CIFAKE.
**Key idea:** train with random transformations (JPEG, blur, resize, noise, color
jitter) so the detector stays accurate when images are mangled.

## Robustness table — baseline (no augmentation) vs ours (augmented)

| Transformation | Baseline (no aug) | **Ours (augmented)** | Gap |
|---|---|---|---|
| clean          | 98.54% | 98.52% | ~0 |
| jpeg_q90       | 95.80% | 98.53% | +2.7 |
| jpeg_q70       | 92.94% | 98.42% | +5.5 |
| jpeg_q50       | 90.72% | 98.39% | +7.7 |
| jpeg_q30       | 87.83% | 98.28% | +10.5 |
| blur_0.5       | 98.36% | 98.50% | +0.1 |
| blur_1.0       | 96.41% | 98.51% | +2.1 |
| blur_2.0       | 91.31% | 98.44% | +7.1 |
| resize_0.5     | 93.18% | 98.52% | +5.3 |
| resize_0.25    | 75.59% | 98.23% | +22.6 |
| noise_0.02     | 59.27% | 98.36% | +39.1 |
| noise_0.05     | 49.45% | 98.16% | +48.7 |
| noise_0.10     | 50.29% | 97.57% | +47.3 |
| color_jitter   | 96.55% | 98.25% | +1.7 |
| center_crop_80 | 85.92% | 87.22% | +1.3 |

Reproduce:
```bash
python evaluate.py                                 # ours (model.pth)
MODEL_PATH=model_baseline.pth python evaluate.py   # baseline
```

## Key findings
- **On clean images the two models are tied (~98.5%)** — augmentation costs nothing on easy data.
- **Under corruption the baseline collapses while ours stays flat (~98%).** The effect is
  largest exactly where it matters most in the real world (compression, downscaling, noise).
- **Gaussian noise is the headline:** the baseline falls to ~50% (random guessing), our model
  stays ~98% — a ~+47–49 point gap. This is the clearest evidence that augmentation, not luck,
  drives robustness.

## Error analysis
- **Weak spot — center crop (both ~86–87%).** Our augmentation set did NOT include cropping,
  so the model never learned crop invariance; both models degrade similarly. This is the honest
  limitation of the current approach.
- **False negatives** concentrate under aggressive crops and, for the baseline, under noise
  (where it essentially stops working).
- **Trade-off:** we intentionally trained on the same transform families we test on. This proves
  robustness to those transforms but is in-distribution (CIFAKE's own test set) — see limitations.

## Limitations & what we'd improve with more time
1. **Add random-crop augmentation** — directly targets the one weak spot (center-crop ~87%).
2. **Cross-dataset generalisation** — train/measure on SID_Set + WildFake and the official
   validation benchmark (COCO val2017 + DALL·E Advanced) to prove it works on *unseen* generators,
   not just CIFAKE. Code already supports this via `DATA_ROOTS`.
3. **Higher-resolution data** — CIFAKE is 32×32; real deployments see full-size images.
