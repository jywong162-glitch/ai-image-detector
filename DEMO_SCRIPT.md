# Demo Video Script (Track 5)

**Target length: 2.5–3 minutes.** The video's job is to make our *robustness story*
land: other detectors break when images get compressed/noised/resized online — ours
doesn't, and we can prove it. Lead with that; be honest about the cross-generator limit.

---

## [0:00–0:20] The problem
*Screen: title slide — "Robust AI-Image Detection Under Real-World Transformations" + team name.*

> "AI-generated images are everywhere — but detecting them is only half the problem.
> The moment an image gets shared, it's JPEG-compressed, resized, screenshotted, and
> noised. Most detectors are trained on clean images and fall apart under exactly these
> real-world transformations. Track 5 asks for a detector that stays robust. That's what
> we built."

## [0:20–0:45] Our approach in one line
*Screen: flow diagram, or `data.py` lines 79–89 (the augmentations).*

> "Our core idea is simple: we deliberately mangle every training image — JPEG
> compression, blur, resize, Gaussian noise, color jitter — so the model learns
> fingerprints that survive mangling. We fine-tune EfficientNet-B0, about 5 million
> parameters, well under the spec's 2-billion limit."

## [0:45–1:25] The proof (money shot)
*Screen: baseline-vs-ours table from RESULTS.md. Highlight the noise rows.*

> "Here's the proof. We trained two identical models — one on clean images, one with our
> augmentation — and tested both under every transformation. On clean images they tie at
> 98.5%, so robustness costs us nothing. But under corruption, the baseline collapses
> while ours stays flat. The clearest example is Gaussian noise: the baseline drops to
> 50% — literally random guessing — while ours holds at 98%. That's a 47-point gap, and
> it's proof this is augmentation working, not luck."

## [1:25–2:15] Live demo
*Screen: Gradio app (`python app.py`). Sequence:*
1. Real photo → "real" + Grad-CAM heatmap.
2. Clearly AI image → "AI-generated", button auto-flags red.
3. **Killer move:** same AI image, JPEG-compressed/noised → still detected.

> "Here's the detector in action. A real photo reads as real. An AI-generated image is
> flagged. And here's the key — even after heavy compression, it still catches it. The
> heatmap shows which regions drove the decision, so it's explainable, not a black box.
> And it outputs clean JSON for batch scoring — the required deliverable."

*Cut to terminal: `python predict_dir.py <folder> out.json`, show the JSON.*

## [2:15–2:45] Honest limitations (our differentiator)
*Screen: Error-analysis section of RESULTS.md, or a bullet slide.*

> "We also want to be honest about where it struggles. Like a human, our detector misses
> the newest photorealistic generators — Midjourney, DALL·E 3 — because they leave fewer
> artifacts than the generators in our training data. This cross-generator gap is a known
> open research problem. With more time, we'd add those generators to training and
> random-crop augmentation, which our config already supports."

## [2:45–3:00] Close
*Screen: repo URL + deliverables checklist.*

> "So: a robust, explainable, spec-compliant AI-image detector — proven under real-world
> transformations, and honest about its limits. Thanks for watching. Code and full
> results are on GitHub."

---

## Recording tips
- Record the demo run **separately** and get a clean, fast take. Start the app BEFORE
  recording so model-load lag doesn't show.
- Zoom the terminal/browser font up — judges watch small windows.
- Narrate over a screen recording; a talking-head intro/outro is optional.
- Hold on the table long enough to read; point the cursor at the noise rows.
- Tools: OBS Studio or Windows Game Bar (Win+G) to record; CapCut/Clipchamp to edit.
- Upload public/unlisted to YouTube, paste link on Devpost, test in an incognito window.
