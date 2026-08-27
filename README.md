# AI-Generated Image Detector (Track 5)

Detect **real vs AI-generated** images that stay accurate even after
JPEG compression, resizing, cropping, and blur.

## Who owns what
| File | Owner | Purpose |
|---|---|---|
| `data.py` | Person 2 | Load images + the transformation/augmentation code (secret sauce) |
| `train.py` | Person 1 (GPU) | Fine-tune the model, save `model.pth` |
| `evaluate.py` | Person 3 | Robustness table (accuracy per transformation) — the key slide |
| `app.py` | Person 4 | Gradio drag-and-drop demo |
| `model.py`, `config.py` | shared | Model + settings everyone imports |
| slides / repo / glue | Person 5 | Coordination + presentation |

## Setup (each person, once)
```bash
cd ai-image-detector
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
GPU person: install torch from https://pytorch.org for your CUDA version instead.

## Get the data (Person 2)
Download **CIFAKE** (Kaggle: "CIFAKE: Real and AI-Generated Synthetic Images")
and arrange it as:
```
data/train/real/  data/train/fake/
data/test/real/   data/test/fake/
```
Then check it:  `python data.py`  (prints how many images were found)

## Day-1 wins (do these in parallel, no waiting)
- Person 3: `python evaluate.py`  → prints a table using a RANDOM model (~50%). Wiring works.
- Person 4: `python app.py`        → demo runs with a random model. UI works.
- Person 1: `python train.py`      → once data exists, produces `model.pth`.
- Then Persons 3 & 4 automatically pick up the real model. Done.

## Required inference script (graded deliverable 5.5.2)
```bash
python predict_dir.py <image_dir> [output.json]
```
Outputs a JSON list of `{"image_path": ..., "pred": <AIGC confidence 0..1>}`.

## Labels
`0 = real`, `1 = fake`. Images resized to 224x224. Don't change these without telling Person 5.

## ⚠️ Do NOT train on the validation set
The COCO val2017 + DALL·E Advanced (WildFake) subset is a **reference benchmark only**.
Keep it in `data/validation/` and never point `train.py` at it. Train on CIFAKE /
SID_Set / (other) WildFake splits instead.

## Model-size rule
Spec requires **< 2B parameters**. EfficientNet-B0 (~5M) is safe. If anyone swaps in a
bigger backbone, check the param count stays under 2B.

## Official deliverables checklist (Person 5 tracks these)
- [ ] Devpost write-up (approach, tools, models, libraries, datasets)
- [ ] Public GitHub repo (this project, well-commented)
- [ ] `predict_dir.py` → JSON of image_path + pred  ✅ (built)
- [ ] README: overview, setup, reproduce steps, limitations, team contributions
- [ ] Demo video (YouTube, public, linked on Devpost)
- [ ] Robustness table: clean vs each transform  → `evaluate.py`
- [ ] Error-analysis note: sample false positives / false negatives + trade-offs

## What judges reward (weights)
Technical 35% · Innovation 20% · Impact 20% · Feasibility 15% · Presentation 10%.
Translation: **make it run reliably, and tell the robustness story clearly.**
