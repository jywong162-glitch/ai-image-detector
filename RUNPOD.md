# Training on RunPod (RTX 5090) — step by step

Goal: rent a GPU for ~30 min, train `model.pth`, download it, shut the pod off.
Estimated cost: an RTX 5090 is roughly ~$0.7–1.0/hr, so a training run is well under $1.

---

## 1. Create the pod
1. Go to https://runpod.io → sign in → **Pods** → **Deploy**.
2. GPU: pick **RTX 5090**.
3. Template: choose an official **PyTorch** template (e.g. "RunPod PyTorch 2.x",
   CUDA 12.4+). This comes with CUDA drivers + PyTorch already installed — important,
   because the 5090 needs a recent CUDA build.
4. Disk: give it ~30–40 GB container/volume (CIFAKE + deps are small, this is headroom).
5. Deploy, then click **Connect → Start Web Terminal** (or use SSH if you prefer).

## 2. Get the code
```bash
git clone https://github.com/jywong162-glitch/ai-image-detector.git
cd ai-image-detector
pip install -r requirements.txt      # torch is already present; installs the rest
```

Verify the GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `True NVIDIA GeForce RTX 5090`.
If it prints `False` or errors with "no kernel image", the template's torch is too old
for the 5090 — install a CUDA 12.8 build:
```bash
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 3. Get the dataset (CIFAKE via Kaggle)
On kaggle.com: **Account → Settings → API → Create New Token** downloads `kaggle.json`.
Upload that file into the pod (drag it into the web file browser), then:
```bash
pip install kaggle
mkdir -p ~/.config/kaggle && mv kaggle.json ~/.config/kaggle/ && chmod 600 ~/.config/kaggle/kaggle.json
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
unzip -q cifake-real-and-ai-generated-synthetic-images.zip -d data
python data.py         # should print a large train/test image count
```
CIFAKE already unzips into `data/train/{REAL,FAKE}` and `data/test/{REAL,FAKE}` —
the loader handles the uppercase names automatically.

## 4. Train (in tmux so it survives a dropped connection)
```bash
tmux new -s train
BATCH_SIZE=256 EPOCHS=5 NUM_WORKERS=8 python train.py
```
- You should see `device = cuda`, the GPU name, and `mixed-precision: ON`.
- Watch `avg train loss` fall and `clean test accuracy` rise each epoch.
- Detach from tmux with **Ctrl+B then D**; reattach with `tmux attach -t train`.
- On a 5090, 5 epochs of CIFAKE is roughly a few minutes.

When it finishes you'll see: `** saved best -> model.pth (xx.xx%)`.

## 5. Check robustness (optional but this is your key result)
```bash
python evaluate.py     # prints accuracy: clean vs each transformation
```
Copy this table down — it's the centerpiece of your submission.

## 6. Get `model.pth` off the pod
Easiest (non-technical): in RunPod's **web file browser**, navigate to
`ai-image-detector/model.pth`, right-click → **Download**. It's ~20 MB.

Then on your laptop, drop `model.pth` into your local project folder. `app.py` and
`evaluate.py` will use it automatically.

(Alternative: `runpodctl send model.pth` on the pod prints a code; run
`runpodctl receive <code>` on your laptop.)

## 7. STOP THE POD
Back on the RunPod dashboard, **Stop** (or **Terminate**) the pod so you stop paying.
Don't skip this.

---

### Notes
- `data/` and `model.pth` are gitignored, so nothing large gets pushed to GitHub.
- Want a stronger "innovation" story? Run once with augmentations OFF (baseline) and
  once ON (yours), and compare the two robustness tables. The baseline collapsing
  under JPEG while yours holds is your headline result.
