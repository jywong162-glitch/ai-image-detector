"""
PERSON 4 OWNS THIS FILE.
A drag-and-drop demo: upload an image -> "AI-generated: 87%".
Works with a random model until Person 1 delivers model.pth.
Run:  python app.py   then open the local URL it prints.
"""
import torch
import gradio as gr

import config
from data import eval_transform
from model import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device)
transform = eval_transform()   # clean; you can add a corruption dropdown later


def predict(image):
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    # index 0 = real, 1 = fake  (see config.CLASS_NAMES)
    return {"real": float(probs[0]), "AI-generated": float(probs[1])}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="AI-Generated Image Detector",
    description="Upload an image. The model estimates whether it's a real photo or AI-generated.",
)

if __name__ == "__main__":
    demo.launch()
