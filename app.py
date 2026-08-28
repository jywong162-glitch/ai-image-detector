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
from gradcam import generate_heatmap, overlay_heatmap

device = config.get_device()
model = load_model(device)
transform = eval_transform()

def predict(image):
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    labels = {"real": float(probs[0]), "AI-generated": float(probs[1])}

    target_layer = model.features[-1]
    cam = generate_heatmap(model, x, target_layer)
    heatmap_img = overlay_heatmap(image.convert("RGB").resize((x.shape[3], x.shape[2])), cam)

    return labels, heatmap_img


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Image(type="pil", label="Grad-CAM Heatmap")
    ],
    title="AI-Generated Image Detector",
    description="Upload an image. The model estimates whether it's real or fake and highlights relevant regions."
)

if __name__ == "__main__":
    demo.launch(share=True)
