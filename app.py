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
    prob_fake = float(probs[1])
    labels = {"real": float(probs[0]), "AI-generated": prob_fake}

    # Threshold decision: flag as AI-generated when confidence >= THRESHOLD.
    # Render a colored banner: RED when flagged, GREEN when real.
    box = ("padding:14px;border-radius:10px;font-size:22px;"
           "font-weight:700;text-align:center;border:2px solid")
    if prob_fake >= config.THRESHOLD:
        verdict = (f"<div style='background:#fdecea;color:#b71c1c;{box} #b71c1c'>"
                   f"🚨 FLAGGED: AI-generated &nbsp;({prob_fake:.1%} ≥ {config.THRESHOLD:.0%})</div>")
    else:
        verdict = (f"<div style='background:#e8f5e9;color:#1b5e20;{box} #1b5e20'>"
                   f"✅ Likely real &nbsp;({prob_fake:.1%} &lt; {config.THRESHOLD:.0%})</div>")

    target_layer = model.features[-1]
    cam = generate_heatmap(model, x, target_layer)
    heatmap_img = overlay_heatmap(image.convert("RGB").resize((x.shape[3], x.shape[2])), cam)

    return verdict, labels, heatmap_img


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.HTML(label="Verdict"),
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Image(type="pil", label="Grad-CAM Heatmap")
    ],
    title="AI-Generated Image Detector",
    description="Upload an image. The model estimates whether it's real or fake and highlights relevant regions."
)

if __name__ == "__main__":
    demo.launch(share=True)
