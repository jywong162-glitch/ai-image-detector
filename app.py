"""
PERSON 4 OWNS THIS FILE.
Drag-and-drop demo: upload an image -> real/AI-generated + Grad-CAM heatmap,
plus a "Flag as AI-generated" button that turns RED when clicked.
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

FLAG_DEFAULT = "🚩 Flag as AI-generated"
FLAG_RED = "🚨 Flagged as AI-generated"


def predict(image):
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    prob_fake = float(probs[1])
    labels = {"real": float(probs[0]), "AI-generated": prob_fake}

    target_layer = model.features[-1]
    cam = generate_heatmap(model, x, target_layer)
    heatmap_img = overlay_heatmap(image.convert("RGB").resize((x.shape[3], x.shape[2])), cam)

    # If the model itself is above the threshold, pre-flag the button red.
    flagged = prob_fake >= config.THRESHOLD
    btn = gr.update(value=FLAG_RED if flagged else FLAG_DEFAULT,
                    variant="stop" if flagged else "secondary")
    return labels, heatmap_img, btn


def flag_as_ai():
    # Turn the button RED when the user clicks it.
    return gr.update(value=FLAG_RED, variant="stop")


with gr.Blocks(title="AI-Generated Image Detector") as demo:
    gr.Markdown(
        "# AI-Generated Image Detector\n"
        "Upload an image — the model estimates **real vs AI-generated** and highlights the "
        "regions it focused on (Grad-CAM). Click **Flag as AI-generated** to mark it (turns red)."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input image")
            analyze = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=2, label="Prediction")
            out_heat = gr.Image(type="pil", label="Grad-CAM heatmap")
            flag_btn = gr.Button(FLAG_DEFAULT, variant="secondary")

    analyze.click(predict, inputs=inp, outputs=[out_label, out_heat, flag_btn])
    flag_btn.click(flag_as_ai, outputs=flag_btn)


if __name__ == "__main__":
    demo.launch()
