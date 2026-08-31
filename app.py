"""
PERSON 4 OWNS THIS FILE.
Drag-and-drop demo: upload an image -> real/AI-generated + attention heatmap,
plus a "Flag as AI-generated" button that turns RED when clicked.
Run:  python app.py   then open the local URL it prints.
"""
import torch
import gradio as gr

import config
from data import eval_transform
from model import load_model
from heatmap import generate_heatmap, overlay_heatmap, clip_attention_rollout

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

    # If the model itself is above the threshold, pre-flag the button red.
    flagged = prob_fake >= config.THRESHOLD
    btn = gr.update(value=FLAG_RED if flagged else FLAG_DEFAULT,
                    variant="stop" if flagged else "secondary")

    if hasattr(model, "features"):   # EfficientNet: gradient-based heatmap; CLIP: attention rollout
        target_layer = model.features[-1]
        cam = generate_heatmap(model, x, target_layer)
        heatmap_img = overlay_heatmap(image.convert("RGB").resize((x.shape[3], x.shape[2])), cam)
        predicted_class = "AI-generated" if int(probs.argmax()) == 1 else "real"
        heatmap_text = (
            f"### How to read this heatmap\n"
            f"The model predicted **{predicted_class}**. "
            f"**Red and yellow** areas had the strongest influence on that prediction, "
            f"**green** areas had some influence, and **blue** areas had little influence.\n\n"
            f"> The heatmap shows where the model focused. It does not prove that a specific "
            f"part of the image was created or edited by AI."
        )
    else:                            # CLIP model — attention-rollout heatmap
        predicted_class = "AI-generated" if int(probs.argmax()) == 1 else "real"
        cam = clip_attention_rollout(model, x)
        if cam is not None:
            heatmap_img = overlay_heatmap(
                image.convert("RGB").resize((x.shape[3], x.shape[2])), cam)
            heatmap_text = (
                f"### How to read this heatmap (CLIP attention)\n"
                f"The model predicted **{predicted_class}**. This is an **attention-rollout** "
                f"map: **red and yellow** areas are the image patches CLIP "
                f"attended to most, **green** some, **blue** little.\n\n"
                f"> It shows where the model looked. It does not prove that a specific part of "
                f"the image was created or edited by AI."
            )
        else:
            heatmap_img = None
            heatmap_text = "_Attention heatmap is not available for this CLIP build._"
    return labels, heatmap_img, heatmap_text, btn


def toggle_flag(current_label):
    # Toggle each click: normal (gray) <-> flagged (red).
    if current_label == FLAG_RED:
        return gr.update(value=FLAG_DEFAULT, variant="secondary")
    return gr.update(value=FLAG_RED, variant="stop")


def reset_all():
    # Clear prediction + heatmap and reset the flag button when the image
    # is cleared or a new one is uploaded — back to the initial state.
    return None, None, "", gr.update(value=FLAG_DEFAULT, variant="secondary")


TOOLTIP_CSS = """
#input-image button[aria-label] { position: relative; overflow: visible; }
#input-image button[aria-label]::after {
    content: attr(aria-label);
    position: absolute; bottom: 135%; left: 50%;
    background: rgba(20,20,20,0.95); color: #fff;
    padding: 5px 9px; border-radius: 6px; font-size: 12px;
    white-space: nowrap; pointer-events: none; z-index: 9999;
    opacity: 0; visibility: hidden;
    transform: translateX(-50%) translateY(4px) scale(0.96);
    /* smooth EXIT: fade/slide out, then hide after the fade finishes */
    transition: opacity 180ms ease, transform 180ms ease, visibility 0s linear 180ms;
}
#input-image button[aria-label]:hover::after {
    opacity: 1; visibility: visible;
    transform: translateX(-50%) translateY(0) scale(1);
    /* smooth ENTER: show immediately, fade/slide in */
    transition: opacity 180ms ease, transform 180ms ease, visibility 0s linear 0s;
}

/* box-name tooltips: hover the whole box. Positioned TOP-LEFT so they never
   collide with the upload/camera/clipboard button tooltips at the bottom. */
.tip-box { position: relative; }
.tip-box::before {
    position: absolute; top: 8px; left: 8px;
    background: rgba(20,20,20,0.95); color: #fff;
    padding: 5px 9px; border-radius: 6px; font-size: 12px;
    white-space: nowrap; pointer-events: none; z-index: 50;
    opacity: 0; visibility: hidden;
    transform: translateY(4px) scale(0.96);
    transition: opacity 180ms ease, transform 180ms ease, visibility 0s linear 180ms;
}
.tip-box:hover::before {
    opacity: 1; visibility: visible;
    transform: translateY(0) scale(1);
    transition: opacity 180ms ease, transform 180ms ease, visibility 0s linear 0s;
}
#input-image::before { content: "Input image"; }
#pred-box::before   { content: "Prediction"; }
#heat-box::before   { content: "Attention heatmap"; }
"""

with gr.Blocks(title="AI-Generated Image Detector") as demo:
    gr.Markdown(
        "# AI-Generated Image Detector\n"
        "Upload an image — the model estimates **real vs AI-generated** and highlights the "
        "regions it focused on (attention heatmap). Click **Flag as AI-generated** to mark it (turns red)."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", show_label=False, elem_id="input-image", elem_classes="tip-box")
            analyze = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=2, show_label=False, elem_id="pred-box", elem_classes="tip-box")
            out_heat = gr.Image(type="pil", show_label=False, elem_id="heat-box", elem_classes="tip-box")
            heatmap_guide = gr.Markdown()
            flag_btn = gr.Button(FLAG_DEFAULT, variant="secondary")

    analyze.click(predict, inputs=inp, outputs=[out_label, out_heat, heatmap_guide, flag_btn])
    flag_btn.click(toggle_flag, inputs=flag_btn, outputs=flag_btn)
    inp.clear(reset_all, outputs=[out_label, out_heat, heatmap_guide, flag_btn])
    inp.upload(reset_all, outputs=[out_label, out_heat, heatmap_guide, flag_btn])


if __name__ == "__main__":
    demo.launch(css=TOOLTIP_CSS)
